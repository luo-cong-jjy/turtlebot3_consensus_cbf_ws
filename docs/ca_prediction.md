# 可切换 CA 障碍预测：2026-11-01 研究阶段

本阶段在 CV/frozen 保留分支上增加 `prediction_mode:=ca`，默认仍为 CV。CA 是已有的恒加速度 Kalman 模型，本身不作为新论文贡献。这里提供可复现的变速预测与闭环对照工具，为后续机制研究建立证据。

## 数据与模型

数据链仍为本机器人 LaserScan → 扫描时刻 TF → 聚类／椭圆拟合 → 预测位置关联 → Kalman 更新 → 控制时刻重对齐 → N+1 个未来椭圆 → MPC-D-CBF。没有共享原始雷达，没有用场景节点或 Gazebo 真值做估计。其他机器人的包络仍使用本机回波形状与通信里程计预测；本次 CA 改动针对外部障碍物跟踪器。

CA 状态为 `[cx,cy,a,b,phi,vx,vy,ax,ay]`，原 CV 未使用的最后两个状态用于加速度。每个方向的均值模型为

```
p_next = p + v*h + 0.5*acc*h²
v_next = v + acc*h
acc_next = acc
```

实际扫描时间差 `h` 决定传播，漏检继续传播，超过原超时窗口删除。测量仍为中心、半轴和方向，没有直接测量速度／加速度，也不读取给定场景轨迹。

加速度受连续白噪声 jerk 驱动。每轴 `[p,v,acc]` 的离散过程协方差为

```
Q(h) = qj * [[h^5/20, h^4/8, h^3/6],
             [h^4/8,  h^3/3, h²/2 ],
             [h^3/6,  h²/2,  h     ]]
```

两个方向独立。几何状态 a/b/phi 沿用参考项目随机游走；测量噪声、形状方差规则、角度模 π 更新和 Joseph 协方差更新保留。CV/frozen 继续使用原传播式和原 Q/P；加入前保存的 CV 样本用于逐项数值回归。

参数 `ca_jerk_density=0.01`（单位 m²/s⁵）及初始加速度方差 `ca_acceleration_variance=0.04`（m²/s⁴）在本轮评价前固定，没有按测试场景逐个调参。这是起始工程参数，不宣称已完成噪声辨识。

## 如何送入原控制器

- 每个预测节点同时计算中心、速度和协方差。原参考膨胀规则继续使用：`0.95*sqrt(max(Pxx,Pyy)) + 0.95*sqrt(max(Paa,Pbb))`；0.95 是乘数，不能写成“95% 安全概率”。
- CA 执行反应裕量逐节点为 `d0 + Tr*||v_j|| + 0.5*Tr²*||acc_est||`，均匀缩放椭圆，将该量加到既有距离屏障的裕量中。这只是估计运动下的工程缓冲，不是已证明的加速度上界或鲁棒安全保证。CV/frozen 的原常量裕量不变。
- 连续绕行软引导在预测移动坐标系中计算，CA 按各段预测位移恢复世界坐标速度；原 CV 仍用单一恒速恢复。软引导不是安全约束。
- 优化变量、原观测器、领导者参考／编队偏移、N=10、dt=0.1 s、速度边界、DCBF、IPOPT 和预算／兜底逻辑没有重构。没有新增硬控制增量约束。
- 名义编队误差、朝向误差、上一规划—执行偏差全部沿用 [schema 2](metrics_schema_v2.md)，没有通过更换公式制造改善。

因此闭环比较评价的是 CA 模型及其一致的下游预测使用；若要单独归因于模型、反应缓冲或移动引导，仍需拆分消融。

## 入口与检查

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rospack find tb3_distributed_control  # 应指向当前研究副本
roslaunch tb3_distributed_control single_turtlebot3_cbf_dynamic.launch \
  scenario:=accelerating_crossing prediction_mode:=ca total_time:=32

# 原 CV 对照；frozen 也继续保留
roslaunch tb3_distributed_control single_turtlebot3_cbf_dynamic.launch \
  scenario:=accelerating_crossing prediction_mode:=cv total_time:=32

# 减速、停车再起步；停走场景至少留出 40 s
roslaunch tb3_distributed_control single_turtlebot3_cbf_dynamic.launch \
  scenario:=stop_go_crossing prediction_mode:=ca total_time:=40

python3 -m unittest discover -s src/tb3_distributed_control/scripts/tests -p 'test_cbf*.py' -v
OPENBLAS_NUM_THREADS=1 python3 src/tb3_distributed_control/scripts/tests/compare_cbf_predictors.py \
  --seeds 10 --output /tmp/new_paired_predictor_replay
```

三个新场景位置／速度连续，加速度分段变化。速度上限不超过 0.22 m/s；并未因 CA 效果不佳而移动原障碍物。旧静态场景和已有五类动态场景均保留。

预测 JSON 新增加速度、各节点速度、位置协方差和执行裕量；CSV 新增 CA 启用标志与最大估计加速度，元数据记录 CA 参数。保留旧字段，不修改 ROS 消息定义。完整样本、失败、参数、源文件 SHA 和本地阶段快照应与结果一同保存。

## 评价边界

共同输入回放使用相同带时间抖动、噪声和漏检的模拟椭圆中心，6 类运动 × 2 档噪声 × 10 个种子 × 3 个模型。它隔离预测器差异，但没有模拟真实可见轮廓偏置、里程计误差或聚类误差。表中 RMSE 为每种子 RMSE 的平均，不是把所有样本当作独立重复。

闭环使用本机雷达和独立 Gazebo 实际位置评价，不能把不同机器人轨迹下的预测误差当作同输入模型消融。真实净距按约 20 Hz 采样及 0.13 m 机器人圆形代理计算，不构成连续时间安全证明。椭圆中心与物体真实中心的误差、未来位移误差需分别报告。

CA 在加减速段可能降低外推滞后，在恒速／静止或加速度突变之后可能放大噪声、过度外推，甚至预测反向运动；这不是识别了未来的停车意图。当前不包含 IMM、自适应模型切换、可证鲁棒安全、终端可行性或实物实验结论。

本轮通过与失败摘要见 [阶段验证](stages/ca_prediction_validation.json)。大体积数据继续独立归档，不上传原始实验数据或内部论文文件。
