# 衰减加速度预测、引导消融与真实测量回放

本阶段是对照研究，默认仍为 `prediction_mode=cv`、`moving_frame_projection=false`。保留已有 CV/CA/frozen、静态版本和所有未通过记录。没有加入自动模型切换、IMM、新求解器或新的安全定理。实验结果见 [阶段摘要](stages/singer_ablation_validation.json)。

## 为什么继续做这一阶段

上一阶段 CA 在模拟中心的变速段比 CV 准，但恒速／静止段噪声更大；减速闭环的单次误差也较大。需要分清均值预测、协方差膨胀、反应缓冲、移动引导、绕行侧和有限速度恢复能力，不能用一次终端误差断定预测模型优劣。

因此先增加已有的 Singer 相关加速度模型、两个下游消融开关，以及可重放的原始拟合椭圆测量；再测试一个移动坐标系速度投影候选。这里的“原始测量”是聚类／拟合后的中心、半轴、方向和点数，不是原始 LaserScan，也没有向其他机器人共享扫描。

## Singer 模型与协方差

每轴 `[p,v,acc]` 满足 `p_dot=v, v_dot=acc, acc_dot=-acc/tau+sqrt(qj)*w`。与 CA 相比，未来加速度均值随时间衰减。状态转移为

```
e = exp(-h/tau)
F = [[1, h, tau*h - tau²*(1-e)],
     [0, 1, tau*(1-e)],
     [0, 0, e]]
```

过程协方差为 `qj * integral_0^h g(u)g(u)^T du`，其中
`g(u)=[tau*u-tau²*(1-exp(-u/tau)), tau*(1-exp(-u/tau)), exp(-u/tau)]^T`。

短区间用正权 Gauss–Legendre 求积，长区间用解析积分，避免小时间消去和大 h/tau 的边界层误差。独立矩阵指数／Lyapunov 方程验证均值、协方差、半正定性和分步传播一致性；tau 趋大时接近 CA。测量模型、几何噪声及原膨胀规则保留。没有只改均值而沿用错误的 CA 协方差。

本阶段固定 `tau=1 s, qj=0.01, acceleration_prior_variance=0.04`，没有按测试结果搜索最优 tau。用新的噪声种子 100–109 评价原有六类运动；这是未用于此前回放的噪声实现，不是六个全新未知场景。

Singer 是已有方法，参考 [MathWorks 的模型公式](https://www.mathworks.com/help/fusion/ref/singer.html)。采用它不自动构成论文创新。

## 消融与速度坐标

- `acceleration_buffer=false`：CA/Singer 恢复原 CV 形式的常量反应缓冲；不仅关掉加速度项，也关掉缓冲随预测节点速度变化。均值／协方差预测不变。
- `stage_velocity_guidance=false`：移动引导加回首段速度，代替逐段速度；预测和硬安全约束不变。该消融在 `moving_frame_projection=false` 下使用。
- `moving_frame_projection=true`：候选引导在每段相对坐标中使用平移后的速度可行球，再恢复世界速度。它需要逐段坐标一致性，因此采用逐段预测速度；不用于同时测试“首段速度”消融。

令 `w` 为障碍物该段速度、`u` 为机器人相对速度，则必要速度条件是 `||u+w||<=v_max`，而不是 `||u||<=v_max`。投影为

```
u_projected = project_ball(u+w, v_max) - w
```

例如 `u=[0.15,-0.20], w=[0,0.20]`，机器人世界速度是 `[0.15,0]`，应可保留，不能因相对速度范数 0.25 而先截断。

该候选同时放宽相对切向参考的幅度，并在平滑后重新做世界速度投影，使软引导的速度坐标一致。它只是平面速度的必要条件处理，不是非完整机器人完整可行域，也不是新增 QP/NLP 的硬约束。NLP 中已有的 v/omega 边界、动力学、DCBF、求解预算和兜底没有重写。坐标一致也不保证绕行场整体的收敛／性能，因此默认关闭，失败结果同样保留。

## 回放、指标与验证边界

`obstacle_predictions.jsonl` 新保存每帧拟合测量，包括空检测帧。`replay_cbf_measured_ellipses.py` 将同一批测量送给 CV/CA/Singer，滤波和预测完成后才读取 Gazebo 真值计分。报告预测覆盖率；没有测量记录的旧数据拒绝反推，不拿滤波后的中心冒充原始测量。

这是同一已录视角下的预测器对照，不是原始扫描重处理，也不是重新闭环运行。绝对中心误差与未来位移误差分开；椭圆可见轮廓中心不等于物体几何中心。所有路径、源文件 SHA、协议、未通过和兜底记录保存。

主误差公式沿用 schema 2。补充的“采样恢复时刻”是在误差峰值之后，末尾至少 2 s 的连续观测都不超过 0.05 m 的起始时刻；观测间隙超过 1.5 个控制周期或时间不足则记缺失。它是有限采样诊断，不是渐近／连续时间收敛证明。32 s 失败记录和 64 s 诊断重跑分开，不改验收终点或挑选最后一个最好样本。

CA 与 CV 的均值／协方差／预测包络回归样本、静态及动态引导回归样本均从独立旧阶段生成，确保默认算法数值保留；本阶段共 70 项检查通过。仅单机仿真验证，五机新模式和实物尚未验证。

## 使用

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rospack find tb3_distributed_control  # 必须是研究副本
roslaunch tb3_distributed_control single_turtlebot3_cbf_dynamic.launch \
  scenario:=decelerating_crossing prediction_mode:=singer singer_tau:=1.0 total_time:=32

# 消融：CA 常量缓冲，原引导
roslaunch tb3_distributed_control single_turtlebot3_cbf_dynamic.launch \
  scenario:=decelerating_crossing prediction_mode:=ca acceleration_buffer:=false total_time:=32

# 速度坐标候选：当前只作实验，不是推荐默认
roslaunch tb3_distributed_control single_turtlebot3_cbf_dynamic.launch \
  scenario:=accelerating_crossing prediction_mode:=singer moving_frame_projection:=true total_time:=32

OPENBLAS_NUM_THREADS=1 python3 src/tb3_distributed_control/scripts/tests/compare_cbf_predictors.py \
  --modes cv ca singer --seed-start 100 --seeds 10 --output /tmp/new_singer_replay
OPENBLAS_NUM_THREADS=1 python3 src/tb3_distributed_control/scripts/tests/replay_cbf_measured_ellipses.py \
  /path/to/new_gazebo_result --output /tmp/new_measured_replay
```
