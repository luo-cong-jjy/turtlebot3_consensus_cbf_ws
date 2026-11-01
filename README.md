# TurtleBot3 分布式编队控制与 MPC-D-CBF

基于 ROS Noetic / Gazebo 的多机器人编队跟踪、避障及纯 Python 仿真工程。

当前已实现的是 **CV（恒速）动态障碍预测版本**：本机雷达检测与椭圆估计、目标关联、Kalman 状态估计、多步预测及不确定性膨胀、MPC-D-CBF 联合优化，并加入执行反应裕量和相对运动绕行引导。研究分支另有可切换的 CA 实验模式；IMM 尚未实现。

## 版本保留

| 引用 | 内容 |
|---|---|
| `main` | 当前可复现的 CV 基线及使用说明 |
| `research/ca-prediction` | 评价口径统一及可切换 CA 研究；默认仍为 CV |
| `baseline/cv-dynamic-2026-10-24` | 已验证 CV 源码的初始保留标签 |
| `teacher/static-2026-10-20-r04` | 10 月 20 日静态 Gazebo，编队半径 0.4 m |
| `teacher/static-2026-10-21-r05` | 10 月 21 日静态 Gazebo，编队半径 0.5 m |

历史静态标签是有来源记录的恢复组装版本：关键控制器和优化器来自原始快照，部分通用依赖由保留工程补齐；0.5 m 辅助模块的默认半径依据已记录的单行差异恢复。不将整个目录宣称为当日完整工作站镜像。详见 [阶段说明](docs/stages/README.md)。

保留标签不覆盖。后续工作使用研究分支，实验结果应记录对应提交、参数和源码 SHA。

## 目录

- `src/tb3_distributed_control/scripts/`：ROS 控制器、领导者、场景与结果记录。
- `src/tb3_distributed_control/launch/`：静态、动态单机和多机入口。
- `src/tb3_distributed_control/纯仿真代码/`：纯 Python 仿真及共享求解核心。
- `src/tb3_distributed_control/config/`、`worlds/`、`rviz/`：场景参数、Gazebo 世界及显示配置。
- `src/MPC-D-CBF-main/`：参考项目源码。
- `src/turtlebot3-noetic-devel/`：TurtleBot3 相关源码与资产。

## 环境与构建

已验证环境为 Ubuntu 20.04 / ROS Noetic，Python 3。需要 Gazebo ROS、TurtleBot3 依赖，以及 NumPy、SciPy、Matplotlib、CasADi（IPOPT 后端）、PyYAML；OSQP 用于相关测试／对照，动图需要 Pillow。ROS/系统依赖未打包到 Git。

```bash
git clone https://github.com/luo-cong-jjy/turtlebot3_consensus_cbf_ws.git
cd turtlebot3_consensus_cbf_ws
source /opt/ros/noetic/setup.bash
catkin_make --only-pkg-with-deps tb3_distributed_control -j2
source devel/setup.bash
rospack find tb3_distributed_control
```

最后一行应指向当前克隆目录的 `src/tb3_distributed_control`，避免运行到其他工作空间。

## 启动

默认参数按 TurtleBot3 Burger 配置。首次对比建议使用单机器人动态场景。

```bash
# 静态环境、五机器人
roslaunch tb3_distributed_control multi_turtlebot3_consensus_gazebo_cbf.launch

# 单机动态穿越
roslaunch tb3_distributed_control single_turtlebot3_cbf_dynamic.launch scenario:=crossing

# 迎面／追越，留出完整恢复时间
roslaunch tb3_distributed_control single_turtlebot3_cbf_dynamic.launch scenario:=head_on total_time:=40
roslaunch tb3_distributed_control single_turtlebot3_cbf_dynamic.launch scenario:=overtaking total_time:=40

# 五机器人，两个动态目标
roslaunch tb3_distributed_control multi_turtlebot3_consensus_gazebo_cbf_dynamic.launch scenario:=mixed

# 预测消融：滤波仍运行，预测窗口中的障碍中心冻结
roslaunch tb3_distributed_control single_turtlebot3_cbf_dynamic.launch scenario:=crossing prediction_mode:=frozen
```

场景还包括 `maneuver`（温和变速／转向）。动态物体由独立 Gazebo 场景节点驱动；控制器不订阅它的真值或预设轨迹。

## 检查与实验记录

```bash
python3 -m unittest discover -s src/tb3_distributed_control/scripts/tests -p 'test_cbf*.py' -v
```

CV 基线版本完成 37 项数值／记录检查及单机穿越、迎面、追越、变速、双目标与五机闭环验证。本研究分支增加评价 schema 2 与 CA 检查后共 61 项检查通过。[小体积验证摘要](docs/stages/cv_dynamic_validation.json)保留场景结果和来源说明。这些是有限场景的单次验证，不是普适安全或统计性能结论。

默认实验输出在包内 `data_collect/<时间>-CBF-结果`；代码版本与结果应一并登记。原始 CSV、完整日志、图表、Git bundle 和完整工程归档继续保存在本地独立阶段目录，不纳入本次源码提交。请另行备份实验数据，不能只依赖 Git 标签。

## 查看历史静态版本

可以用独立 worktree 查看，避免覆盖当前研究：

```bash
git fetch origin --tags
git worktree add ../tb3_static_20261021 teacher/static-2026-10-21-r05
```

历史版本使用其自身 README 和构建方式；不应拿当前动态测试覆盖历史静态源码。

## 参考源码

动态预测和包络处理参考 [MPC-D-CBF](https://github.com/jianzhuozhuTHU/MPC-D-CBF)。第三方源码、模型资产和随附许可／作者信息保留在原目录中；不将参考算法声明为本工程原创。

## 当前研究分支：评价口径升级

本分支已完成 [metrics schema 2](docs/metrics_schema_v2.md)：统一测量时刻、目标减实际的朝向符号，分离名义编队误差、上一采用规划—执行偏差和对齐观测参考误差。缺失内部参考不填零，旧数据只在新目录中重算。控制／预测数学定义未变；研究分支另有可切换的 CA 实验模式；IMM 尚未实现。

main 和既有基线标签保留。阶段提交及独立本地归档用于追溯，不将本次评价调整当作新的控制性能提升。

## 当前研究分支：变速预测对照

已增加 `prediction_mode:=ca` 和 `accelerating_crossing`、`decelerating_crossing`、`stop_go_crossing` 三类动态场景。CV/frozen 默认与历史阶段保留；CA 不是全场景替换方案，也不单独作为原创贡献。模型、参数、指令、测试边界见 [CA 阶段说明](docs/ca_prediction.md)，有限验证结果见 [阶段数据](docs/stages/ca_prediction_validation.json)。

## 研究对照：衰减预测与引导消融

本分支增加 Singer 相关加速度模型、真实拟合测量共同输入回放和可关闭的缓冲／引导消融。速度坐标投影候选默认关闭，CV 仍为默认。70 项检查通过；14 次单机闭环有 3 次未通过终端跟踪要求，全部保留，不能当作全面性能提升。详见 [技术说明](docs/singer_ablation_2026-11-01.md)、[实验与局限](docs/stages/singer_ablation_research_notes_2026-11-01.md)。没有自动模型切换或新模式实物验证。
