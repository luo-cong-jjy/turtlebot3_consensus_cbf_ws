# TurtleBot3 分布式编队控制与 MPC-D-CBF

基于 ROS Noetic / Gazebo 的多机器人编队跟踪、避障及纯 Python 仿真工程。

当前已实现的是 **CV（恒速）动态障碍预测版本**：本机雷达检测与椭圆估计、目标关联、Kalman 状态估计、多步预测及不确定性膨胀、MPC-D-CBF 联合优化，并加入执行反应裕量和相对运动绕行引导。CA/IMM 尚未实现。

## 版本保留

| 引用 | 内容 |
|---|---|
| `main` | 当前可复现的 CV 基线及使用说明 |
| `research/ca-prediction` | 后续研究分支，目前仍为 CV 基线 |
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

CV 版本已完成 37 项数值／记录检查，以及单机穿越、迎面、追越、变速、双目标与五机闭环验证。[小体积验证摘要](docs/stages/cv_dynamic_validation.json)保留场景结果和来源说明。这些是有限场景的单次验证，不是普适安全或统计性能结论。

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
