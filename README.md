# TurtleBot3 分布式编队控制与 MPC-D-CBF

基于 ROS Noetic / Gazebo 的多机器人编队跟踪、避障及纯 Python 仿真工程。

当前已实现的是 **CV（恒速）动态障碍预测版本**：本机雷达检测与椭圆估计、目标关联、Kalman 状态估计、多步预测及不确定性膨胀、MPC-D-CBF 联合优化，并加入执行反应裕量和相对运动绕行引导。研究分支另有可切换的 CA 实验模式；IMM 尚未实现。

## 版本保留

| 引用 | 内容 |
|---|---|
| `main` | 当前可复现的 CV 基线及使用说明 |
| `research/geometry-time-2026-11-07` | 测量几何时标与模型外可行恢复候选；两开关默认关闭 |
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

## 在线可靠性预测候选与固定种子对照

新增 `prediction_mode:=adaptive`，使用本机历史预测的延迟误差评分，平滑选择 CV/CA/Singer 并保留模型分歧协方差。默认仍 CV，既有模式保留；不是 IMM 或安全概率估计。79 项检查通过，19 次预定配对／重复试验及完整负结果已归档。停走场景改善，加速／减速仍有恢复失败，不推荐替换默认。

[方法、参数与运行命令](docs/reliability_prediction_2026-11-01.md)、[完整研究报告](docs/stages/reliability_seeded_research_notes_2026-11-01.md)、[小体积验证记录](docs/stages/reliability_seeded_validation.json)。固定 Gazebo 种子不保证整个闭环逐位复现。

## 时变参考与恢复研究

新增受限运动预览、保留运动假设的参考选择和无效参考撤销；默认 `progress_guidance=false`。开启后 `progress_abstain=true` 表示无参考时撤销额外引导，由原名义 MPC 继续优化，不是直接停止。89项检查，两批34次冻结试验及5次开发完整保留。首批主试验6/6但重复出现负代理净距，修正批主试验5/6且仍有恢复／追越局限，不推荐作为默认实物方案。

[方法与运行命令](docs/progress_reference_2026-11-02.md)、[完整正反结果](docs/stages/progress_reference_research_notes_2026-11-02.md)、[验证摘要](docs/stages/progress_reference_validation.json)。

## 可行后备与局部绕行开发

研究分支新增默认关闭的 `execution_backup`：提前准备经原约束检查的后备控制，复用可行参考前缀初始化／后备，并加入短窗口绕行及方向保持退出条件。默认预测和原工程保留。这是效果改进候选，仍有恢复失败和偶发超时，不能当作稳定实物版本。

[运行方法与限制](docs/execution_backup_2026-11-04.md)、[全部开发结果（包含失败）](docs/stages/execution_backup_validation_2026-11-04.json)。本阶段优先排查闭环效果，没有将开发试验当作论文统计对比。

## 批量数值与参考预算改进

研究分支新增批量椭圆计算、预算内短窗口候选保留，以及独立默认关闭的 infeasible_escape。107 项数值检查、14 次开发完整保留；最终六次五过、三次五机代理净距为正，减速恢复仍失败，仍有尽力动作与超时。见 [方法、参数与全部正反结果](docs/bounded_preview_2026-11-04.md)。不是默认实物版本或安全证明。

## 参考生成与验收一致性修正

研究分支把已有各运动假设检查纳入局部参考生成，保留原接受条件。113 项数值检查；10 次尝试中 9 次完成、8 次通过，三次减速恢复至毫米级；仍有多机负净距、紧距离与超时，不能提升为默认实物版本。见 [方法和全部结果](docs/reference_projection_2026-11-05.md)。默认 CV 与全部研究开关状态保留。

## 相对运动预判实验与末值有效性

本轮相对预判／侧别锁定／运动补偿未稳定改善多机净距，保留为 predictive_bypass 默认关闭的实验；另修复部分候选索引异常与任务数据提前中断时的伪末值。121 项检查，18 次尝试的成功与失败全部保留，不能只选最新通过案例宣称安全。见 [完整记录](docs/relative_bypass_2026-11-05.md)。

## 测量几何时标与模型外恢复候选

新研究分支保持原滤波统计规则，增加可选的测量轮廓搬运和有界控制下的恢复速率选取，并修正初始化的并发／重试。135 项检查、构建通过，20 次尝试及所有失败保留；组合完成的10次均通过（含6次五机），但仍有求解失败／超时，尚非物理安全或实物结论。详见 [本阶段说明与运行方法](docs/geometry_time_recovery_2026-11-07.md)。原工程、旧分支和默认 CV 继续保留。

## 绕行质量复盘（2026-11-07）

当前研究诊断分支 `research/bypass-quality-2026-11-07` 新增默认关闭的 `active_guidance_only` 候选，保留原始后备，仅撤去无实际绕行参考的强引导。137项检查通过，但7次尝试中有2次原生启动崩溃、2次闭环负代理净距；**未接受为稳定改进**。同状态离线配对确认运动预测影响求解，但流畅绕行仍待解决；不推荐重新打开旧提前绕行组合。详见 [诊断与完整试验](docs/bypass_quality_2026-11-07.md)。

## 预测风险代价候选（2026-11-07）

`research/prediction-risk-2026-11-07` 加入默认关闭的预测风险软代价及独立引导关闭开关，可选软前视尾段默认0。148项检查通过，但减速场景恢复失败，**未通过整体效果改进验收，不推荐替换演示版**。8次尝试、含对照的5次通过及失败均保留，见 [实现与完整验证](docs/prediction_risk_2026-11-07.md)。
