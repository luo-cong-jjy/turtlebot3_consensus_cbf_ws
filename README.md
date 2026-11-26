最新`research/formation-intent-support-2026-11-26`加入12浮点/约5Hz的小参数避让请求与同侧支撑，不共享雷达。294检查，三类五机开启交换2/3通过（迎面/追越无停顿、无保护失败），斜向3/4号仍未过；相同本地参考关闭通信的迎面对照未过、末pair .650m。实际消息收发/参考和原当前硬约束审计通过，全部失败保留；旧入口默认关闭，独立formation_intent入口作为后续候选，不称稳定五机/实物。[实现、对照、限制与启动](docs/formation_intent_support_2026-11-26.md)。

本轮`research/peer-recovery-corridor-2026-11-26`完成移动车道/外移/回归候选及倒车参考兼容，286检查、两版各三场景严格均0/3，未接受替换。V2迎面末pair误差1.096m、追越退化，所有失败保留；旧后台入口默认关闭本项，保留853bfca父候选。下一步需研究小参数意图协调，尚未实现。[完整负结果、边界和复现](docs/peer_recovery_corridor_2026-11-26.md)。

最新本地后台求解候选`research/local-mpc-worker-2026-11-24`隔离各机参考预览/NLP并作一步延迟补偿，主循环用当前测量和原硬约束重验。277检查，V2五机三类仅追越通过，发布最大10.82/30.32/42.78ms；迎面回归邻机冲突4次保护停车，斜向仍停滞，不提升稳定版。原名义同步入口仍默认保留。[实现、全部结果及下一步](docs/local_mpc_worker_2026-11-24.md)。

# 当前候选主线：名义几何＋原参考不确定性

按用户意见，新增`five_turtlebot3_cbf_nominal_mainline.launch`，默认nominal，整帧99%仅`protection_mode:=scan_confidence`可选对照。原传感器σ=.01未关闭；保持原协方差膨胀、硬约束、目标/领导者观测，关闭新置信集合与预算/代价/调度试验，仅保留日志压缩。稀疏弧采用已知桶R=.12的名义圆心拟合，失败有工程补全padding，不宣称99%或硬噪声界。V1过保守保留未提升；V2六次同控制设置对照，名义通行较好，追越/迎面运动指标达原标准但仍有保护失败，斜向低速，严格两套各0/3仍保留，不作为论文成败标准。267检查，9次完整/两次启动崩溃全保留；17909首步/75805显示，45份15040扫描＋14315控制重放差0，V2标记帧未见原类漏包。本阶段是简化候选，不是稳定五机；下一步重计算隔离与协同绕行。见[完整数据、限制与启动](docs/nominal_mainline_2026-11-23.md)。

# 最新V5独立实测结论

用户恢复授权后，保持冻结d4807d5代码，4尝试含1次Gazebo控制前段错误，三类完整40s严格0/3。自动GC开启/阈值不变，本批15节点任务记录未出现二代GC，完整已保存行快照保留；仍有最高115.80ms指令墙钟，未解决整体超时。斜向无窗口故障但0号最低1s速度.07152<.075。6008采用首步/三指标等通过，15份5117扫描+4806控制重放差0；25404匹配显示检查通过但追越11个任务结束后标记未关联，全显示审计未过，不隐藏。V5不提升为稳定版。下一步隔离耗时感知/预览/NLP与发布，保留当前状态安全复核并修关停日志关联，尚未实现。见[完整报告](docs/compact_history_validation_2026-11-22.md)。

# 当前周期预算研究状态（2026-11-22）

263项单元检查通过，6060次旧校验器对照一致。V1–V4十次实测：V1 1/3、V2 0/3、V3 1/3且迎面恢复退化、V4诊断1/1；没有稳定解决超时，不提升为已完成版。GC诊断捕捉15–20ms二代暂停，但该次仍通过，不归因全部旧超时。V5仅压缩统计历史重复快照，last_records/磁盘完整、自动GC保持，263检查通过；Gazebo执行授权被拒绝，未启动，不能宣称闭环改善。十次19989采用首步/10指标等/86103显示复核及快照完整性通过，原控制/噪声/硬约束/100ms标准不放宽，所有失败保留。见[完整版本账本与待验证入口](docs/cycle_budget_2026-11-22.md)。

# 最新扫描噪声置信建模（2026-11-20）

新增独立noise_confidence入口；σ=.01、每帧alpha=.01、实际原始射线N及独立分辨率项计算范围，360射线时e≈.05691m；旧机制sigma0关闭，不代表物理无噪声。254检查，旧同输入943/2276条件超界变0、运动不变。新三类40s严格0/3：追越/迎面仅故障超时卡住，斜向仍停滞；6249帧实际N360/4820点簇置信元数据、540保护、5992首步/25151显示通过。15份5088扫描+4764控制预测差0，2286匹配簇及11次活动匹配保护未见必要条件超界。99%仅单机器人单帧原始距离噪声事件，不是任务/碰撞/定位/关联置信。原MPC、Q/R、限速、半径及协议保持，未提升为稳定五机。下一步先周期级100ms预算，再斜向通道协调。见[完整结果与复现](docs/bin012_scan_noise_confidence_2026-11-20.md)。

# 最新紧包络实验及噪声前提复核（2026-11-19）

新增独立`tight_initialization`入口，250有界数值检查通过；同输入496次保护预测面积平均减小6.5%（最大26.7%），运动不变，原18帧反例仍覆盖。4次尝试含1次Gazebo启动崩溃；3次完整五机严格1/3，斜向最长停住.847→.291s仍未达标准。6014首步/25287显示及15份5102扫描+4785控制重放通过。重要：Gazebo雷达Gaussian σ=.01不等于点误差硬界.01，2276匹配簇中943违反圆盘条件，其中22次活动匹配保护有2次违反。条件性数学正确不等于已证物理安全，此版本不提升为稳定/实物推荐；下一步先纠正噪声/置信建模，再协调邻机。旧源码和失败保留。见[完整报告及复现](docs/bin012_tight_center_guard_2026-11-19.md)。

# 最新初始化保护阶段（2026-11-19）

独立入口`five_turtlebot3_cbf_bin012_initialization.launch`，半径.12垃圾桶/R.4五机，MVEE新ID时以名义尺寸及显式误差界外包完整物体，不伪造位置测量、不改原硬约束或运动估计；通用默认关闭。246检查，15份旧同输入反事实消除18帧反例且运动值不变；三次新闭环1/3严格通过，迎面/斜向低速与超时仍保留、斜向停滞更明显。新15份5016扫描＋4751控制预测差0，标记窗口未见原类覆盖缺口；496保护/4786构造/6014首步/25217显示核对通过。不代表全程物理保证，误差界未实物标定，不能当稳定五机版。见[结果、方法假设与命令](docs/bin012_initialization_guard_2026-11-19.md)。

# 最新实物尺寸基准（2026-11-19）

独立入口`five_turtlebot3_cbf_bin012.launch`对齐半径.12m垃圾桶，保持R=.4五机、原算法/裕量/100ms截止，旧.18场景不覆盖。242检查，三类40s均完成但严格0/3通过；代理净距正/末pair误差≤.006767m不代表效果已稳定。5951首步/3指标/25307显示审计通过；15份预测重放差0。同时确认18帧新建目标MVEE未包含完整评估圆轮廓，需要优先处理未有可信圆时的尺寸先验/中心不确定性，再做预算和邻机协调。未测试实物或同型小车。见[完整记录、反例与启动命令](docs/five_robot_bin012_2026-11-19.md)。

# 当前五机验证状态（2026-11-19）

已按用户要求跳过0.8m双机，直接测R=.4正五边形。独立入口`five_turtlebot3_cbf_motion_geometry.launch`，239检查；4次真实40s试验仅1次严格通过（迎面），追越截止停车、斜向邻机冲突低速、迎面GUI复测低于速度门槛均保留。全部代理净距正、末pair误差≤.004323m不代表全过程丝滑或已稳定。8013首步/4指标/34655显示审计通过；原MPC内核/观测/误差和安全界限不改，不能称五机已完成。下一步邻机绕行协调与100ms预算。见[完整结果与复现命令](docs/five_robot_motion_geometry_2026-11-19.md)。

# TurtleBot3 分布式编队控制与 MPC-D-CBF

最新双机基础接入`research/two-robot-motion-geometry-2026-11-19`：两台本机雷达独立优化/原邻机状态互避，最新观测/回归/运行预算组合接入。236检查，6次新双机+单机回归通过，旧双机追越低速失败保留；6007首步/8指标/29484显示核对，4742控制预测重放差0。点簇仅各控制器本地文件，公开预测摘要无点，不增加机器人间原始雷达共享。R0.5/1m纵队基础场景，不是R0.4紧邻五边形/五机或实物验证。[全部结果、限制及新入口](docs/two_robot_motion_geometry_2026-11-19.md)。

最新失效处理`research/circle-prior-retention-2026-11-19`保留完整圆形状与最新测量的包含关系，低质量时仅传播运动/协方差，不按0.35 s切换参考点，原目标/扫描超时不延长。234检查、8次原闭环验收通过（6新/2旧），196先验包含/3200首步/8指标/20538显示审计通过。旧451同输入缺口消除，1.5 s仅离线质量屏蔽，不是现场故障保证；正常新旧精度非全提升，不宣称通用物理安全。[方法、全部结果与开关](docs/circle_prior_retention_2026-11-19.md)。

最新观测研究`research/geometry-motion-observer-2026-11-19`修正近圆柱局部轮廓参考点漂移，保留通用MVEE默认。229检查，8次有效闭环通过（5圆弧/3MVEE，含451和GUI）；速度向量RMSE三对约降88–90%。原控制/裕量不改，3456控制预测重放差0，4406首步及11份指标通过。补采1次截止失败、8次启动失败及451一次完整物体覆盖未认证均保留，不宣称全时刻安全或通用形状支持。[方法、全部结果和新入口](docs/geometry_motion_observer_2026-11-19.md)。

最新控制研究`research/qualified-return-2026-11-18`增加模型筛查的提前回归和3/4.5/6 s时程，原硬安全/0.22/限速保持。223检查，10次原验收通过（7新/3旧，含三类GUI）；配对XY稳定恢复提前1.55/3.47/3.52 s。侧偏拖尾减小，但峰值侧偏/角速度变化并非全面改善；保留99 ms样本和参考重选记录，不保证硬实时或新多机效果。[方法、全部结果和启动](docs/qualified_return_2026-11-18.md)。

最新显示诊断`research/prediction-direction-view-2026-11-18`分离淡绿安全区与彩色预测包络，增加真实中心点/速度与固定长度方向图标。218检查，两次真实GUI控制通过，4990标注/802首步审计通过。旧斜向样例原始几何中心速度已偏低，估计偏差和保守性未修复；不把显示改善当算法进步。[数据、含义与截图](docs/prediction_direction_diagnosis_2026-11-18.md)。

最新显示阶段`research/rviz-prediction-regions-2026-11-18`补上绿色MPC危险区域/黄色软触发预览，修复单机RViz配置选择和窗口布局。216项检查，四次GUI控制验收通过，首轮显示失败保留；4630区域/1601首步/四次指标核对通过。控制数学不变，不作为新算法或多机验证。[显示含义、实测截图位置与启动](docs/rviz_prediction_regions_2026-11-18.md)。

上一控制阶段为`research/single-trigger022-2026-11-18`：h≤0.22 m启用外部障碍辅助引导，最终2 s软预测提前量、2/3/4 s侧移，原N10硬DCBF始终有效，peer约束不改。213项检查，最终四次单机通过；三批17次及3次失败保留。不是固定物理间隙或更窄绕行保证。[定义、全部结果和启动](docs/single_trigger022_2026-11-18.md)。

上一研究阶段为`research/compact-formation-2026-11-17`：验证0.4m五边形与0.32m基础邻机边界兼容，五机静止雷达审计和最终空场景名义保持通过；不是五机动态新避障。204项检查；两档更小外部参考余量分别2/5、3/5通过，未接受替换，旧单机推荐保留。[全部成功、失败及初始化保护](docs/compact_formation_2026-11-17.md)。

上一阶段`research/single-repeatability-2026-11-17`：针对旧单机预览预算频繁撤销，加入原生参考计算图、当前模型重验衔接、短前缀保留及完成阶段预算预留。200项检查，最终单机七次预定测试通过；仅加速原型的截止失败和原V5配对失败均保留。不是新求解器或硬实时保证，旧入口默认不变。[新单机入口、完整结果与限制](docs/single_repeatability_2026-11-17.md)。

上一阶段`research/two-robot-pass-2026-11-12`接入双机CV直线动态避障，保留各自雷达/分布式观测与原互避。195项检查，最终团队4/4通过，仍是有限场景候选；该阶段发现的单机V5两次重复失败继续保留，不被后续成功覆盖。[双机阶段记录](docs/two_robot_pass_2026-11-12.md)。

上一阶段`research/spatial-pass-2026-11-12`新增提前侧移、保持侧别与包络更新后收缩参考，保留定时目标及原DCBF。186项检查；当轮单机三场景+追越复测4/4通过连续验收，配对基线仍停住；后续双机阶段发现单机重复性边界。存在误差/计算开销取舍，仅单机CV直线验证，默认spatial_pass=false。[该阶段记录](docs/spatial_pass_2026-11-12.md)。

上一阶段`research/wang-reference-2026-11-12`增加Wang最近直线参考机制对照，默认仍timed。179项检查通过，3次理想诊断（含对照）与2次Gazebo均完成，但连续运动验收0/5，未接受为效果改进；不是原论文完整复现。[完整正反结果与适配边界](docs/wang_reference_2026-11-12.md)。

上一阶段`research/active-scenes-2026-11-10`新增固定迎面、慢目标追越、斜向横穿三场景，以及独立连续运动验收。175项检查通过；四次Gazebo运行均完成，但当前控制器均未通过新的连续运动要求。旧算法与旧场景保留。[场景、结果与启动命令](docs/active_scenes_2026-11-10.md)。

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

## Wang核心对照与完整窗口诊断

`research/wang-core-2026-11-07` 提供论文中心距离代价/同禁止区域椭圆约束的单机核心对照（不是完整复现），保留原DCBF默认。159项检查、13次闭环及全部失败独立归档。完整25步无附加代价三次通过，但尚未稳定改善平滑性；默认仍10步。初值/后备共存候选与精确输入记录均默认关闭。见 [全部对照和局限](docs/wang_core_2026-11-07.md)。

## 单机CV后方通行候选

`research/pass-order-2026-11-08` 已完成老师静态版同条件回归与理想/真实雷达对照。新候选优先筛选限速、可达的后方参考首段，保留原MPC/DCBF。横穿三组配对位置误差和角速度变化降低，减速场景位置误差反而增加；默认关闭，尚未验证多机/实物。固定测试入口 `single_turtlebot3_cbf_cv_test.launch rear_pass:=true`，默认场景crossing；rear_pass:=false为配对的既有CV设置。详见 [实现与完整记录](docs/rear_pass_2026-11-08.md)。
