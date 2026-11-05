# 相对运动预判实验与任务末值修正（2026-11-05）

本阶段从 `7ce5fd5` 继续，仅修改研究副本。原工程、main、CV 默认与此前阶段保留。源码／数据独立归档为 `阶段成果归档/2026-11-05_相对运动预判与侧别锁定`；原始输出在工作空间根目录 `四脚本对比证据/相对运动预判_2026-11-05`。

**本轮没有稳定解决多机负净距。** 新预判策略保留为显式实验选项 `predictive_bypass=false`，默认不启用。兼容运行保持此前参考策略；不能把最后一次通过的五机运行单独挑出来称为安全改进。

## 实验改动

此前失败前，ccw/cw 多次完全相同，且均未实际产生环绕作用：静态的“当前位置到目标”线段没有考虑障碍从后侧追近。实验增加约 2 s 的相对路径采样触发，使用已有 CV/CA/Singer 均值预测；超过原 MPC 窗口的半轴／方向保持最后节点，仅作为软引导近似。

机器人预想路径为有速度上限的直线，不是非完整机器人真正的 2 s 可行轨迹；后续仍用原 1 s 非完整模型与原验收检查。追近／横穿可以触发参考，单纯远离不触发。新规则仅在参考真正产生环绕且候选搜索完整时首次锁定方向；未锁定时轮换两侧计算顺序，已锁定且可行的方向可在部分搜索时继续保持。

V1 暴露一个引入的程序错误：轮换后只完成另一侧时，旧代码仍直接索引未计算的旧侧，导致控制定时器 KeyError 退出。错误源码、两次失败及回归输入保留；现已用可缺省查询修复，另有针对“只完成对侧”的回归检查。

V2 的闭环依旧出现负净距。V3 再将所选运动假设的节点位移率加入绕行速度：

```
v_world = v_relative_bypass + v_obstacle_prediction
```

世界速度限幅与原参考可行性检查随后执行。限幅后不保证还能实现原相对速度；假设预测也不是真值。这不是新增安全证明。静止障碍下该转换保持原参考。该实验没有改原 NLP 状态／变量、速度硬界、DCBF、求解器、领导者／观测器、编队偏移或场景位置。

多次五机运行仍有动态或机间负代理净距，新策略未形成稳定优势。V4 将其置于显式开关之后，默认关闭；完整实验代码继续保留。

## 修复任务末值缺失

首次定时器异常揭示旧导出器的问题：虽然任务要求 40 s，如果某机器人只记录到 9 s，也可能用这条旧数据标作“末端误差”。测试器的末时刻检查已使该试验失败，但导出统计仍可能误导。

现在声明了任务时长时，末端测量必须落在结束前 1.5 个控制周期内，否则 final/final_abs 为 NaN；不回填以前的有效误差。最新误差本身缺失也不回填。已记录样本的均值／RMSE计算保持，增加端点测量是否可用、年龄和样本时间，避免将有限样本误读为完整覆盖。

未声明任务结束时刻的历史导出保持“最后记录样本”的兼容行为，并标记终点合同未知。旧 CSV、PKL 和原统计不改写；本阶段汇总另外给出按新合同检查的任务末值。误差公式和 schema 2 的坐标／符号没有改变。

## 配置与保留

默认 CV、progress_guidance=false、execution_backup=false、infeasible_escape=false、predictive_bypass=false。开启新实验必须同时开启 progress_guidance、execution_backup 和实时求解。以下仅用于隔离 Gazebo 复查：

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rospack find tb3_distributed_control
roslaunch tb3_distributed_control multi_turtlebot3_consensus_gazebo_cbf_dynamic.launch \
  scenario:=mixed prediction_mode:=adaptive gazebo_seed:=211 total_time:=40 \
  progress_guidance:=true execution_backup:=true infeasible_escape:=false predictive_bypass:=true
```

V1–V3 是开关交付前的实验默认配置，各次源码与实际决策参数均保存；V4 提供默认关闭／显式开启的参数入口。旧 `align_reference_hypotheses` 与预算内参考仍保留。单独关闭 predictive_bypass 不会被描述成回到一个已证明安全的版本：此前候选也存在多机风险。

完整结果见 [阶段摘要](stages/relative_bypass_validation_2026-11-05.json)。这些是已看过种子上的开发测试，不是新颖性、统计显著性或物理安全验证。固定 Gazebo 种子不能固定 ROS／IPOPT 墙钟调度。

## 全部开发尝试

| 版本／场景／种子／角色 | 任务末误差最大（里程计 m） | 动态代理净距（m） | 机间代理净距（m） | 原场景验收 |
|---|---:|---:|---:|---|
| v4 / decelerating_crossing / 211 / compatibility | 0.0004 | 0.1511 | — | 通过 |
| v2 / accelerating_crossing / 101 / development | 0.0043 | 0.1807 | — | 通过 |
| v3 / accelerating_crossing / 101 / development | 0.0033 | 0.1714 | — | 通过 |
| v2 / decelerating_crossing / 211 / development | 0.0013 | 0.1445 | — | 通过 |
| v3 / decelerating_crossing / 211 / development | 0.0018 | 0.1366 | — | 通过 |
| v1 / mixed / 211 / development | 不可用 | 0.1016 | 0.0233 | 失败 |
| v2 / mixed / 211 / development | 0.0069 | -0.0449 | -0.0541 | 失败 |
| v3 / mixed / 211 / development | 0.0047 | -0.0428 | 0.0460 | 失败 |
| v1 / mixed / 239 / development | 不可用 | -0.0292 | 0.0425 | 失败 |
| v2 / mixed / 239 / development | 不可用 | 不可用 | 不可用 | 未启动 |
| v3 / mixed / 239 / development | 0.0047 | -0.0280 | 0.0871 | 失败 |
| v2 / overtaking / 211 / development | 0.0004 | 0.1555 | — | 通过 |
| v3 / overtaking / 211 / development | 0.0044 | 0.1703 | — | 通过 |
| v2 / stop_go_crossing / 211 / development | 0.0048 | 0.1252 | — | 通过 |
| v3 / stop_go_crossing / 211 / development | 0.0029 | 0.1584 | — | 通过 |
| v4 / mixed / 211 / explicit | 0.0042 | 0.0291 | 0.0047 | 通过 |
| v2 / mixed / 211 / repeat | 0.0043 | -0.0429 | -0.0171 | 失败 |
| v3 / mixed / 211 / repeat | 0.0079 | -0.0408 | 0.0608 | 失败 |

18 次尝试中 10 次通过原场景验收，1 次未进入任务阶段。V1 两次因候选异常导致部分控制定时器退出，任务末值不可用，不能列为正常完成的闭环。V2、V3 的多机负净距均保留。

V3 两次五机主测试及同种子重复均未通过；V4 最新显式开启时通过一次，但这不是对前述失败的替代或稳定性证明。默认关闭实验开关的兼容运行通过，任务末值可用性合同也已核查。

121 项数值检查通过；已记录行的误差公式／参考重建通过不代表整段控制流完整。回调健康与端点可用性检查明确标出 V1 的残缺运行。所有原评估、原统计、异常日志和独立源码继续保留。

后续应优先处理感知预测更新下的可行性与停止／恢复设计。本轮只尝试软引导预判，没有消除已有的一步不可行状态，也没有证明多机物理安全。
