# 参考生成与假设验收一致性（2026-11-05）

从 `977fdab` 继续，仍只改研究副本。没有修改原工程、main、旧阶段或默认 CV。原始数据在工作空间根目录 `四脚本对比证据/参考可行性一致性_2026-11-05`，独立快照、数据与 bundle 在 `阶段成果归档/2026-11-05_参考可行性一致性`。

## 本轮修正

此前减速试验约任务 12 s 的同一输入中，ccw 参考满足融合模型：最小 h 约 0.00889、最小 DCBF 残差约 0.00129；但 CA 均值假设最小 h 为 −0.002017。最后验收拒绝它并强制换成 cw，随后产生明显的回退／恢复迟延。

生成阶段原来只投影融合模型 DCBF，最后却要求全部保留假设的每个节点 h>=0。现在对局部 cw/ccw 的每一步有限控制网格，同步筛选已有的各假设下一节点 h>=−2e-6，再按原成本选择。中心、半轴、方向、假设集合、容差、原融合模型 DCBF 与最终验收条件都保留；没有丢弃低权重假设或给 MPC 增加逐假设硬约束。

同一冻结输入修正后 ccw 可通过原验收，CA 最小 h 约 +0.000469。成本从约 1.1588 到 1.1656，说明改的是生成的控制，不是验收阈值。对初始已进入假设包络的输入仍拒绝；启用的输入增量边界继续保留。单个反例的修复不证明所有参考都会可行或不再换侧。

该修正随 execution_backup 的研究参考链路启用；在启动时的 `/cbf/mpc/align_reference_hypotheses` 配置设 false 可保留旧投影。默认 progress_guidance、execution_backup、infeasible_escape 均仍关闭；领导者、观测器、编队偏移、原速度界、求解器、误差公式和任务窗口没有改变。

## 验证口径

最初两轮显式允许应急逃逸，但实际没有使用。随后关闭 infeasible_escape 做重复与场景复查。所有已完成运行均没有 best_effort_escape 命令。仍保留求解失败／超时停止，不能称每个实际命令都通过 DCBF 检查。

一次五机种子 211 测试出现两次 Gazebo Spawn service failed，未开始任务；原失败保留。按相同源码与参数原样重试，重试完成但出现负动态代理净距，仍按失败保留，不能用其他通过运行替换。

全部尝试、源码 SHA 与参数见 [阶段摘要](stages/reference_projection_validation_2026-11-05.json)。已看过的种子均为开发资料；同种子不冻结进程调度。约 20 Hz 圆形代理净距不是连续物理安全证明。

## 已明确的后续问题

减速跟踪恢复得到改善，多机安全仍不稳定。五机种子 239 的关闭应急选项重复虽然过原验收，最小代理净距仅约 1.7 mm。种子 211 重试出现约 −5.2 cm 动态代理净距，机器人 0 有 95 个 solver_failed_stop。不能只看毫米级末端误差宣称整体成功。

新增只读工具 `scripts/tests/diagnose_cbf_first_step.py`，不参与控制。对固定的下一节点椭圆，h 关于机器人位置的 Lipschitz 常数为 `L=max(1,b/a)`；单步位移不超过 `dt*(vmax+输入数值容差)`。因此任何控制都有：

```
h1(p_next) <= h1(p_hold) + L*dt*(vmax+2e-6)
```

若此上界仍低于原 DCBF 要求 `(1-gamma)*h0 - 2e-6`，则当前冻结模型的一步约束不可能满足，整个该次 MPC 也不可能可行。这是充分的模型不可行证据；没有证据只表示未知，不能推出可行，也不能推出物理上无路可走。

失败的五机重试共检查 203 个已记录决策帧，14 个满足上述条件。只检查了决策帧，没有把它等同于全部 95 次失败。证据使用控制器日志，不用真值。部分帧的 h0 已为负，前向不变性所需的初始集合条件也已不满足。

```bash
python3 src/tb3_distributed_control/scripts/tests/diagnose_cbf_first_step.py \
  /path/to/result --output /path/to/new_diagnostic.json
```

这些帧不能仅靠加速求解同一个冻结问题消除；后续需要处理提前规避、感知更新引起的集合变化与可行恢复／停止策略。本轮没有悄悄放松 DCBF 或提高硬速度上限，也没有将此前的尽力逃逸当作安全证明。

## 在研究副本复查

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rospack find tb3_distributed_control
roslaunch tb3_distributed_control single_turtlebot3_cbf_dynamic.launch \
  scenario:=decelerating_crossing prediction_mode:=adaptive gazebo_seed:=211 total_time:=32 \
  progress_guidance:=true execution_backup:=true infeasible_escape:=false
```

该版本保留为实验候选，尚不是默认实物版本。

## 全部尝试

| 场景／种子／角色 | 逃逸开关 | 最大里程计末端误差（m） | 动态代理净距（m） | 机间代理净距（m） | 验收 |
|---|---|---:|---:|---:|---|
| accelerating_crossing / 101 / development | False | 0.0049 | 0.1405 | — | 通过 |
| decelerating_crossing / 211 / development | True | 0.0013 | 0.1547 | — | 通过 |
| decelerating_crossing / 239 / development | False | 0.0033 | 0.1561 | — | 通过 |
| mixed / 211 / development | False | — | — | — | 启动失败 |
| mixed / 239 / development | True | 0.0039 | 0.1731 | 0.0701 | 通过 |
| overtaking / 211 / development | False | 0.0009 | 0.1335 | — | 通过 |
| stop_go_crossing / 211 / development | False | 0.0005 | 0.1417 | — | 通过 |
| mixed / 211 / infrastructure | False | 0.0068 | -0.0523 | 0.0582 | 失败 |
| decelerating_crossing / 211 / repeat | False | 0.0042 | 0.1503 | — | 通过 |
| mixed / 239 / repeat | False | 0.0072 | 0.0017 | 0.0542 | 通过 |

共 10 次尝试：9 次完成闭环、8 次通过场景验收；另一次启动失败完整保留。三次减速均在原 32 s 窗口内通过，末端约 1.3–4.2 mm。但五机 211 的重试失败，239 的关闭逃逸重复净距也很紧，不能称多机安全已解决。所有已完成运行均没有实际使用 best_effort_escape。

113 项数值检查通过。9 次已完成运行的全部机器人误差口径、机器人 0 参考重建与命令输入界／未使用尽力动作检查通过；启动失败不伪造审计或指标。原参考验收的容差 2e-6、输入界、任务窗口和场景阈值保持。审计通过不替代物理验收。
