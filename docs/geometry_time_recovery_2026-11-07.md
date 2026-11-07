# 测量几何时标与模型外恢复选取（2026-11-07）

当前在独立研究分支 `research/geometry-time-2026-11-07`，父提交 `9daa779`。原工程、main、旧研究分支与全部阶段不覆盖。源码／数据归档位于工作空间根目录 `阶段成果归档/2026-11-07_测量几何时标`；原始测试位于 `四脚本对比证据/测量几何时标_2026-11-07`。

论文工作仍在第一项框架的动态避障与工程闭环验证。新的感知一致性／可行恢复是研究候选，独立新颖性和理论论证未完成，新方案多机实物验证尚未完成。不能将有限仿真通过直接写成三项贡献均已成立。

## 先排除的误解

本工程 Burger 仿真 URDF 配置 odometrySource=world。旧失败案例中，按同一时刻插值的 Gazebo 位置与输入里程计约为毫米级差异；这是仿真 world 位姿反馈，不是验证真实轮式里程计抗漂移。没有为本轮修改此来源，也没有给控制器接入额外真值话题。

旧案例在约任务 10–16 s 连续停止，运动物体继续接近，部分时刻的模型一步约束已不可行。不能只看末端误差小或求解器耗时来判断安全。

## 1. 把测量轮廓与预测中心放到同一时刻

旧保护包络以控制时刻的滤波中心为中心，却要求包含最后扫描时刻的原轮廓，几何上同时覆盖两个时刻。旧方案作为保守约定保留；本轮新选项采用另一建模约定：按同一个运动模型搬运已测轮廓后再构造保护包络。

```
c_measurement_at_now = c_measurement_at_seen
                      + c_model_at_now - c_model_at_seen
```

保存的是最后一次匹配测量后的后验中心；漏检后的传播不改这个参照。融合模式用本次权重同时加权各模型的位移，避免权重变化本身被当作几何平移。协方差传播、模型评分／权重、原 0.95 膨胀规则和原执行裕量计算保留，不把 0.95 称作 95% 安全概率。

相同测量回放先核验配置下预测与保存的控制输入一致，再比较两个时标约定。旧失败案例验证了 93 条预测，启用搬运的新案例验证了 88 条，配置预测最大数值差均为 0。旧失败记录中，年龄约 0.6 s 时两约定的长半轴差约 24 cm、短半轴差约 9 cm；并非所有输入都会缩小，单元检查也覆盖了新包络扩大的情况。

保护包络仍包含按该估计平移搬运的完整测量椭圆。它不保证遮挡部分目标、未知加速度／转动或真实物体始终被覆盖，也不校准真实位姿误差。MPC 原 barrier 形式、输入界与目标未因时标选项改写。

## 2. 只在当前模型已 h0<0 时选恢复速率

原代码在当前模型越界时使用固定恢复 gamma=0.03，其他屏障用原 gamma=0.3。新独立选项 `adaptive_recovery` 在原恢复参数下有限网格没有可行点时，尝试降低负 h 屏障的恢复速率；当前非负 h 屏障的 gamma 完全保留。

负 h 的条件为：

```
h1 >= (1-gamma)*h0 = h0 + gamma*abs(h0)
```

在输入界及已启用的增量界内，对共同控制候选寻找所有负 h 屏障可承受的最大共同尺度 s，取 gamma_i=s*gamma_i_base，s 属于 [0,1]。正 h 的原约束也必须满足。候选计算留出数值余量，再用原一步检查器复核证据控制；没有模型非恶化候选就保留原问题，不设负 gamma 或伪造可行解。

这改变的是**模型外的恢复参数**，不是声称仍在求解原固定恢复参数问题。gamma=0 只要求不继续恶化，不保证有限时间回到安全集。证据只覆盖本次冻结模型的一步，不证明整段 MPC、递归可行或物理安全；后续感知集合更新仍可能破坏可行性。

没有新增优化变量或更换 IPOPT。每周期仍最多一次 NLP；实际解／后备按使用的参数验证。日志 recovery_selection 保存基本／实际 gamma、初始 h、证据控制与下一节点 h，不能把初始 h<0 时的恢复求解成功当作始终位于安全域。

## 3. 启动基础设施

多次失败没有进入任务：旧生成模型服务失败／ready 超时，另一次 Gazebo 原生段错误。它们均原样保留，不归因于控制算法，也不从尝试总数删除。

新增仅用于初始化的串行工厂锁、模型存在检查、有限重试；机器人传感器就绪后才放行下一次生成。初始位置、URDF、轨迹与控制目标不改，不删除或移动已存在模型。动态物体生成使用同一锁。该设置程序不向控制器提供模型位姿／速度；控制器仍使用原反馈。测试器对原生段错误更快报告；这些措施不等于修复了 Gazebo 内部所有崩溃。

## 默认和复查

CV 默认保留；transport_geometry、adaptive_recovery、predictive_bypass、infeasible_escape 均默认 false。本轮组合测试固定关闭此前不稳定预判与未通过 DCBF 的尽力逃逸。

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rospack find tb3_distributed_control  # 必须指向研究副本
roslaunch tb3_distributed_control multi_turtlebot3_consensus_gazebo_cbf_dynamic.launch \
  scenario:=mixed prediction_mode:=adaptive gazebo_seed:=211 total_time:=40 \
  progress_guidance:=true execution_backup:=true transport_geometry:=true adaptive_recovery:=true \
  predictive_bypass:=false infeasible_escape:=false
```

`prediction_mode=adaptive` 是 CV/CA/Singer 经验融合，与恢复参数选项是两个独立概念；本轮也单独检查组合在 CV 模式下的五机运行。

全部开关、源码 SHA、正反结果、参数选取及审计见 [阶段数据](stages/geometry_time_recovery_validation_2026-11-07.json)。单一参数选项也有通过运行，时标单独选项也有负净距，不把组合的正结果直接当作两个模块的因果或统计优越性证明。V3/V4 修正启动设施后另列，不与旧启动批次混成独立重复。

## 全部尝试

| 场景／种子／版本／角色 | 搬运 T | 恢复 R | 动态代理净距 m | 机间代理净距 m | 原场景验收 |
|---|---|---|---:|---:|---|
| mixed / 211 / v3 / baseline | False | False | -0.0304 | -0.0412 | 失败 |
| accelerating_crossing / 101 / v2 / development | True | True | 0.1720 | — | 通过 |
| decelerating_crossing / 211 / v2 / development | True | True | 0.1362 | — | 通过 |
| mixed / 211 / v1 / development | True | False | -0.0345 | 0.0723 | 失败 |
| mixed / 211 / v2 / development | False | False | -0.0280 | 0.1006 | 失败 |
| mixed / 211 / v2 / development | False | True | — | — | 启动失败 |
| mixed / 211 / v2 / development | True | False | 0.1725 | 0.0660 | 通过 |
| mixed / 211 / v2 / development | True | True | 0.0856 | 0.1006 | 通过 |
| mixed / 239 / v1 / development | True | False | 0.1209 | 0.1005 | 通过 |
| mixed / 239 / v2 / development | True | True | — | — | 启动失败 |
| overtaking / 211 / v2 / development | True | True | 0.1072 | — | 通过 |
| stop_go_crossing / 211 / v2 / development | True | True | 0.2221 | — | 通过 |
| mixed / 211 / v2 / infrastructure | False | True | 0.0407 | 0.1006 | 通过 |
| mixed / 211 / v3 / infrastructure | True | True | 0.1650 | 0.1006 | 通过 |
| mixed / 239 / v3 / infrastructure | True | True | 0.1650 | 0.0615 | 通过 |
| mixed / 239 / v4 / infrastructure | True | True | 0.0364 | 0.1006 | 通过 |
| mixed / 211 / v3 / mode | True | True | 0.1753 | 0.0615 | 通过 |
| mixed / 211 / v2 / repeat | True | True | — | — | 启动失败 |
| mixed / 211 / v3 / repeat | True | True | 0.1063 | 0.1005 | 通过 |
| mixed / 239 / v3 / repeat | True | True | — | — | 启动失败 |

本轮 20 次尝试，16 次进入并结束任务、13 次通过场景验收。4 次启动失败（含一次原生段错误）保留；基线与仅搬运方案的负净距也完整保留。

组合共有 13 次尝试，10 次完成任务均通过，其中 6 次五机（包含 CV 模式检查和两种子／重复）。组合已完成运行最小动态／机间代理净距约 3.6／6.1 cm。该条件性描述不能将启动失败排除后称为成功概率，更不是统计显著优势。仅搬运和仅恢复也有通过运行，当前不能严格归因各模块效应。

135 项数值检查、catkin 构建及 launch 解析通过。16 次已完成运行的全部机器人误差口径、机器人 0 参考重建、恢复证据代数和命令界检查通过；adaptive 模式的各观察者预测回放通过，CV 的混合权重检查不适用。证据检查不独立重建每个实际下发控制的全部屏障，更不证明连续物理安全。

通过运行仍存在求解失败／无可用恢复候选／超时停止。组合结果是在当前仿真反馈和采样代理下取得的进一步验证候选，默认开关继续关闭；实物、估计误差覆盖、感知更新下的递归可行性与停止策略仍待研究。
