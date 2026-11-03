# 时变绕行参考与恢复检查（阶段编号 2026-11-02）

本阶段研究动态障碍物引起的跟随／大绕行。默认仍 `progress_guidance=false`，原工程及全部旧版本保留。当前实现为可关闭的参考候选，不是全场景安全或收敛保证。两批验证分别冻结，失败没有被修正版覆盖；详见 [结果与局限](stages/progress_reference_research_notes_2026-11-02.md)。

## 为什么改参考决策

前阶段失败记录显示：机器人约 10–20 s 持续向障碍物运动方向偏移，guide_level 接近 1，很多周期求解成功。仅换预测器不能解决参考场产生的跟随平衡。单一融合均值也会忽略尚未获得高权重、但已经预警风险的机动假设。

新开关不改变观测器、名义目标、MPC 变量、速度边界、DCBF 或 IPOPT。原连续引导完整保留。开启后，在局部外部运动目标附近比较直接恢复、等待、正向环绕（ccw）和反向环绕（cw）四种参考。

## 预览与评分

1. 原 1 s MPC 窗口中的中心、轴和角度节点保持。额外的 4 s 软预览沿用各专家的均值模型；超过 MPC 窗口的轴长保持最后节点。它不是 4 s 鲁棒安全预测，不用于延长硬约束。
2. 已知目标窗口逐点保留；窗口之外仅按末端速度外推。没有用真值轨迹替换观测器／领导者参考。
3. 预览采用与后端相同的离散非完整运动学（先更新朝向，再更新位置），约束 v、omega 及已启用时的增量边界。circulation 仅表示环绕方向，不能按世界坐标简单称为“左／右侧”。
4. 对每条参考检查融合模型全预览的采样 h，以及原 MPC 窗口的原始 DCBF 不等式。启用 `progress_hypotheses` 时，生成与检查也保留各专家均值，包括权重为 0 的专家；使用相同的融合轴长，不是每个专家完整不确定集合的并集。
5. 只在通过这些模型检查的参考中评分：平均位置误差平方 + 2 倍末端误差平方 + 0.03 倍恢复时间代理 + 0.01 倍归一化角速度平方。恢复代理按沿参考方向的落后量／速度余量和横向偏离量构造，不是实际恢复时间保证。
6. 每约 0.5 s 重评，切换至少保持 0.8 s；通常需要成本改善超过 `max(0.03,0.15*旧成本)`。若旧参考已不可用，不强行保持。中间周期从当前状态重新计算同一参考的原窗口前缀并再检查。

等待参考需要预测到释放时刻；通道半宽估计用椭圆支撑宽度、裕量和额外缓冲。该估计是启发式，不能冒充严格清障时刻。没有有限释放估计则拒绝等待；并未证明永远不会等待或所有局部平衡均被排除。短前缀与长参考逐步计算一致，已有数值检查。

## 无效参考时撤销附加引导

首版 V3 在无参考通过时退回旧引导。预定重复试验出现负代理净距，且多次处于该回退分支。修正版引入 `progress_abstain=true`：在已激活的动态决策内，没有有效参考或新鲜重查失败时，返回零向量、引导权重 0，撤销额外速度／朝向引导，由原名义 MPC 继续优化。

**这不是直接输出 v=omega=0。** 静态／未激活场景仍走原引导；实际控制继续由原 MPC 求解与验证。`progress_abstain=false` 可恢复 V3 回退作同源对照。全局参考开关关闭时不启用上述机制。

该修正不解决原后端“求解失败或超时后强制停止”的全部风险。动态场景中零控制未必通过一步 DCBF 检查。新候选只是软目标，实际优化解也不必等于预览；没有证明新的递归可行性或物理安全。

## 冻结试验，按版本分别报告

- 开发用已看过的 41/73；前三种开发版本和全部失败源码／数据保留。
- V3 冻结提交 `f18442d`：种子 101/137、原参考／新参考、三场景，共 12 次主试验；另有两次预定重复、两次单种子均值消融及迎面／追越检查，共 18 次。主试验新参考 6/6、原参考 3/6，但新参考的同种子重复出现 −0.0421 m 代理净距。
- 撤销修正冻结提交 `89e1407`：看到上一反例后，明确换用新的 211/239；同一预览和预测器，仅比较旧回退与撤销参考，共 12 次主试验，加重复和迁移检查，共 16 次。修正主试验 5/6、旧回退 3/6；修正仍有减速及追越恢复超时。

两批不能混成一组独立重复。第二批不是在已看过的 101/137 上调好以后继续叫未见验证。全部试验使用同一 adaptive 预测器、相同场景位置／输入界／MPC／求解器；固定 Gazebo 种子不保证整个闭环逐位复现。两个种子仅作描述，不给统计显著性或安全概率结论。

主误差沿用 schema 2；从本阶段开始，验收直接读取任务窗内末值，保留末 CSV 行值作诊断。完整失败也参与均值。净距采用约 20 Hz 实际 Gazebo 位置和 0.13 m 机器人圆形代理，不是连续物理碰撞证明。

## 日志与复核

每次重评记录输入状态、原目标窗口、上一实际命令、速度界、障碍预测、各运动假设、候选分数／可用性、此前选择和本次选择。离线审计用经过 SHA 核验的该次保存源码重建，不读取真值或场景脚本；复核预览运动学、原窗口 DCBF、假设检查、成本和滞回。它证明数值决策可重建，不证明实际命令沿预览执行。

CSV 参考模式：0 为旧参考／未激活，1 直接恢复，2 等待，3 ccw，4 cw，5 撤销附加参考；具体原因见 JSON。`progress_decision_ms` 仅在新决策帧记录。原图表类别和统计公式保留。

## 运行研究副本

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rospack find tb3_distributed_control  # 应指向研究副本
roslaunch tb3_distributed_control single_turtlebot3_cbf_dynamic.launch \
  scenario:=accelerating_crossing prediction_mode:=adaptive gazebo_seed:=211 \
  progress_guidance:=true progress_abstain:=true total_time:=32

# 同源旧回退对照
roslaunch tb3_distributed_control single_turtlebot3_cbf_dynamic.launch \
  scenario:=accelerating_crossing prediction_mode:=adaptive gazebo_seed:=211 \
  progress_guidance:=true progress_abstain:=false total_time:=32

python3 -m unittest discover -s src/tb3_distributed_control/scripts/tests -p 'test_cbf*.py' -v
python3 src/tb3_distributed_control/scripts/tests/run_cbf_abstention_matrix.py \
  --seeds 211 239 --output /path/to/new_comparison
python3 src/tb3_distributed_control/scripts/tests/audit_cbf_progress.py /path/to/result
python3 src/tb3_distributed_control/scripts/tests/analyze_cbf_progress.py /path/to/comparison
```

当前仍有恢复失败、紧距离和计算预算问题；只完成单机仿真研究。新机制的多机与实物、可靠安全边界和新颖性论证尚未完成，不能据此宣称第二项论文贡献已经成立。
