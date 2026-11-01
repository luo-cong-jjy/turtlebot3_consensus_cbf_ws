# 统一评价口径（schema 2）

本阶段只统一评价与记录，不改变 CV 预测、联合 MPC 的目标／约束或求解器。在该评价升级阶段 CA/IMM 尚未实现；后续 CA 研究沿用本文件定义。旧标签和原实验文件保留。

## 公式与时间轴

所有误差用输入里程计的时间戳 `evaluation_ros_time` 计算，世界坐标系，位置单位 m、角度单位 rad。控制周期原时间仍保留在 `time/ros_time`；导出数据的 `t` 为测量时间轴，`control_t` 为控制周期时间轴。

| 指标 | 定义 | 字段 |
|---|---|---|
| 名义编队位置误差 | `p_actual(t) - p_leader(t) - d_nominal` | `consensus_error_x/y` |
| 编队朝向误差 | `wrap(theta_leader(t) - theta_actual(t))` | `heading_error` |
| 上一采用解的规划—执行偏差 | `r_previous(t) - p_actual(t)` | `plan_execution_error_x/y` |
| 对齐观测参考误差 | `observer_reference_aligned(t) - p_leader(t)` | `observer_error_x/y` |

`observer_x/y` 仍保存原始观测器状态，未改变观测器更新律；对齐后的控制参考不是另一条原始观测器状态。原协议存在下一拍领导者发布和观测器积分相位，评价端沿用控制端的时标校正后再比较，不能把图上的变小误读为观测器算法提升。

## 领导者协议解码

原 leader 状态消息给出标签时刻 `t` 的下一帧 `t+dt` 位姿与解析速度，clock 消息给出标签的 ROS 时间。评价端精确配对两个消息，再对同一测量时刻做 Hermite 插值；有限的端点外推标记为质量 2。

- `reference_quality=0`：不可用，误差记 NaN。
- `1`：Hermite 插值。
- `2`：至多 1.5dt 的恒速端点外推。
- `3`：精确落在数据节点。

不跨大间隙强行插值，不生成轨迹起点之前的参考。各类样本数量一并记录。

## 为什么不能直接取本次规划起点

联合 MPC 每次强制辅助规划 `r[0]=p_actual`，拿它相减会得到恒为零的伪指标。

新指标使用上一周期已采用且通过全窗口校验解的辅助规划控制积分参考，按当前测量时间插值。它是短时的“规划—执行偏差”，不冒充独立跟踪控制器的长期轨迹误差。

不同方法的内部规划结构不同，不能拿本指标的小数值直接与第一篇独立规划轨迹的长期跟踪误差比较并宣称提升。跨方法的主要评价应是固定名义编队目标误差、真实净距、任务完成／恢复和输入开销；本指标用于同一结构内的诊断／消融。

只在命令已采用之后、参考年龄不超过 1.5dt 的范围使用；首次、过期、停机及无全窗口规划的单步兜底，不制造参考。`plan_reference_valid/age/source_cycle` 给出来源；完整采用参考保存在 `*_solver_trace.jsonl` 的 `accepted_auxiliary_plan` 中，可逐行重建核验。

## 字段兼容与绘图

`reference_x/y`、导出数据 `xr/yr` 以及兼容 RobotHistoryData 的 xr/yr 仍是名义参考别名；不能再拿它们当内部辅助规划。

- `x_error/y_error`：名义编队误差。
- `heading_error`：目标角减实际角，范围仍 ±4 rad。
- `tracking_error_xy`：专用的上一规划—执行偏差，不再是名义误差反号。
- `observer_estimation`：新记录显示对齐观测参考，原始状态仍保存在数据中。
- `observer_error`：新增对齐观测参考误差图。

旧可视化程序直接使用 RobotHistoryData.xr-xc 时，不能自动获得新内部指标；新 CBF 实验应使用随 launch 启动的专用结果节点。

统计采用任务窗口 `[0,total_time]` 内的新鲜、不重复测量。保留缺失、过期、窗外和参考质量计数。输出 mean_signed、MAE、RMSE、最大绝对值、末端值；末端不可用时不偷换成此前最后一个有效点。求解失败但测量新鲜时的位置误差仍保留，不能靠删除失败阶段美化结果。

## 旧 CBF 数据转换

只写新目录，原 CSV／PKL／图片不覆盖：

```bash
python3 src/tb3_distributed_control/scripts/tests/recompute_cbf_metrics.py /path/to/old_CBF_result \
  --evaluation-end 40 --output /path/to/new_metrics_v2_result
```

旧 leader history 在赋新值前发布上一帧状态，因此要用前一条发布时刻+dt解释下一行保存的值。无法确认前驱的丢帧间隙跳过。旧 CBF CSV 若保留 ros_time 和 odom_age，可恢复位置测量时刻并重算名义误差和朝向误差。

旧记录没有采用的辅助规划／对齐观测参考时，两项保持 NaN/N/A；不能反推虚构。该适配器不是任意第一篇 pickle 格式的通用转换器。缺少必要时标的历史基线需在比较副本中用同口径重新记录，不追溯修改原发表图。

## 验证

```bash
python3 -m unittest discover -s src/tb3_distributed_control/scripts/tests -p 'test_cbf*.py' -v
python3 src/tb3_distributed_control/scripts/tests/audit_cbf_metrics.py /path/to/new_CBF_result
```

审计按已保存参考重建每行内部误差，检查源周期早于测量周期、符号、时标和缺失语义；对默认直线／慢八字再独立用解析领导者交叉检查。

独立 Gazebo 静态评价还修正了原评估脚本的两个问题：使用实际元数据中的编队半径；末端跟踪误差在任务结束时刻计算，不使用导出图片期间继续推进的参考时间。净距检查仍包含停机后采样。
