# 下一阶段研究工作空间：CA 预测

这是独立的 catkin 工作空间。来源是 2026-10-24 已验证的 CV 动态预测版本，复制日期为 2026-10-26。**当前算法仍是 CV；CA 和 IMM 尚未实现。**

后续研究修改这里的 `src/`。原工程的 `src/` 和 `阶段成果归档/` 作为保留基准。源码是普通文件副本，不使用链接或硬链接指向原工程。

## 构建和运行

```bash
source /opt/ros/noetic/setup.bash
cd /home/gdut/24_LC/turtlebot3_consensus_cbf_ws/研究副本/ca_prediction_ws_2026-10-26
catkin_make --only-pkg-with-deps tb3_distributed_control -j2
source devel/setup.bash
rospack find tb3_distributed_control
```

最后一行必须指向本目录下的 `src/tb3_distributed_control`，避免实际运行到原工作空间。

```bash
roslaunch tb3_distributed_control single_turtlebot3_cbf_dynamic.launch scenario:=crossing
roslaunch tb3_distributed_control multi_turtlebot3_consensus_gazebo_cbf_dynamic.launch scenario:=mixed
```

实验结果默认写到本副本包内的 `data_collect`，不会覆盖原实验结果。

## 本地版本记录

本目录具有独立的本地 Git 仓库，活动分支为 `research/ca-prediction`。初始标签为 `baseline/cv-dynamic-2026-10-24`；标签日期指已验证算法阶段，实际归档和提交创建于 2026-10-26。

`BASELINE_SHA256.json` 保存复制时的源码清单，不随新研究改动覆盖。生成的 build/devel、缓存及实验输出不进入 Git；应在阶段完成后另行归档并记录对应提交和参数。

本地 Git 没有远程地址。不能把本地提交理解为已经上传到 GitHub/Gitee。

## 阶段归档

完整原工程（含历史证据、原始数据和图表）位于上级根工程的：

`阶段成果归档/2026-10-26_阶段保护/完整工程与结果.tar.gz`

独立只读源码基线位于：

`阶段成果归档/2026-10-26_阶段保护/03_CV动态预测_源码工作空间`

这两个路径均相对于 `/home/gdut/24_LC/turtlebot3_consensus_cbf_ws`。完整清单、归档 SHA 和副本核验记录也在该阶段目录。
