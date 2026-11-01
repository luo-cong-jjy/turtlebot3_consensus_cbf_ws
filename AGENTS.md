# 研究与阶段保护

- 当前默认基线仍为 CV；2026-11-01 研究分支已加入可切换 CA 及变速场景。新增可选 Singer、缓冲／引导消融、默认关闭的速度坐标投影候选；没有自动模型选择，IMM 尚未实现。先阅读 docs/singer_ablation_2026-11-01.md、docs/ca_prediction.md 和阶段验证，不默认把 CA 当作全场景更优的方法。
- 在本机，该仓库位于独立研究副本中。原工程与本地阶段归档保留为历史基准，不默认写回。
- 修改前检查 Git 状态，保留用户已有改动；新研究使用分支，不覆盖已发布的阶段标签。
- 保持 CV/frozen 对照可运行；新增 CA 模式时不要无记录地替换 CV 基线。
- 每个稳定阶段保留提交、标签、独立源码快照及验证记录。大体积结果不入 Git 时，应单独归档并记录对应提交、参数和源码 SHA，保留失败试验。
- 构建／roslaunch 前 source 当前工作空间 devel/setup.bash，并确认 rospack 定位到本仓库。
- origin 为用户指定的 https://github.com/luo-cong-jjy/turtlebot3_consensus_cbf_ws.git，用户于 2026-11-01 授权首次提交。后续远程推送按用户任务授权执行，不自动上传论文、凭据和原始实验数据。

- 本研究分支评价口径为 metrics schema 2，先阅读 docs/metrics_schema_v2.md。不要把名义 reference_x/y 或本次锚定的 r[0] 当作内部规划跟踪参考；缺失字段不补零。
