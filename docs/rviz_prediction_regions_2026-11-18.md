# RViz预测危险区域（2026-11-18）

从b68f36e派生research/rviz-prediction-regions-2026-11-18，原工程只读、0.22阶段保留。参考本机MPC-D-CBF-main/obs_param/src/obs_kf.cpp的visEllipse：用世界坐标、预测中心/方向/半轴绘制半透明CYLINDER（参考项目绿色alpha0.4、高0.7 m）。当前原先仅默认画中心线，未来椭圆要开debug且只有端点，不能直接看出危险区域。

新主题/cbf/geometry/prediction_regions为所选观察机器人，/tb3_i/cbf_geometry/prediction_regions为各自独立观察；不共享原始雷达。RViz单机/多机配置默认启用新主题，原测量轮廓继续显示。窗口改为1280×800，单机视野适配40 s直线实验；不改机器人状态或控制。

绿色半透明扁椭圆表示原MPC节点0、N/2、N（当前0/0.5/1 s）的h=0区域。半轴使用实际solver输入prediction_axes，已含现有拟合保护、不确定性、执行裕量；再按a_safe=a(1+margin/b)、b_safe=b+margin画原CBF零边界，不重复加裕量。方向使用各预测节点自己的角度。它是机器人中心需要避开的区域，不是物体真实轮廓，也不是额外增加的约束。

开启0.22预测触发时，黄色虚线画1.5/2 s额外均值预测（保持原硬窗口末半轴/方向），明确属于软触发预览，不是2 s硬安全认证。阈值0.22不是图中椭圆半径；不把h=0.22等高线画成障碍实体。慢目标即使未越过dynamic分类速度门槛，也显示已有预测，不能用分类标签隐藏正在使用的包络。peer继续使用原独立显示，避免重复叠加。

控制回调只保存原输入引用，独立4 Hz计时器按需渲染。无订阅者不生成Marker，旧中心线移到debug以免重复；过期/丢失/ROS时间回退会DELETE旧ID，marker自身亦有有限寿命。所有代码仍在tb3_controller_node_cbf.py内，没有新增感知模块。未改求解器、目标、MPC/DCBF、0.22判据和误差计算。

216项检查通过，包括旋转椭圆轮廓逐点验证原h=0、输入不被修改、软节点末轴保持、peer排除和过期删除。事前冻结两次实际40 s试验：迎面211开启RViz并仅捕获本次RViz窗口；斜向211无GUI但订阅实际Marker。两次都开启现有0.22/2 s/2-4 s时程，原连续运动和100 ms验收不改。保存显示订阅下的原始Marker与控制输入，逐节点核对坐标/方向/半轴；如果有显示或实时性失败完整保留。

启动沿用研究副本single_turtlebot3_cbf_trigger022.launch，RViz默认开启；Displays中查看“CBF predicted danger regions (green MPC, amber soft preview)”。原工程与旧阶段不会自动更新。

首轮迎面GUI记录head_on_rviz_v1：控制/Marker验收通过，但人工检查截图发现布局未显示3D视口，不能称可视化通过。保留该图与所有数据。恢复已有可用Qt窗口布局状态、缩小面板高度；另修复roslaunch对robot_count的数值自动转换导致与字符串'1'比较恒假的问题（改int(...)==1）。修复后重新做迎面GUI验证，再进行斜向订阅测试。没有以“截图文件存在”代替人工查看画面。

第二次迎面head_on_rviz_v2的真实截图已人工确认有3D视口、绿色实心预测区域、黄色虚线和正常Global Status，控制验收通过。进一步只调整默认侧栏隐藏以扩大画面，可从Panels/Displays重新打开；单机缩放180。原定斜向订阅试验在执行前改为也打开RViz并截取本次窗口，以验证最终默认布局，不改控制/验收或场景。
