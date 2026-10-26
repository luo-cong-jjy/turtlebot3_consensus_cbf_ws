#!/usr/bin/env python3  # 使用 Python3 解释器
"""leader_follower model predictive control + cv using do-mpc SQP solver"""  # 模块说明
from __future__ import print_function  # 兼容打印函数
import sys  # 系统相关
import rospy  # ROS Python API
import cv2  # OpenCV
from std_msgs.msg import String  # ROS 标准消息
from sensor_msgs.msg import Image  # ROS 图像消息
from cv_bridge import CvBridge, CvBridgeError  # ROS-CV 桥接
from nav_msgs.msg import Odometry  # 里程计消息
from geometry_msgs.msg import Twist, Pose2D  # 速度与位姿消息
import math  # 数学函数
import time  # 时间函数
import numpy as np  # 数值计算
from tf.transformations import euler_from_quaternion  # 四元数转欧拉角
import os  # 文件路径
#
# do-mpc imports  # do-mpc 相关
import do_mpc  # do-mpc 包
from casadi import *  # CasADi 符号运算
#
# 添加 scripts 目录到 Python 路径  # 扩展模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件目录
scripts_dir = os.path.dirname(current_dir)  # scripts 上级目录
sys.path.insert(0, scripts_dir)  # 插入搜索路径
#
from image.photo1 import TargetTracker  # 目标跟踪器
#
class MPCDoMPC():  # MPC 控制类
    """  # 类文档开始
    使用do-mpc SQP求解器的MPC控制器  # 类说明
    替代原来的GPNN方法  # 兼容说明
    """  # 类文档结束
    #
    def __init__(self, N, Nu, gamma):  # 构造函数
        """  # 方法文档开始
        初始化MPC控制器  # 方法说明
        #
        参数:  # 参数说明
            N: 预测时域  # 预测长度
            Nu: 控制时域 (注意: do-mpc默认 Nu=N)  # 控制长度
            gamma: GPNN参数(现在不使用，但保留接口兼容性)  # 兼容参数
        """  # 方法文档结束
        rospy.loginfo("初始化 do-mpc SQP 控制器...")  # 日志
        #
        # 基本参数  # 基础配置
        self.N = N  # 预测时域
        self.Nu = Nu  # 控制时域
        self.gamma = gamma  # 兼容参数
        #
        # 目标跟踪器  # 视觉跟踪
        self.target_detector = TargetTracker()  # 实例化跟踪器
        #
        # ROS Publishers  # 发布器
        self.pub = rospy.Publisher('/tb3_1/cmd_vel', Twist, queue_size=10)  # 速度话题
        self.obj_pub = rospy.Publisher('/tb3_1/errors_pub', Pose2D, queue_size=10)  # 误差话题
        #
        # ROS Subscribers  # 订阅器
        self.odom_subl = rospy.Subscriber('/tb3_0/odom', Odometry, self.odom_callbackl, queue_size=1)  # 领导者里程计
        self.odom_subf = rospy.Subscriber('/tb3_1/odom', Odometry, self.odom_callbackf, queue_size=1)  # 跟随者里程计
        #
        # 控制频率  # 循环频率
        self.rate = rospy.Rate(10)  # 10Hz 频率
        #
        # 机器人状态变量初始化  # 状态初始化
        self.robot_pose_xf = 0.0  # 跟随者 x
        self.robot_pose_yf = 0.0  # 跟随者 y
        self.robot_pose_thetaf = 0.0  # 跟随者 yaw
        self.linear_velocity_xf = 0.0  # 跟随者线速度
        self.angular_velocity_zf = 0.0  # 跟随者角速度
        #
        self.robot_pose_xl = 0.0  # 领导者 x
        self.robot_pose_yl = 0.0  # 领导者 y
        self.robot_pose_thetal = 0.0  # 领导者 yaw
        self.linear_velocity_xl = 0.0  # 领导者线速度
        self.angular_velocity_zl = 0.0  # 领导者角速度
        #
        # 时间管理  # 时间步长
        self.last_time = time.time()  # 上次时间
        self.dtt = 0.1  # 默认 dt
        #
        # 控制参数  # 几何参数
        self.d = 0.045  # 机器人中心到头部的距离
        #
        # 期望值  # 编队目标
        self.des_distance = 0.8  # 期望距离
        self.des_angle = math.pi / 8  # 期望角度
        #
        # 权重矩阵  # 代价权重
        self.Qd_weight = 200.0  # 距离误差权重
        self.Qt_weight = 50.0  # 角度误差权重
        self.R_weight = 5.0  # 控制输入权重
        #
        # 约束  # 输入与速度约束
        self.u_min = -0.5  # 输入下界
        self.u_max = 0.5  # 输入上界
        self.max_linear_velocity = 0.22  # 最大线速度
        self.max_angular_velocity = 2.84  # 最大角速度
        #
        # 设置do-mpc MPC控制器  # 初始化 MPC
        self.mpc, self.model = self.setup_mpc_sqp()  # 建模与控制器
        #
        # 初始状态  # 初始化状态
        self.x0 = np.array([[self.des_distance], [self.des_angle]])  # 初始误差
        self.mpc.x0 = self.x0  # 设定初值
        self.mpc.set_initial_guess()  # 初始猜测
        #
        # Twist消息  # 速度消息
        self.rot = Twist()  # 复用消息对象
        #
        rospy.loginfo("do-mpc SQP 控制器初始化完成!")  # 日志
    #
    def setup_mpc_sqp(self):  # MPC 配置函数
        """  # 方法文档开始
        配置do-mpc MPC控制器，使用SQP求解器  # 方法说明
        """  # 方法文档结束
        rospy.loginfo("配置 do-mpc 模型...")  # 日志
        #
        # 创建离散时间模型  # 模型构建
        model = do_mpc.model.Model('discrete')  # 离散模型
        #
        # 定义状态变量  # 状态定义
        # x[0]: di - 距离误差  # 注释
        # x[1]: thetai - 角度误差  # 注释
        di = model.set_variable('_x', 'di')  # 距离误差
        thetai = model.set_variable('_x', 'thetai')  # 角度误差
        #
        # 定义控制输入  # 输入定义
        # u[0]: vc - 线速度  # 注释
        # u[1]: wc - 角速度  # 注释
        vc = model.set_variable('_u', 'vc')  # 线速度
        wc = model.set_variable('_u', 'wc')  # 角速度
        #
        # 定义时变参数 (领导者速度和几何参数)  # TVP
        vL = model.set_variable('_tvp', 'vL')  # 领导者线速度
        WL = model.set_variable('_tvp', 'WL')  # 领导者角速度
        alpha = model.set_variable('_tvp', 'alpha')  # 几何角
        beta = model.set_variable('_tvp', 'beta')  # 几何角
        #
        # 系统动力学 (离散化)  # 动力学
        # 这里使用欧拉离散化，delta_t 会在运行时通过参数传递  # 离散化方式
        delta_t = 0.1  # 默认采样时间
        #
        # 状态更新方程  # 方程辅助
        d = self.d  # 几何参数
        #
        # 避免除零  # 数值安全
        di_safe = if_else(di < 0.01, 0.01, di)  # 安全距离
        #
        # 连续时间动力学  # 动力学表达式
        ddi_dt = -vL * cos(alpha) + vc * cos(beta) + d * wc * sin(beta)  # 距离导数
        dthetai_dt = (  # 角度导数
            -vL * sin(alpha) / di_safe  # 领导者项
            + vc * sin(beta) / di_safe  # 跟随者项
            + wc * (1 - d * sin(beta) / di_safe)  # 自身角速度项
        )  # 结束
        #
        # 离散化  # 欧拉步进
        di_next = di + delta_t * ddi_dt  # 下一步距离
        thetai_next = thetai + delta_t * dthetai_dt  # 下一步角度
        #
        model.set_rhs('di', di_next)  # 设置右端项
        model.set_rhs('thetai', thetai_next)  # 设置右端项
        #
        model.setup()  # 模型完成
        #
        rospy.loginfo("配置 MPC 控制器...")  # 日志
        #
        # 创建MPC控制器  # 控制器实例
        mpc = do_mpc.controller.MPC(model)  # MPC 对象
        #
        # MPC参数设置 - 修正IPOPT选项  # 求解器配置
        setup_mpc = {  # 参数字典
            'n_horizon': self.N,  # 预测时域
            't_step': 0.1,  # 采样时间
            'n_robust': 0,  # 无鲁棒
            'store_full_solution': True,  # 保存全解
            # 使用IPOPT求解器的正确选项  # 求解器选项
            'nlpsol_opts': {  # IPOPT 选项
                'ipopt.max_iter': 50,  # 最大迭代次数
                'ipopt.print_level': 0,  # 减少输出
                'ipopt.sb': 'yes',  # 抑制横幅
                'ipopt.acceptable_tol': 1e-4,  # 可接受容差
                'ipopt.tol': 1e-6,  # 最优性容差
                'ipopt.acceptable_iter': 15,  # 可接受迭代
                'ipopt.mu_strategy': 'adaptive',  # 屏障策略
                'ipopt.hessian_approximation': 'limited-memory',  # L-BFGS
                'ipopt.warm_start_init_point': 'yes',  # 热启动
                'print_time': 0,  # 不打印时间
                # 移除不支持的选项  # 说明
                # 'acceptable_obj_change_tol': 1e-3,     # 这个选项不存在  # 保留注释
            }  # 结束 IPOPT 选项
        }  # 结束参数字典
        #
        mpc.set_param(**setup_mpc)  # 写入参数
        #
        # 目标函数 - 跟踪期望距离和角度  # 代价函数
        lterm = (  # 阶段代价
            self.Qd_weight * (di - self.des_distance) ** 2  # 距离误差
            + self.Qt_weight * (thetai - self.des_angle) ** 2  # 角度误差
        )  # 结束
        mterm = lterm  # 终端代价相同
        #
        mpc.set_objective(lterm=lterm, mterm=mterm)  # 设置目标
        #
        # 控制输入权重  # 输入权重
        mpc.set_rterm(vc=self.R_weight, wc=self.R_weight)  # 输入加权
        #
        # 控制约束  # 输入约束
        mpc.bounds['lower', '_u', 'vc'] = self.u_min  # 线速度下界
        mpc.bounds['upper', '_u', 'vc'] = self.u_max  # 线速度上界
        mpc.bounds['lower', '_u', 'wc'] = self.u_min  # 角速度下界
        mpc.bounds['upper', '_u', 'wc'] = self.u_max  # 角速度上界
        #
        # 状态约束  # 状态边界
        mpc.bounds['lower', '_x', 'di'] = 0.1  # 最小距离
        mpc.bounds['upper', '_x', 'di'] = 5.0  # 最大距离
        mpc.bounds['lower', '_x', 'thetai'] = -np.pi  # 最小角度
        mpc.bounds['upper', '_x', 'thetai'] = np.pi  # 最大角度
        #
        # 设置时变参数函数  # TVP 函数
        tvp_template = mpc.get_tvp_template()  # TVP 模板
        #
        def tvp_fun(t_now):  # TVP 回调
            return tvp_template  # 返回模板
        #
        mpc.set_tvp_fun(tvp_fun)  # 注册 TVP
        #
        # 完成设置  # MPC 建立
        mpc.setup()  # 完成初始化
        #
        rospy.loginfo("MPC 控制器配置完成!")  # 日志
        #
        return mpc, model  # 返回对象
    #
    def odom_callbackf(self, msg):  # 跟随者回调
        """跟随者里程计回调"""  # 方法说明
        self.robot_pose_xf = msg.pose.pose.position.x  # 位置 x
        self.robot_pose_yf = msg.pose.pose.position.y  # 位置 y
        orientation_qf = msg.pose.pose.orientation  # 四元数
        orientation_listf = [orientation_qf.x, orientation_qf.y, orientation_qf.z, orientation_qf.w]  # 列表
        _, _, self.robot_pose_thetaf = euler_from_quaternion(orientation_listf)  # yaw
        #
        self.linear_velocity_xf = msg.twist.twist.linear.x  # 线速度
        self.angular_velocity_zf = msg.twist.twist.angular.z  # 角速度
    #
    def odom_callbackl(self, msg):  # 领导者回调
        """领导者里程计回调"""  # 方法说明
        self.robot_pose_xl = msg.pose.pose.position.x  # 位置 x
        self.robot_pose_yl = msg.pose.pose.position.y  # 位置 y
        orientation_ql = msg.pose.pose.orientation  # 四元数
        orientation_listl = [orientation_ql.x, orientation_ql.y, orientation_ql.z, orientation_ql.w]  # 列表
        _, _, self.robot_pose_thetal = euler_from_quaternion(orientation_listl)  # yaw
        #
        self.linear_velocity_xl = msg.twist.twist.linear.x  # 线速度
        self.angular_velocity_zl = msg.twist.twist.angular.z  # 角速度
    #
    def compute_formation_errors(self):  # 误差计算
        """  # 方法文档开始
        计算编队误差 (di, thetai)  # 方法说明
        #
        返回:  # 返回说明
            di: 距离误差  # 距离
            thetai: 角度误差  # 角度
            alpha, beta: 几何角度参数  # 几何参数
        """  # 方法文档结束
        # 获取当前位姿  # 当前状态
        thetaf = self.robot_pose_thetaf  # 跟随者角度
        thetaL = self.robot_pose_thetal  # 领导者角度
        xl = self.robot_pose_xl  # 领导者 x
        yl = self.robot_pose_yl  # 领导者 y
        #
        # 跟随者头部位置  # 头部坐标
        xf = self.robot_pose_xf + self.d * math.cos(thetaf)  # 头部 x
        yf = self.robot_pose_yf + self.d * math.sin(thetaf)  # 头部 y
        #
        # 在跟随者坐标系中的误差  # 坐标变换
        ex = (xl - xf) * math.cos(thetaf) + (yl - yf) * math.sin(thetaf)  # x 误差
        ey = -(xl - xf) * math.sin(thetaf) + (yl - yf) * math.cos(thetaf)  # y 误差
        #
        # 转换坐标系  # 坐标调整
        exx = -ey  # x 轴
        eyy = ex  # y 轴
        #
        # 计算距离和角度  # 极坐标
        di = math.sqrt(exx**2 + eyy**2)  # 距离
        thetai = math.atan2(exx, eyy)  # 角度
        #
        # 计算几何角度  # 几何关系
        fai = thetaL - thetaf  # 相对角
        alpha = math.pi - fai - thetai  # alpha
        beta = alpha + fai  # beta
        #
        return di, thetai, alpha, beta  # 返回误差
    #
    def controlLaw(self):  # 控制律
        """  # 方法文档开始
        主控制律 - 使用do-mpc求解MPC问题  # 方法说明
        """  # 方法文档结束
        # 更新时间  # 时间更新
        current_time = time.time()  # 当前时间
        self.dtt = current_time - self.last_time  # 计算 dt
        self.last_time = current_time  # 更新时间戳
        #
        # 限制时间步长  # 限制 dt
        if self.dtt > 1.0 or self.dtt < 0.01:  # 异常 dt
            delta_t = 0.1  # 使用默认 dt
        else:  # 正常 dt
            delta_t = self.dtt  # 使用测量 dt
        #
        # 计算编队误差  # 误差估计
        di, thetai, alpha, beta = self.compute_formation_errors()  # 误差计算
        angle, distance, Xc, Yc, Zc, found = self.target_detector.calculate_target_info()  # 视觉信息
        thetai = angle  # 使用视觉角度
        di = distance  # 使用视觉距离
        vL = self.linear_velocity_xl  # 领导者线速度
        thetaf = self.robot_pose_thetaf  # 跟随者角度
        thetaL = self.robot_pose_thetal  # 领导者角度
        WL = self.angular_velocity_zl  # 领导者角速度
        fai = thetaL - thetaf  # 相对角
        alpha = np.pi - fai - thetai  # 更新 alpha
        beta = alpha + fai  # 更新 beta
        rospy.loginfo("Errors - di: %.4f, thetai: %.4f", di, thetai)  # 打印误差
        #
        # 获取领导者速度  # 速度更新
        vL = self.linear_velocity_xl  # 线速度
        WL = self.angular_velocity_zl  # 角速度
        #
        # 更新时变参数 (领导者速度和几何参数)  # TVP 更新
        tvp_template = self.mpc.get_tvp_template()  # TVP 模板
        #
        # 为整个预测时域设置相同的时变参数  # 填充 TVP
        for k in range(self.N + 1):  # 遍历时域
            tvp_template['_tvp', k, 'vL'] = vL  # 线速度
            tvp_template['_tvp', k, 'WL'] = WL  # 角速度
            tvp_template['_tvp', k, 'alpha'] = alpha  # alpha
            tvp_template['_tvp', k, 'beta'] = beta  # beta
        #
        # 创建闭包捕获参数  # 回调封装
        def make_tvp_fun(tvp_vals):  # 工厂函数
            def tvp_fun(t):  # TVP 回调
                return tvp_vals  # 返回固定模板
            return tvp_fun  # 返回回调
        #
        self.mpc.set_tvp_fun(make_tvp_fun(tvp_template))  # 注册 TVP
        #
        # 当前状态  # 状态向量
        x_current = np.array([[di], [thetai]])  # 状态输入
        #
        # 求解MPC优化问题  # 求解器调用
        try:  # 异常保护
            u_opt = self.mpc.make_step(x_current)  # 求解一步
            #
            vc = float(u_opt[0])  # 解析线速度
            wc = float(u_opt[1])  # 解析角速度
            #
            rospy.loginfo("MPC solution - vc: %.4f, wc: %.4f", vc, wc)  # 输出解
            #
        except Exception as e:  # 求解失败
            rospy.logwarn("MPC求解失败: %s, 使用默认控制", str(e))  # 警告
            # 使用简单的比例控制作为备份  # 备用策略
            vc = 0.2 * (di - self.des_distance)  # 比例线速度
            wc = 0.5 * (thetai - self.des_angle)  # 比例角速度
        #
        # 应用速度约束  # 速度限制
        vc = np.clip(vc, -self.max_linear_velocity, self.max_linear_velocity)  # 线速度限幅
        wc = np.clip(wc, -self.max_angular_velocity, self.max_angular_velocity)  # 角速度限幅
        #
        rospy.loginfo("Applied control - vc: %.4f, wc: %.4f", vc, wc)  # 输出控制
        #
        # 发布控制命令  # 速度发布
        self.rot.linear.x = vc  # 线速度赋值
        self.rot.angular.z = wc  # 角速度赋值
        self.pub.publish(self.rot)  # 发布速度
        #
        # 发布误差信息  # 误差发布
        err_di = di - self.des_distance  # 距离误差
        err_thetai = thetai - self.des_angle  # 角度误差
        self.obj_pub.publish(Pose2D(err_di, 0, err_thetai))  # 发布误差
    #
    def run_control_loop(self):  # 主循环
        """主循环函数"""  # 方法说明
        rospy.loginfo("do-mpc SQP Controller node is running...")  # 日志
        try:  # 捕获中断
            while not rospy.is_shutdown():  # ROS 循环
                self.controlLaw()  # 执行控制
                self.rate.sleep()  # 睡眠
                #
        except KeyboardInterrupt:  # 键盘中断
            rospy.loginfo("Keyboard interrupt detected.")  # 日志
        finally:  # 总是执行
            # 停止机器人  # 安全停止
            self.stop()  # 停止命令
    #
    def stop(self):  # 停止函数
        """停止机器人"""  # 方法说明
        stop_cmd = Twist()  # 构造停止命令
        stop_cmd.linear.x = 0.0  # 线速度归零
        stop_cmd.angular.z = 0.0  # 角速度归零
        self.pub.publish(stop_cmd)  # 发布停止
        rospy.loginfo("Robot stopped.")  # 日志
#
#
if __name__ == '__main__':  # 入口判断
    try:  # 启动保护
        rospy.init_node('mpc_dompc_sqp', anonymous=False)  # 初始化节点
        #
        # 创建MPC控制器  # 参数说明
        # N=5: 增加预测时域以提高性能  # 预测时域
        # Nu=2: 控制时域 (注意: do-mpc实际使用 Nu=N)  # 控制时域
        # gamma=0.1: 保留参数，兼容接口  # 兼容参数
        controller = MPCDoMPC(N=5, Nu=2, gamma=0.1)  # 构造控制器
        #
        rospy.sleep(2)  # 等待初始化和话题连接
        #
        # 运行控制循环  # 启动循环
        controller.run_control_loop()  # 进入循环
        #
    except rospy.ROSInterruptException:  # ROS 中断
        rospy.loginfo("ROS node interrupted.")  # 日志
    except Exception as e:  # 其它异常
        rospy.logerr("发生错误: %s", str(e))  # 打印错误
        import traceback  # 回溯模块
        traceback.print_exc()  # 打印堆栈
    finally:  # 总是执行
        rospy.loginfo("Shutting down MPC controller node.")  # 退出日志