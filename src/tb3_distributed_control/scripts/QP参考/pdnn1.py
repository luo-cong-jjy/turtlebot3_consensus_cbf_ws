#!/usr/bin/env python3
"leader_follower model predictive control + cv"
from __future__ import print_function
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D
import math
import time
import numpy as np
from tf.transformations import euler_from_quaternion
import os

# 添加 scripts 目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
sys.path.insert(0, scripts_dir)

from image.photo1_0 import TargetTracker
class MPC():
    
    # ... 您的 MPC 类代码保持不变 ...
    def __init__(self, N, Nu, gamma):
        self.target_detector = TargetTracker()
        self.N = N
        self.Nu = Nu
        self.m = 4*N + 4*Nu
        self.gamma = gamma
        self.pub = rospy.Publisher('/tb3_1/cmd_vel',Twist,queue_size=10) # <<< MODIFIED: queue_size can be smaller
        self.obj_pub = rospy.Publisher('/tb3_1/errors_pub',Pose2D,queue_size=10) # <<< MODIFIED: queue_size can be smaller
        self.rot=Twist()
        self.odom_subl = rospy.Subscriber('/tb3_0/odom', Odometry, self.odom_callbackl, queue_size=1)
        self.odom_subf = rospy.Subscriber('/tb3_1/odom', Odometry, self.odom_callbackf, queue_size=1)
        self.rate = rospy.Rate(10)  # 10Hz
        self.uu = np.zeros((2*self.Nu,1))
        self.last_time = 0
        self.dtt = 0
        self.yo = np.zeros((4*self.N + 6*self.Nu,1))
    def RungeKutta(self,func,y0):
        self.func = func
        h = 0.01
        n = 10
        y11 = np.zeros((4*self.N+6*self.Nu,1))  # Assuming y11 is initialized elsewhere
        y11 = y0
        for i in range(n):
            k11 = self.func(y11)
            k21 = self.func(y11 + h * k11 / 2)
            k31 = self.func(y11 + h * k21 / 2)
            k41 = self.func(y11 + h * k31)
            y11 = y11 + h * (k11 + 2*k21 + 2*k31 + k41) / 6
        return y11

    def gfuntion(self,uy):
        detabarumin = -0.5 * np.ones((2*self.Nu,1))
        detabarumax = 0.5 * np.ones((2*self.Nu,1))
        ones = np.ones((self.m,1))
        pinfty = np.inf * ones
        minfty = -pinfty
        yp = np.concatenate((detabarumax, pinfty))
        ym = np.concatenate((detabarumin, minfty))
        for i in range(4*self.N+6*self.Nu):
            if uy[i] > yp[i]:
                uy[i] = yp[i]
            elif uy[i] < ym[i]:
                uy[i] = ym[i]
            else:
                uy[i] = uy[i]
        return uy

    def func(self,y2):
        udot = self.gamma * self.IpHt.dot(self.gfuntion(self.ImH.dot(y2) - self.p) - y2)
        return udot
    
    def odom_callbackf(self, msg):
        # This function is fine as it is, just storing data.
        self.robot_pose_xf = msg.pose.pose.position.x
        self.robot_pose_yf = msg.pose.pose.position.y
        orientation_qf = msg.pose.pose.orientation
        orientation_listf = [orientation_qf.x, orientation_qf.y, orientation_qf.z, orientation_qf.w]
        _, _, self.robot_pose_thetaf = euler_from_quaternion(orientation_listf)

        self.linear_velocity_xf = msg.twist.twist.linear.x
        self.angular_velocity_zf = msg.twist.twist.angular.z

    def odom_callbackl(self, msg):
        # This function is fine as it is, just storing data.
        self.robot_pose_xl = msg.pose.pose.position.x
        self.robot_pose_yl = msg.pose.pose.position.y
        orientation_ql = msg.pose.pose.orientation
        orientation_listl = [orientation_ql.x, orientation_ql.y, orientation_ql.z, orientation_ql.w]
        _, _, self.robot_pose_thetal = euler_from_quaternion(orientation_listl)

        self.linear_velocity_xl = msg.twist.twist.linear.x
        self.angular_velocity_zl = msg.twist.twist.angular.z

    def controlLaw(self):
        current_time = time.time()
        self.dtt = current_time - self.last_time
        self.last_time = current_time
        rospy.loginfo("dtt: %f", self.dtt)
    
        angle, distance, Xc, Yc, Zc, found = self.target_detector.calculate_target_info()
        thetai = angle
        di = distance
        Disturbance_di = 0.02 * math.sin(0.5 * time.time()) * np.cos(0.3 * time.time())
        Disturbance_thetai = 0.1 * math.sin(0.4 * time.time()) * np.cos(0.6 * time.time())
        # di = di + Disturbance_di
        # thetai = thetai + Disturbance_thetai  
        # delta_t = 0.1
        if self.dtt > 10:
            delta_t = 0.1
        else:
            delta_t = self.dtt
        vL = self.linear_velocity_xl
        thetaf = self.robot_pose_thetaf
        thetaL = self.robot_pose_thetal
        WL = self.angular_velocity_zl
        # 不使用视觉，直接使用里程计数据计算误差(使用视觉时需注释掉)
        # xl = self.robot_pose_xl
        # yl = self.robot_pose_yl
        # xf = self.robot_pose_xf
        # yf = self.robot_pose_yf
        # ex = (xl - xf) * math.cos(thetaf) + (yl - yf) * math.sin(thetaf)
        # ey = -(xl - xf) * math.sin(thetaf) + (yl - yf) * math.cos(thetaf)
        # exx= - ey
        # eyy= ex
        # etheta = thetaL - thetaf
        # rospy.loginfo("ex: %.4f, ey: %.4f", exx, eyy)
        # di = math.sqrt(exx**2 + eyy**2)
        # thetai =  math.atan2(exx, eyy)
        # rospy.loginfo("di: %.4f, thetai: %.4f", di, thetai)

        ##################
        u = np.zeros((2, 1))
        Qd =1000 * np.eye(self.N)
        Qt =100 * np.eye(self.N)
        Qz = np.zeros((self.N, self.N))
        Q = np.block([[Qd, Qz], [Qz, Qt]])
        # Q = 5000 * np.eye(2 * self.N)
        R = 5 * np.eye(2 * self.Nu) 
        # R = np.diag([1.0, 20.0, 1.0, 20.0])
        # print("R:", R)
        z = np.zeros((2, 2))
        abaru = np.zeros((2 * self.Nu,1))
        abaru = self.uu
        u = abaru[:2]
        # rospy.loginfo("u:" + str(u))
        des = np.vstack((1.0, math.pi/8))  #期望
        des1 = np.vstack((des, des, des))

        x = np.vstack([di, thetai])  # Assuming xe, ye, thetae are defined elsewhere
        tildrI = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1]])

        abarumin = np.ones((2*self.Nu,1)) * -0.5 * 1
        abarumax = np.ones((2*self.Nu,1)) * 0.5 * 1
        abarxmin = -5 * np.ones((2 * self.N,1))
        abarxmax = 5 * np.ones((2 * self.N,1))

        thetaii = thetai - thetaf - thetaL
        # ttt = thetaL - thetaf
        # rospy.loginfo("thetaL - thetaf: %.4f", ttt)
        thetaii = np.pi -thetai - (thetaL - thetaf)
        fai = thetaL - thetaf
        alpha = np.pi - fai -thetai
        beta = alpha + fai
        fxx = np.vstack([-vL*math.cos(alpha),
                         -vL*math.sin(alpha)/di ])
        fx = x + delta_t * fxx
        # print("fx:", fx.shape)
        d = 0.045
        gxx = np.vstack([[math.cos(beta), d * math.sin(beta)],
                         [math.sin(beta)/di, 1-d * math.sin(beta)/di]])
        gx = delta_t * gxx
        # print("gx:", gx.shape)

        G = np.vstack((np.hstack((gx, z)), np.hstack((gx, gx)), np.hstack((gx, gx))))
        print("u:", u)
        tildrg = np.vstack((np.dot(gx, u), np.dot(gx, u), np.dot(gx, u)))
        tildrf = np.vstack((fx, fx, fx))
        # print("G:", G.shape)
        # print("tildrg:", tildrg.shape)
        # print("tildrf:", tildrf.shape)


        W = 2 * (np.dot(np.dot(G.T, Q), G) + R)
        C = 2 * np.dot(np.dot(G.T, Q), tildrg + tildrf - des1)   # 这里改为负号,不用
        E = np.vstack((-tildrI, tildrI, -G, G))
        b = np.vstack((-abarumin + abaru, 
                        abarumax - abaru, 
                       -abarxmin + tildrf + tildrg, 
                        abarxmax - tildrf - tildrg))
        # print("W:", W.shape)
        # print("C:", C.shape)
        # print("E:", E.shape)
        # print("b:", b.shape)

        M = np.block([[W, -E.T], [E, np.zeros((self.m, self.m))]])
        self.p = np.vstack((C, -b))
        I = np.eye(4 * self.N + 6 * self.Nu)
        self.ImH = I - M
        self.IpHt = I + M.T
        # print("M", M.shape)
        # print("p", self.p.shape)
        # print("ImH", self.ImH.shape)
        # print("IpHt", self.IpHt.shape)

        func = self.func
        print("func:", func)
        y11 = self.RungeKutta(func,self.yo)
        self.yo = y11
        detabaru = np.vstack([y11[0], y11[1], y11[2], y11[3]])
        abaru = detabaru
        self.uu = abaru
        u = abaru[:2]
        x = fx + np.dot(gx, u)
        j = x
        dis = j[0]
        thetai = j[1]

        vc = u[0,:]    
        wc = u[1,:]
        # 速度上的扰动
        # vc = vc + 0.02 * np.sin(0.5 * time.time())*np.cos(0.1 * time.time())
        # wc = wc + 0.02 * np.sin(0.3 * time.time())*np.cos(0.7 * time.time())
        rospy.loginfo("vc: %.4f, wc: %.4f", vc, wc)
        rospy.loginfo("self.linear_velocity_xf: %.4f, self.angular_velocity_zf: %.4f", self.linear_velocity_xf, self.angular_velocity_zf)
        self.rot.linear.x = vc 
        rospy.loginfo(self.rot.linear.x)
        self.rot.angular.z = wc
        err = x - des
        self.obj_pub.publish(Pose2D(err[0], 0, err[1]))
        self.pub.publish(self.rot)
    def run_control_loop(self):
            """主循环函数"""
            rospy.loginfo("Tube Controller node is running...")
            try:
                while not rospy.is_shutdown():
                    
                    self.controlLaw()
                    
                    self.rate.sleep()
            except KeyboardInterrupt:
                rospy.loginfo("Keyboard interrupt detected.")
            finally:
                # 在停止前发送一个停止命令以确保机器人停下来
                stop_cmd = Twist()
                stop_cmd.linear.x = 0.0
                stop_cmd.angular.z = 0.0
                self.pub.publish(stop_cmd)
                rospy.loginfo("Emergency stop command sent.")
    #             # 绘制结果图
    #             # self.plot_results()

# class TakePhoto:
#     def __init__(self):
        
#         self.pub = rospy.Publisher('tb3_1/cmd_vel',Twist,queue_size=10) # <<< MODIFIED: queue_size can be smaller
#         self.obj_pub = rospy.Publisher('tb3_1/errors_pub',Pose2D,queue_size=10) # <<< MODIFIED: queue_size can be smaller
#         self.rot=Twist()

#         self.mpc = MPC(3, 2, 0.1)

#         # <<< NEW: Add a variable to store the latest image
#         self.cv_image = None

#         self.cx=0
#         self.cy=0
#         self.distance=0
#         self.dis=0
#         self.center_x=0
#         self.center_y=0
#         self.Xc=0
#         self.Yc=0
#         self.Zc=0
#         self.angle_rad=0
#         self.dt = 0.1 # <<< MODIFIED: Set dt to our fixed loop rate initially
#         self.X=0
#         self.Y=0
#         self.W=0
#         self.H=0
#         self.linear_velocity_xl = 0;self.robot_pose_thetal = 0;self.robot_pose_thetaf = 0

#         self.found = False
#         self.notFoundCount = 0

#         self.bridge = CvBridge()
#         self.image_received = False

#         self.measSize = 4
#         self.meas = np.zeros((self.measSize, 1), np.float32)

#         # Connect image topic
#         img_topic = "tb3_1/camera/rgb/image_raw"
#         # <<< MODIFIED: queue_size=1 is important to get the latest image
#         self.image_sub = rospy.Subscriber(img_topic, Image, self.callback, queue_size=1, buff_size=2**24)

#         # Connect odometry topic
#         self.odom_subl = rospy.Subscriber('tb3_0/odom', Odometry, self.odom_callbackl, queue_size=1)
#         self.odom_subf = rospy.Subscriber('tb3_1/odom', Odometry, self.odom_callbackf, queue_size=1)

#         self.last_time = time.time()
#         rospy.loginfo("TakePhoto node initialized.")


#     def callback(self, data):
#         # <<< MODIFIED: This callback now ONLY stores the latest image.
#         # It's fast and doesn't block.
#         try:
#             self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
#             self.image_received = True
#         except CvBridgeError as e:
#             print(e)
       
#     def odom_callbackl(self, msg):
#         # This function is fine as it is, just storing data.
#         self.robot_pose_xl = msg.pose.pose.position.x
#         self.robot_pose_yl = msg.pose.pose.position.y
#         orientation_ql = msg.pose.pose.orientation
#         orientation_listl = [orientation_ql.x, orientation_ql.y, orientation_ql.z, orientation_ql.w]
#         _, _, self.robot_pose_thetal = euler_from_quaternion(orientation_listl)

#         self.linear_velocity_xl = msg.twist.twist.linear.x
#         self.angular_velocity_zl = msg.twist.twist.angular.z
    
#     def odom_callbackf(self, msg):
#         # This function is fine as it is, just storing data.
#         self.robot_pose_xf = msg.pose.pose.position.x
#         self.robot_pose_yf = msg.pose.pose.position.y
#         orientation_qf = msg.pose.pose.orientation
#         orientation_listf = [orientation_qf.x, orientation_qf.y, orientation_qf.z, orientation_qf.w]
#         _, _, self.robot_pose_thetaf = euler_from_quaternion(orientation_listf)

#         self.linear_velocity_xf = msg.twist.twist.linear.x
#         self.angular_velocity_zf = msg.twist.twist.angular.z
    
#     # ... All your other helper functions (Get_Distance, etc.) remain unchanged ...
#     def Get_Distance(self, actual_Width, focal_length_P, Object_width_P):
#         D = (actual_Width * focal_length_P) / Object_width_P
#         D = round(D, 2)
#         return D
    
#     def get_image_center(self, img):
#         image_height, image_width = img.shape[:2]
#         center_x = image_width // 2
#         center_y = image_height // 2
#         return center_x, center_y
    
#     def pixel_to_camera_coordinate(self, u, v, fx, fy, cx, cy, Z):
#         Xc = (u - cx) * Z / fx
#         Yc = (v - cy) * Z / fy
#         Zc = Z
#         return Xc, Yc, Zc
               
#     def find_ball(self,img):
#         hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         hsv_frame = cv2.resize(hsv_frame,(640,360))

#         MIN_H_RED = 0
#         MAX_H_RED = 10
#         mask_frame = cv2.inRange(hsv_frame, (MIN_H_RED / 2, 100, 80), (MAX_H_RED / 2, 255, 255))
#         contours, hierarchy = cv2.findContours(mask_frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#         balls = []
#         bBox = None # <<< NEW: Initialize bBox
#         for cnt in contours:
#             bBox = cv2.boundingRect(cnt)
#             ratio = float(bBox[2]) / float(bBox[3])
#             if ratio > 1.0:
#                 ratio = 1.0 / ratio
#             if ratio > 0.75 and bBox[2] * bBox[3] >= 100:
#                 balls.append(bBox)

#         if len(balls) == 0 or bBox is None: # <<< MODIFIED: Check if bBox exists
#             self.found = False
#             self.notFoundCount += 1
#         else:
#             self.found = True
#             self.notFoundCount = 0

#             self.meas[0, 0] = bBox[0] + bBox[2] / 2
#             self.meas[1, 0] = bBox[1] + bBox[3] / 2
#             self.meas[2, 0] = bBox[2]
#             self.meas[3, 0] = bBox[3]

#             self.center_x, self.center_y = self.get_image_center(img)
#             self.distance = self.Get_Distance(0.06, 402.2965906510785, bBox[2])
#             self.Xc, self.Yc, self.Zc = self.pixel_to_camera_coordinate(self.meas[0, 0], self.meas[1, 0], 402.2965906510785, 402.2965906510785, self.center_x, self.center_y, self.distance)
#             self.angle_rad = math.atan2(self.Xc, self.Zc)
#             self.dis = math.sqrt(pow((self.Xc+0.03*math.sin(self.angle_rad)),2) + pow((self.Zc+0.03*math.cos(self.angle_rad)+0.023),2))

#     def move_to_object(self):
#         # This function is now just for calculation and publishing
#         current_time = time.time()
#         self.dt = current_time - self.last_time
#         self.last_time = current_time

#         self.rot.linear.x, self.rot.angular.z = self.mpc.controlLaw(self.dis, self.angle_rad, self.linear_velocity_xl, 
#                                                                     self.robot_pose_thetal, self.robot_pose_thetaf, self.dt)

#         self.obj_pub.publish(Pose2D(self.dis-0.8, 0, self.angle_rad - np.pi/8))
#         self.pub.publish(self.rot)

#     def stop(self):
#         self.rot.angular.z=0
#         self.rot.linear.x=0
#         self.pub.publish(self.rot)

#     # <<< NEW: This is the main control loop function
#     def run_control_loop(self):
#         """主循环函数"""
#         rospy.loginfo("Tube Controller node is running...")
#         try:
#             while not rospy.is_shutdown():
#                 if self.is_initialized:
#                     self.contr
#                 else:
#                     rospy.loginfo_throttle(1, "Waiting for odometry data...")
#                 self.rate.sleep()
#         except KeyboardInterrupt:
#             rospy.loginfo("Keyboard interrupt detected.")
#         finally:
#             # 在停止前发送一个停止命令以确保机器人停下来
#             stop_cmd = Twist()
#             stop_cmd.linear.x = 0.0
#             stop_cmd.angular.z = 0.0
#             self.cmd_vel_pub.publish(stop_cmd)
#             rospy.loginfo("Emergency stop command sent.")
#             # 绘制结果图
#             # self.plot_results(
#         # # Wait until we have received the first image
#         # if not self.image_received or self.cv_image is None:
#         #     rospy.loginfo("Waiting for image...")
#         #     return

#         # # Process the latest image to find the ball
#         # self.find_ball(self.cv_image)

#         # # Decide whether to move or stop based on the result
#         # if self.found:               	
#         #     self.move_to_object()
#         #     print("move")
#         # else:
#         #     self.stop()
#         #     print("stop")


if __name__ == '__main__':
    rospy.init_node('take_photo', anonymous=False)
    camera = MPC(3,2,0.1)
    
    # <<< MODIFIED: This is the correct way to create a fixed-rate loop
    # Set the loop rate to 10 Hz (0.1 seconds)
    rate = rospy.Rate(10) 
    rospy.sleep(1)  # Give some time to initialize
    while not rospy.is_shutdown():
        # Call the main logic function
        camera.run_control_loop()
        
        # Sleep to maintain the 10 Hz rate
        rate.sleep()

    # Ensure the robot stops when the node is shut down
    camera.stop()