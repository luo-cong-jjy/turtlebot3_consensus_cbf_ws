#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import time

def rotate_robots():
    # 初始化节点
    rospy.init_node('tb3_rotator', anonymous=True)
    
    # 创建5个发布者
    publishers = []
    for i in range(5):
        pub = rospy.Publisher(f'tb3_{i}/cmd_vel', Twist, queue_size=10)
        publishers.append(pub)
    
    # 旋转指令（逆时针，0.5rad/s）
    rotate_cmd = Twist()
    rotate_cmd.angular.z = 0.5
    
    # 停止指令
    stop_cmd = Twist()
    
    # 发布旋转指令（10Hz频率，持续约13秒）
    rate = rospy.Rate(10)
    start_time = time.time()
    
    while not rospy.is_shutdown() and (time.time() - start_time) < 13:
        for pub in publishers:
            pub.publish(rotate_cmd)
        rate.sleep()
    
    # 发布停止指令
    for pub in publishers:
        pub.publish(stop_cmd)
    rospy.loginfo("所有TurtleBot3旋转完成并停止")

if __name__ == '__main__':
    try:
        rotate_robots()
    except rospy.ROSInterruptException:
        pass