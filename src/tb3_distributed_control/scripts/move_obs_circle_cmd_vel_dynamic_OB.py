#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates


def main():
    rospy.init_node("move_obs_circle_cmd_vel")

    model_name = rospy.get_param("~model_name", "obs_circle")
    y_min = rospy.get_param("~y_min", -0.5)
    y_max = rospy.get_param("~y_max", 0.5)
    speed = rospy.get_param("~speed", 0.2)
    rate_hz = rospy.get_param("~rate", 30.0)

    pub = rospy.Publisher(f"/{model_name}/cmd_vel", Twist, queue_size=1)

    state = {"y": 0.0, "ready": False}

    def states_cb(msg):
        if model_name in msg.name:
            idx = msg.name.index(model_name)
            state["y"] = msg.pose[idx].position.y
            state["ready"] = True

    rospy.Subscriber("/gazebo/model_states", ModelStates, states_cb, queue_size=1)

    rate = rospy.Rate(rate_hz)
    direction = 1.0

    while not rospy.is_shutdown():
        if state["ready"]:
            if state["y"] >= y_max:
                direction = -1.0
            elif state["y"] <= y_min:
                direction = 1.0

        cmd = Twist()
        cmd.linear.y = direction * speed
        pub.publish(cmd)
        rate.sleep()


if __name__ == "__main__":
    main()
