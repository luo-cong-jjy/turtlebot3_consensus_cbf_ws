#!/usr/bin/env python3
"""Use the unchanged baseline trajectory after all CBF nodes/sensors are ready."""
import rospy
from pathlib import Path
import sys
import rospkg
sys.path.insert(0, str(Path(rospkg.RosPack().get_path('tb3_distributed_control'))/'scripts'))
from leader_node import LeaderNode


class ReadyLeader(LeaderNode):
    def publish_leader_state(self,event):
        names=rospy.get_param('/cbf/robot_names',['tb3_'+str(i) for i in range(5)])
        run_id=rospy.get_param('/cbf/run_id','')
        results_ready=not run_id or rospy.has_param('/cbf/results/'+run_id+'/directory')
        if self.start_time is None and (not results_ready or not all(rospy.get_param('/cbf/ready/'+n,False) for n in names)):
            rospy.loginfo_throttle(3.,'CBF leader waiting for five odometry/geometry/MPC readiness flags')
            return
        super().publish_leader_state(event)


if __name__=='__main__':
    node=ReadyLeader()
    rospy.spin()
