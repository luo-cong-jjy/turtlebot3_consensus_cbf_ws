#!/usr/bin/env python3
"""Evaluation-only model-pose recorder. Publishes no control or obstacle input."""
import csv
import json
from pathlib import Path
import sys
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Bool
from tb3_distributed_control.msg import LeaderHistoryData
from evaluate_cbf_active_motion import evaluate_run


class Monitor:
    def __init__(self):
        self.lock=threading.RLock();self.origin=None;self.closed=False;self.last=-1.;self.worker=None
        if len(rospy.get_param('/cbf/robot_names'))!=1:raise ValueError('Active monitor is single robot only')
        scenario=rospy.get_param('/cbf/dynamic_scenario');specs=rospy.get_param('/cbf/scenarios/'+scenario)
        if len(specs)!=1:raise ValueError('Active monitor requires one actor')
        self.actor=specs[0]['name']
        while not rospy.is_shutdown():
            run_id=rospy.get_param('/cbf/run_id','');key='/cbf/results/'+run_id+'/directory'
            if run_id and rospy.has_param(key):break
            rospy.sleep(.05)
        self.path=Path(rospy.get_param(key));self.stream=(self.path/'active_truth.csv').open('x',newline='')
        fields=['ros_time','time','tb3_0_x','tb3_0_y',self.actor+'_x',self.actor+'_y']
        self.writer=csv.DictWriter(self.stream,fieldnames=fields);self.writer.writeheader()
        self.subs=[rospy.Subscriber('/leader/history',LeaderHistoryData,self.leader,queue_size=1),
            rospy.Subscriber('/gazebo/model_states',ModelStates,self.models,queue_size=1),
            rospy.Subscriber('/system/shutdown',Bool,lambda m:self.close() if m.data else None,queue_size=1)]
        rospy.on_shutdown(self.close)

    def leader(self,msg):
        with self.lock:
            if self.origin is None:self.origin=msg.ros_timestamp-msg.trajectory_time

    def models(self,msg):
        with self.lock:
            if self.closed or self.origin is None:return
            now=rospy.get_time()
            if now-self.last<.045 or not all(n in msg.name for n in ('tb3_0',self.actor)):return
            row=dict(ros_time=now,time=now-self.origin)
            for name in ('tb3_0',self.actor):
                pose=msg.pose[msg.name.index(name)].position;row[name+'_x']=pose.x;row[name+'_y']=pose.y
            self.writer.writerow(row);self.stream.flush();self.last=now

    def report(self):
        deadline=time.monotonic()+30.
        while time.monotonic()<deadline and not rospy.is_shutdown() and not (self.path/'RESULTS_SAVED').exists():time.sleep(.05)
        try:result=evaluate_run(self.path)
        except Exception as exc:result=dict(passed=False,failed_checks=['evaluation_error'],error=str(exc))
        with (self.path/'active_motion_report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
        rospy.loginfo('Active avoidance evaluation: passed=%s, failed=%s',result['passed'],result.get('failed_checks'))

    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True;self.stream.flush();self.stream.close()
            self.worker=threading.Thread(target=self.report);self.worker.start()


if __name__=='__main__':
    rospy.init_node('cbf_active_monitor');monitor=Monitor();rospy.spin()
