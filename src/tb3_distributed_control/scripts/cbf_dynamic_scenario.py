#!/usr/bin/env python3
"""Prescribed moving Gazebo cylinders; truth is logged ONLY for evaluation.

This node is not part of the controller/perception path. It never sends obstacle
poses, velocities, model names or trajectories to the control nodes.
"""
import csv
import json
import math
from pathlib import Path
import threading

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
from std_msgs.msg import String, Bool
from tb3_distributed_control.msg import LeaderHistoryData


def trajectory(spec,t,shift_y=0.):
    t=max(0.,float(t));x,y=spec['start'];vx,vy=spec['velocity']
    if 'acceleration_segments' in spec:
        remaining=t
        for segment in spec['acceleration_segments']:
            duration=float(segment['duration']);ax,ay=segment['acceleration']
            if not (math.isfinite(duration) and duration>0 and math.isfinite(ax) and math.isfinite(ay)):
                raise ValueError('Invalid prescribed acceleration segment')
            step=min(remaining,duration)
            x+=vx*step+.5*ax*step*step;y+=vy*step+.5*ay*step*step
            vx+=ax*step;vy+=ay*step;remaining-=step
        return x+vx*remaining,y+shift_y+vy*remaining,vx,vy
    amplitude=spec.get('sine_x',0.);frequency=spec.get('frequency',0.);phase=spec.get('phase',0.)
    return (x+vx*t+amplitude*math.sin(frequency*t+phase),y+shift_y+vy*t,
            vx+amplitude*frequency*math.cos(frequency*t+phase),vy)


def cylinder_sdf(name,radius):
    return '''<sdf version="1.6"><model name="%s"><static>false</static><link name="body">
    <kinematic>true</kinematic><gravity>false</gravity>
    <inertial><mass>10</mass><inertia><ixx>1</ixx><iyy>1</iyy><izz>1</izz></inertia></inertial>
    <collision name="collision"><geometry><cylinder><radius>%f</radius><length>0.8</length></cylinder></geometry></collision>
    <visual name="visual"><geometry><cylinder><radius>%f</radius><length>0.8</length></cylinder></geometry>
    <material><ambient>0.9 0.25 0.05 1</ambient><diffuse>0.9 0.25 0.05 1</diffuse></material></visual>
    </link></model></sdf>'''%(name,radius,radius)


class Scenario:
    def __init__(self):
        self.lock=threading.RLock();self.start=None;self.closed=False
        self.case=rospy.get_param('/cbf/dynamic_scenario','crossing')
        self.specs=rospy.get_param('/cbf/scenarios/'+self.case)
        self.shift_y=.8 if rospy.get_param('/cbf/trajectory_type','line')!='line' else 0.
        self.publisher=rospy.Publisher('/gazebo/set_model_state',ModelState,queue_size=10)
        self.truth=rospy.Publisher('/cbf/scenario/truth',String,queue_size=10)
        rospy.set_param('/cbf/scenario/ready',False)
        rospy.wait_for_service('/gazebo/spawn_sdf_model',timeout=60)
        spawn=rospy.ServiceProxy('/gazebo/spawn_sdf_model',SpawnModel)
        for spec in self.specs:
            x,y,_,_=trajectory(spec,0.,self.shift_y)
            pose=Pose();pose.position.x=x;pose.position.y=y;pose.position.z=.4;pose.orientation.w=1.
            response=spawn(spec['name'],cylinder_sdf(spec['name'],spec['radius']),'',pose,'world')
            if not response.success:raise RuntimeError(response.status_message)
        key='/cbf/results/'+rospy.get_param('/cbf/run_id')+'/directory'
        while not rospy.has_param(key) and not rospy.is_shutdown():rospy.sleep(.05)
        self.path=Path(rospy.get_param(key));self.stream=(self.path/'dynamic_truth.csv').open('w',newline='')
        self.writer=csv.DictWriter(self.stream,fieldnames=['ros_time','time','name','x','y','vx','vy','radius'])
        self.writer.writeheader();self.last_flush=0.
        (self.path/'dynamic_scenario.json').write_text(json.dumps(dict(scenario=self.case,
            specifications=self.specs,shift_y=self.shift_y,source='Prescribed kinematic Gazebo movers; evaluation only'),indent=2))
        self.subscriber=rospy.Subscriber('/leader/history',LeaderHistoryData,self.on_leader,queue_size=1)
        self.stop_sub=rospy.Subscriber('/system/shutdown',Bool,self.on_stop,queue_size=1)
        self.timer=rospy.Timer(rospy.Duration(.05),self.tick,reset=True)
        rospy.on_shutdown(self.close);rospy.set_param('/cbf/scenario/ready',True)
        rospy.loginfo('CBF dynamic scenario %s: %d independent moving cylinders',self.case,len(self.specs))

    def on_leader(self,msg):
        with self.lock:
            if self.start is None:self.start=msg.ros_timestamp-msg.trajectory_time

    def on_stop(self,msg):
        if msg.data:self.close()

    def tick(self,event):
        with self.lock:
            if self.closed:return
            now=rospy.get_time();t=max(0.,now-self.start) if self.start is not None else 0.
            rows=[]
            for spec in self.specs:
                x,y,vx,vy=trajectory(spec,t,self.shift_y)
                state=ModelState();state.model_name=spec['name'];state.reference_frame='world'
                state.pose.position.x=x;state.pose.position.y=y;state.pose.position.z=.4;state.pose.orientation.w=1.
                # Pose updates prescribe the trajectory. Zero twist avoids extra
                # free integration between updates of this kinematic actor.
                self.publisher.publish(state)
                row=dict(ros_time=now,time=t,name=spec['name'],x=x,y=y,vx=vx,vy=vy,radius=spec['radius'])
                rows.append(row);self.writer.writerow(row)
            self.truth.publish(String(data=json.dumps(dict(stamp=now,time=t,obstacles=rows))))
            if now-self.last_flush>1.:self.stream.flush();self.last_flush=now

    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True;self.stream.flush();self.stream.close()


if __name__=='__main__':
    rospy.init_node('cbf_dynamic_scenario');scenario=Scenario();rospy.spin()
