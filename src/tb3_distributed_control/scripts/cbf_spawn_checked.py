#!/usr/bin/env python3
"""Serialize Gazebo factory requests and retry missing models during setup.

No model pose/velocity is read or sent to the controller. Existing models are
not deleted, moved, or silently adopted on initial entry.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import tempfile
import time


def ensure_spawned(spawn,exists,pause=time.sleep,attempts=3):
    if exists():raise RuntimeError('Model already exists before this spawn request')
    last='no response'
    for attempt in range(1,attempts+1):
        if attempt>1 and exists():return attempt-1
        response=spawn();last=response.status_message
        for _ in range(10):
            if exists():return attempt
            pause(.2)
    raise RuntimeError('Model did not appear after %d requests: %s'%(attempts,last))


@contextmanager
def factory_lock(timeout=60.,shutdown=lambda:False):
    master=os.environ.get('ROS_MASTER_URI','http://localhost:11311')
    suffix=hashlib.sha256(master.encode()).hexdigest()[:16]
    path=Path(tempfile.gettempdir())/('cbf_factory_%d_%s.lock'%(os.getuid(),suffix))
    begin=time.monotonic()
    with path.open('a+') as stream:
        while True:
            if shutdown():raise RuntimeError('Shutdown while waiting for Gazebo factory')
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()-begin>timeout:raise RuntimeError('Gazebo factory lock timeout')
                time.sleep(.05)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def main():
    import rospy
    from gazebo_msgs.srv import SpawnModel,GetWorldProperties
    from geometry_msgs.msg import Pose
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from tf.transformations import quaternion_from_euler
    rospy.init_node('spawn_checked')
    parser=argparse.ArgumentParser()
    parser.add_argument('-urdf',action='store_true',required=True)
    parser.add_argument('-model',required=True);parser.add_argument('-param',required=True)
    parser.add_argument('-robot_namespace',default=rospy.get_namespace())
    parser.add_argument('-reference_frame',default='')
    for key in ['x','y','z','R','P','Y']:parser.add_argument('-'+key,type=float,default=0.)
    args=parser.parse_args(rospy.myargv()[1:]);xml=rospy.get_param(args.param)
    pose=Pose();pose.position.x=args.x;pose.position.y=args.y;pose.position.z=args.z
    q=quaternion_from_euler(args.R,args.P,args.Y)
    pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=q
    rospy.wait_for_service('/gazebo/spawn_urdf_model',timeout=60)
    rospy.wait_for_service('/gazebo/get_world_properties',timeout=60)
    spawn=rospy.ServiceProxy('/gazebo/spawn_urdf_model',SpawnModel)
    world=rospy.ServiceProxy('/gazebo/get_world_properties',GetWorldProperties)
    def exists():
        result=world()
        if not result.success:raise RuntimeError(result.status_message)
        return args.model in result.model_names
    with factory_lock(shutdown=rospy.is_shutdown):
        attempts=ensure_spawned(lambda:spawn(args.model,xml,args.robot_namespace,pose,args.reference_frame),exists)
        # Factory insertion can complete before native sensor plugins finish
        # initializing. Do not load the next robot during that interval.
        rospy.wait_for_message('odom',Odometry,timeout=15.)
        rospy.wait_for_message('scan',LaserScan,timeout=15.)
    rospy.loginfo('Verified spawn %s after %d request(s)',args.model,attempts)


if __name__=='__main__':main()
