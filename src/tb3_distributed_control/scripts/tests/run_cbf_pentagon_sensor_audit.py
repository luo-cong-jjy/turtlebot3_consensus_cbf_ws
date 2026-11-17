#!/usr/bin/env python3
"""Five stationary robots: independently audit each own scan/peer-state pipeline.

This evaluation harness collects scans for checking only. It runs no controller
and publishes no commands or fused perception to robots. It is NOT five-robot
dynamic closed-loop validation. Production geometry functions/parameters reused.
"""
import argparse
from collections import Counter,deque
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import xmlrpc.client
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
from run_cbf_gazebo import free_port,stop


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--duration',type=float,default=8.)
    args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    sources=out/'sources';sources.mkdir();manifest={}
    for rel in ['scripts/tb3_controller_node_cbf.py','scripts/cbf_mpc_core.py','纯仿真代码/mpc_dcbf_cached_solver.py',
                'launch/five_tb3_pentagon_sensor_audit.launch','urdf/turtlebot3_burger.urdf.xacro',
                'urdf/turtlebot3_burger.gazebo.xacro','scripts/tests/run_cbf_pentagon_sensor_audit.py']:
        file=PACKAGE/rel;shutil.copy2(file,sources/file.name);manifest[rel]=hashlib.sha256(file.read_bytes()).hexdigest()
    (sources/'sha256.json').write_text(json.dumps(manifest,indent=2))
    env=os.environ.copy();env.pop('ROS_HOSTNAME',None);env.pop('ROS_NAMESPACE',None)
    env.update(ROS_MASTER_URI='http://127.0.0.1:'+str(free_port()),ROS_IP='127.0.0.1',
        GAZEBO_MASTER_URI='http://127.0.0.1:'+str(free_port()),ROS_HOME=str(out/'ros'),ROS_LOG_DIR=str(out/'roslog'),
        GAZEBO_MODEL_DATABASE_URI='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    os.environ.update(env);processes=[];streams=[];records=[];failures=Counter();failure=None;graph={}
    def launch(command,name):
        f=(out/(name+'.log')).open('x');streams.append(f)
        process=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process);return process
    try:
        launch(['roscore','-p',env['ROS_MASTER_URI'].rsplit(':',1)[1]],'master')
        server=xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
        for _ in range(150):
            try:
                if server.getPid('/pentagon_audit')[0]==1:break
            except (OSError,xmlrpc.client.Error):pass
            time.sleep(.1)
        else:raise RuntimeError('Master readiness timeout')
        import rospy
        import tf2_ros
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import LaserScan
        from tf.transformations import euler_from_quaternion
        import tb3_controller_node_cbf as geometry
        from mpc_dcbf_cached_solver import barrier_values
        rospy.init_node('pentagon_sensor_auditor',anonymous=True,disable_signals=True)
        rospy.set_param('/use_sim_time',True)
        buffer=tf2_ros.Buffer(cache_time=rospy.Duration(5.));listener=tf2_ros.TransformListener(buffer)
        lock=threading.RLock();histories={i:deque(maxlen=200) for i in range(5)};scans={};processed={};subscribers=[]
        def odom(msg,i):
            q=msg.pose.pose.orientation;r=msg.pose.pose.position
            state=np.array([r.x,r.y,euler_from_quaternion([q.x,q.y,q.z,q.w])[2]])
            with lock:histories[i].append((msg.header.stamp.to_sec(),state,np.array([msg.twist.twist.linear.x,msg.twist.twist.angular.z])))
        def scan(msg,i):
            with lock:scans[i]=msg
        for i in range(5):
            subscribers.extend([rospy.Subscriber('/tb3_%d/odom'%i,Odometry,odom,callback_args=i,queue_size=1),
                                rospy.Subscriber('/tb3_%d/scan'%i,LaserScan,scan,callback_args=i,queue_size=1)])
        launch(['roslaunch','tb3_distributed_control','five_tb3_pentagon_sensor_audit.launch'],'launch')
        began=None;wall=time.monotonic();nominal=.4*np.column_stack((np.cos(2*np.pi*np.arange(5)/5),np.sin(2*np.pi*np.arange(5)/5)))
        with (out/'measured_inputs.jsonl').open('x') as inputs:
            while time.monotonic()-wall<180.:
                with lock:packets=dict(scans);history={i:list(h) for i,h in histories.items()}
                if len(packets)<5 or any(not h for h in history.values()):time.sleep(.03);continue
                if began is None:began=rospy.get_time()
                if rospy.get_time()-began>=args.duration:break
                for i,msg in packets.items():
                    stamp=msg.header.stamp.to_sec()
                    if stamp<=0 or stamp-processed.get(i,-1.)<.2-1e-6:continue
                    poses={j:geometry.odometry_at_stamp(h,stamp,.5) for j,h in history.items()}
                    if any(v is None for v in poses.values()):failures['missing_aligned_odom']+=1;continue
                    frame=msg.header.frame_id.lstrip('/')
                    if '/' not in frame:frame='tb3_%d/'%i+frame
                    try:
                        tr=buffer.lookup_transform('world',frame,msg.header.stamp,rospy.Duration(.05))
                        trn=tr.transform.translation;q=tr.transform.rotation
                        xyz=[trn.x,trn.y,trn.z];quaternion=[q.x,q.y,q.z,q.w]
                        points=geometry.scan_points(msg.ranges,msg.angle_min,msg.angle_increment,max(.12,msg.range_min),min(3.5,msg.range_max))
                        world=geometry.transform_points(points,xyz,quaternion)
                        xy=geometry.voxel_downsample(world[:,:2],.02)
                        rest,groups=geometry.split_robot_returns(xy,poses,i,.16)
                        obstacles=[];pairs=[]
                        for j in range(5):
                            if j==i:continue
                            peer_points=groups.get(j,np.empty((0,2)));shape=None
                            if len(peer_points)>=3:
                                shape=geometry.fit_peer_shape(peer_points,poses[j][0],.2,.025,.03,.02,80)
                                shape['stamp']=stamp
                            obs=geometry.predicted_peer_envelope(j,stamp,poses[j][0],poses[j][1],stamp,.1,10,.2,shape,.5)
                            obstacles.append(obs)
                            pairs.append(dict(peer=j,points=len(peer_points),fitted=shape is not None,
                                a=obs['a'],b=obs['b'],theta=obs['theta'],outer_boundary=max(obs['a'],obs['b'])*(1+.12/obs['b'])))
                        h=barrier_values(np.tile(poses[i][0],(11,1)),obstacles,.12)
                        drift=max(float(np.linalg.norm(poses[j][0][:2]-nominal[j])) for j in range(5))
                        records.append(dict(observer=i,stamp=stamp,points=len(xy),unassigned_points=len(rest),
                            min_model_h=float(h.min()),maximum_position_deviation=drift,pairs=pairs))
                        inputs.write(json.dumps(dict(observer=i,stamp=stamp,angle_min=msg.angle_min,angle_increment=msg.angle_increment,
                            range_min=msg.range_min,range_max=msg.range_max,ranges=[float(v) if math.isfinite(v) else None for v in msg.ranges],
                            translation=xyz,quaternion=quaternion,poses={str(j):dict(state=v[0].tolist(),velocity=v[1].tolist()) for j,v in poses.items()}))+'\n')
                        processed[i]=stamp
                    except (tf2_ros.LookupException,tf2_ros.ConnectivityException,tf2_ros.ExtrapolationException,ValueError,np.linalg.LinAlgError) as exc:
                        failures[type(exc).__name__]+=1
                time.sleep(.01)
            else:raise RuntimeError('Sensor audit wall timeout')
        _,_,system=server.getSystemState('/pentagon_audit');graph={topic:nodes for topic,nodes in system[0]}
        for sub in subscribers:sub.unregister()
    except Exception as exc:failure=str(exc)
    finally:
        for process in reversed(processes):stop(process)
        for stream in streams:stream.close()
    observed=Counter((r['observer'],p['peer']) for r in records for p in r['pairs'] if p['points']>=3)
    no_commands=not any(t.endswith('/cmd_vel') and v for t,v in graph.items())
    checks=dict(no_controller_commands=no_commands,all_observers_sampled=all(sum(r['observer']==i for r in records)>=10 for i in range(5)),
        every_adjacent_peer_observed=all(observed[(i,(i+offset)%5)]>=3 for i in range(5) for offset in (-1,1)),
        no_robot_returns_left_as_external=bool(records) and all(r['unassigned_points']==0 for r in records),
        peer_model_compatible=bool(records) and min(r['min_model_h'] for r in records)>0,
        stationary_pose=bool(records) and max(r['maximum_position_deviation'] for r in records)<.02)
    result=dict(passed=failure is None and all(checks.values()),failure=failure,checks=checks,frames=len(records),
        minimum_peer_model_h=min((r['min_model_h'] for r in records),default=None),
        maximum_peer_outer_boundary=max((p['outer_boundary'] for r in records for p in r['pairs']),default=None),
        discarded_samples=dict(failures),directed_peer_observations={str(k):v for k,v in observed.items()},
        scope=__doc__,source_sha256=manifest)
    (out/'frame_metrics.json').write_text(json.dumps(records,indent=2));(out/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
