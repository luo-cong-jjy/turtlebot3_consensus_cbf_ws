#!/usr/bin/env python3
"""Isolated real Gazebo test; ground truth is evaluated, never fed to control."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import xmlrpc.client
import xml.etree.ElementTree as ET
import numpy as np
from run_cbf_gazebo import free_port,stop


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scenario',default='crossing',choices=['crossing','head_on','overtaking','maneuver','mixed','accelerating_crossing','decelerating_crossing','stop_go_crossing'])
    parser.add_argument('--robots',type=int,default=1,choices=[1,5])
    parser.add_argument('--duration',type=float,default=32.)
    parser.add_argument('--prediction-mode',choices=['cv','frozen','ca'],default='cv')
    parser.add_argument('--output',required=True)
    parser.add_argument('--wall-timeout',type=float,default=240.)
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    package=Path(__file__).resolve().parents[2]
    sources=out/'sources';sources.mkdir(exist_ok=True)
    manifest={}
    for relative in ['scripts/tb3_controller_node_cbf.py','scripts/cbf_mpc_core.py','scripts/leader_node_cbf.py',
            'scripts/cbf_dynamic_scenario.py','scripts/cbf_experiment_results.py','config/cbf_dynamic_scenarios.yaml',
            '纯仿真代码/mpc_dcbf_cached_solver.py','launch/multi_turtlebot3_consensus_gazebo_cbf.launch',
            'launch/single_turtlebot3_cbf_dynamic.launch','launch/multi_turtlebot3_consensus_gazebo_cbf_dynamic.launch',
            'scripts/tests/run_cbf_dynamic_gazebo.py']:
        p=package/relative;shutil.copy2(p,sources/p.name);manifest[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
    (sources/'sha256.json').write_text(json.dumps(manifest,indent=2))
    env=os.environ.copy();env.pop('ROS_HOSTNAME',None);env.pop('ROS_NAMESPACE',None)
    env.update(ROS_MASTER_URI='http://127.0.0.1:'+str(free_port()),ROS_IP='127.0.0.1',
        GAZEBO_MASTER_URI='http://127.0.0.1:'+str(free_port()),ROS_HOME=str(out/'ros'),ROS_LOG_DIR=str(out/'roslog'),
        GAZEBO_MODEL_DATABASE_URI='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg')
    os.environ.update(env);processes=[];streams=[];truth=[];begins=[];completed=[];last={};audit={}
    def launch(command,name):
        stream=(out/(name+'.log')).open('w');streams.append(stream)
        process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process);return process
    failure=None;wall=time.monotonic()
    try:
        launch(['roscore','-p',env['ROS_MASTER_URI'].rsplit(':',1)[1]],'master')
        server=xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
        for _ in range(150):
            try:
                if server.getPid('/cbf_dynamic_test')[0]==1:break
            except (OSError,xmlrpc.client.Error):pass
            time.sleep(.1)
        else:raise RuntimeError('ROS master startup timeout')
        import rospy
        rospy.set_param('/use_sim_time',True)
        from gazebo_msgs.msg import ModelStates
        from std_msgs.msg import Bool,String
        from tb3_distributed_control.msg import LeaderHistoryData
        rospy.init_node('cbf_dynamic_evaluator',anonymous=True,disable_signals=True)
        def lead(msg):
            if not begins:begins.append(msg.ros_timestamp-msg.trajectory_time)
        def models(msg):
            stamp=rospy.get_time()
            if truth and stamp-truth[-1]['ros_time']<.045:return
            if not begins or not all('tb3_'+str(i) in msg.name for i in range(args.robots)):return
            row=dict(ros_time=stamp,time=stamp-begins[0])
            for name in msg.name:
                if name.startswith('tb3_') or name.startswith('cbf_dynamic_'):
                    p=msg.pose[msg.name.index(name)].position
                    row[name+'_x']=p.x;row[name+'_y']=p.y
            truth.append(row)
        def diagnostic(msg,i):last[i]=json.loads(msg.data)
        subscribers=[rospy.Subscriber('/leader/history',LeaderHistoryData,lead,queue_size=1),
            rospy.Subscriber('/gazebo/model_states',ModelStates,models,queue_size=1),
            rospy.Subscriber('/system/shutdown',Bool,lambda m:completed.append(time.monotonic()) if m.data else None)]
        for i in range(args.robots):subscribers.append(rospy.Subscriber('/tb3_%d/cbf_diagnostics'%i,String,diagnostic,callback_args=i,queue_size=1))
        name='single_turtlebot3_cbf_dynamic.launch' if args.robots==1 else 'multi_turtlebot3_consensus_gazebo_cbf_dynamic.launch'
        process=launch(['roslaunch','tb3_distributed_control',name,'gui:=false','open_rviz:=false',
            'scenario:='+args.scenario,'total_time:='+str(args.duration),'output_dir:='+str(out),
            'prediction_mode:='+args.prediction_mode],'launch')
        report=0.
        while time.monotonic()-wall<args.wall_timeout:
            if begins and not audit:
                _,_,graph=server.getSystemState('/cbf_dynamic_test')
                audit={topic:members for topic,members in graph[1] if any('/tb3_controller_cbf_' in n for n in members)}
            if completed and (out/'RESULTS_SAVED').exists():break
            if completed and time.monotonic()-completed[0]>45:raise RuntimeError('Result export timeout')
            if process.poll() is not None:raise RuntimeError('roslaunch exited')
            if not begins and time.monotonic()-wall>75:raise RuntimeError('Sensor/MPC/scenario readiness timeout')
            if time.monotonic()-report>10:
                print('Dynamic Gazebo %s: %s'%(args.scenario,{i:(round(d['time'],1),d['status']) for i,d in last.items()}),flush=True);report=time.monotonic()
            time.sleep(.2)
        else:raise RuntimeError('Experiment wall timeout')
    except Exception as exc:failure=str(exc)
    finally:
        for p in reversed(processes):stop(p)
        for stream in streams:stream.close()
    result=dict(scenario=args.scenario,robots=args.robots,prediction_mode=args.prediction_mode,failure=failure,
        completed=bool(completed),controller_subscriptions=audit,robot_statistics={})
    if truth:
        fields=sorted(set().union(*(r.keys() for r in truth)))
        with (out/'gazebo_truth.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(truth)
        specs=json.loads((out/'dynamic_scenario.json').read_text())['specifications']
        clearance=1e9;pair=1e9;static_clearance=1e9
        metadata=json.loads((out/'run_metadata.json').read_text())
        links=ET.parse(metadata['world_file']).findall('.//model[@name="obstacles"]/link')
        for row in truth:
            for i in range(args.robots):
                pos=np.array([row['tb3_%d_x'%i],row['tb3_%d_y'%i]])
                for link in links:
                    pose=[float(x) for x in link.findtext('pose').split()]
                    geometry=link.find('collision/geometry');delta=pos-pose[:2]
                    if geometry.find('cylinder') is not None:
                        distance=np.linalg.norm(delta)-float(geometry.findtext('cylinder/radius'))
                    else:
                        a,b,_=[float(x) for x in geometry.findtext('box/size').split()]
                        c,s=np.cos(pose[5]),np.sin(pose[5]);q=np.abs(delta@np.array([[c,-s],[s,c]]))-[a/2,b/2]
                        distance=np.linalg.norm(np.maximum(q,0))+min(float(max(q)),0.)
                    static_clearance=min(static_clearance,float(distance)-.13)
                for spec in specs:
                    if spec['name']+'_x' not in row:continue
                    actor=np.array([row[spec['name']+'_x'],row[spec['name']+'_y']])
                    clearance=min(clearance,float(np.linalg.norm(pos-actor)-.13-spec['radius']))
                for j in range(i):pair=min(pair,float(np.linalg.norm(pos-[row['tb3_%d_x'%j],row['tb3_%d_y'%j]])-.26))
        result.update(body_radius_proxy=.13,dynamic_clearance_min=clearance,peer_clearance_min=pair,
            static_clearance_min=static_clearance,
            truth_samples=len(truth),clearance_note='20 Hz actual Gazebo positions; circular body proxy, not a continuous collision proof.')
    for i in range(args.robots):
        file=out/('tb3_%d.csv'%i)
        if not file.exists():continue
        rows=list(csv.DictReader(file.open()))
        if not rows:continue
        times=np.array([float(r['solve_ms']) for r in rows]);statuses={s:sum(r['status']==s for r in rows) for s in set(r['status'] for r in rows)}
        result['robot_statistics'][str(i)]=dict(rows=len(rows),last_time=float(rows[-1]['time']),
            statuses=statuses,timing_p95_ms=float(np.percentile(times,95)),timing_max_ms=float(times.max()),
            final_error=float(np.hypot(float(rows[-1]['consensus_error_x']),float(rows[-1]['consensus_error_y']))),
            dynamic_cycles=sum(float(r['dynamic_tracks'])>0 for r in rows),
            max_estimated_speed=max(float(r['max_obstacle_speed']) for r in rows))
    result['no_truth_input_passed']=bool(audit) and not any(topic in audit for topic in ('/gazebo/model_states','/cbf/scenario/truth','/gazebo/set_model_state'))
    result['passed']=(failure is None and bool(completed) and (out/'RESULTS_SAVED').exists()
        and len(result['robot_statistics'])==args.robots and result.get('dynamic_clearance_min',-1)>0.
        and result.get('peer_clearance_min',-1)>0. and result.get('static_clearance_min',-1)>0.
        and result['no_truth_input_passed']
        and all(s['last_time']>args.duration-.5 and s['dynamic_cycles']>0 and s['final_error']<.25
                for s in result['robot_statistics'].values()))
    (out/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
