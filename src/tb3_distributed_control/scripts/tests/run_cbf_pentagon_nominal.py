#!/usr/bin/env python3
"""Five-robot empty-world nominal tracking test; NOT new five-robot avoidance."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xmlrpc.client
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
from run_cbf_gazebo import free_port,stop


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);sources=out/'sources';sources.mkdir()
    import shutil
    hashes={}
    for rel in ['scripts/tb3_controller_node_cbf.py','scripts/cbf_mpc_core.py','scripts/leader_node.py',
                'scripts/cbf_experiment_results.py','纯仿真代码/mpc_dcbf_cached_solver.py',
                'launch/five_tb3_pentagon_nominal.launch','launch/multi_turtlebot3_consensus_gazebo_cbf.launch',
                'scripts/tests/run_cbf_pentagon_nominal.py']:
        f=PACKAGE/rel;shutil.copy2(f,sources/f.name);hashes[rel]=hashlib.sha256(f.read_bytes()).hexdigest()
    (sources/'sha256.json').write_text(json.dumps(hashes,indent=2))
    env=os.environ.copy();env.pop('ROS_HOSTNAME',None);env.pop('ROS_NAMESPACE',None)
    env.update(ROS_MASTER_URI='http://127.0.0.1:'+str(free_port()),ROS_IP='127.0.0.1',GAZEBO_MASTER_URI='http://127.0.0.1:'+str(free_port()),
        ROS_HOME=str(out/'ros'),ROS_LOG_DIR=str(out/'roslog'),GAZEBO_MODEL_DATABASE_URI='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg')
    os.environ.update(env);processes=[];streams=[];truth=[];origins=[];finished=[];audit={};failure=None
    def launch(command,name):
        stream=(out/(name+'.log')).open('x');streams.append(stream)
        process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process);return process
    try:
        launch(['roscore','-p',env['ROS_MASTER_URI'].rsplit(':',1)[1]],'master');server=xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
        for _ in range(150):
            try:
                if server.getPid('/pentagon_nominal')[0]==1:break
            except (OSError,xmlrpc.client.Error):pass
            time.sleep(.1)
        else:raise RuntimeError('Master timeout')
        import rospy
        from gazebo_msgs.msg import ModelStates
        from std_msgs.msg import Bool
        from tb3_distributed_control.msg import LeaderHistoryData
        rospy.set_param('/use_sim_time',True);rospy.init_node('pentagon_nominal_evaluator',anonymous=True,disable_signals=True)
        def leader(msg):
            if not origins:origins.append(msg.ros_timestamp-msg.trajectory_time)
        def models(msg):
            if not origins or not all('tb3_'+str(i) in msg.name for i in range(5)):return
            now=rospy.get_time()
            if truth and now-truth[-1]['ros_time']<.045:return
            row=dict(ros_time=now,time=now-origins[0])
            for i in range(5):
                point=msg.pose[msg.name.index('tb3_'+str(i))].position;row['tb3_%d_x'%i]=point.x;row['tb3_%d_y'%i]=point.y
            truth.append(row)
        subs=[rospy.Subscriber('/leader/history',LeaderHistoryData,leader,queue_size=1),rospy.Subscriber('/gazebo/model_states',ModelStates,models,queue_size=1),
              rospy.Subscriber('/system/shutdown',Bool,lambda m:finished.append(time.monotonic()) if m.data else None,queue_size=1)]
        process=launch(['roslaunch','tb3_distributed_control','five_tb3_pentagon_nominal.launch','gui:=false','open_rviz:=false','total_time:=30','output_dir:='+str(out)],'launch')
        begin=time.monotonic()
        while time.monotonic()-begin<300:
            if origins and not audit:
                _,_,graph=server.getSystemState('/pentagon_nominal');audit={topic:nodes for topic,nodes in graph[1] if any('/tb3_controller_cbf_' in n for n in nodes)}
            if finished and (out/'RESULTS_SAVED').exists():break
            if process.poll() is not None:raise RuntimeError('Launch exited early')
            if finished and time.monotonic()-finished[0]>60:raise RuntimeError('Export timeout')
            time.sleep(.1)
        else:raise RuntimeError('Nominal test timeout')
    except Exception as exc:failure=str(exc)
    finally:
        for process in reversed(processes):stop(process)
        for stream in streams:stream.close()
    metrics={};checks={};minimum_peer=None;maximum_pair_error=None
    if truth:
        with (out/'gazebo_truth.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(truth[0]));w.writeheader();w.writerows(truth)
        t=np.array([r['time'] for r in truth]);positions=np.array([[[r['tb3_%d_x'%i],r['tb3_%d_y'%i]] for r in truth] for i in range(5)])
        offsets=.4*np.column_stack((np.cos(2*np.pi*np.arange(5)/5),np.sin(2*np.pi*np.arange(5)/5)))
        selected=(t>=5)&(t<=29)
        if not selected.any():raise ValueError('Independent truth does not cover the frozen simulation-time window')
        endpoint=int(np.argmin(abs(t-30)))
        minimum_peer=min(float(np.linalg.norm(positions[i]-positions[j],axis=1).min()-.26) for i in range(5) for j in range(i))
        maximum_pair_error=max(float(np.linalg.norm((positions[i]-positions[j])[:, :]-offsets[i]+offsets[j],axis=1)[selected].max()) for i in range(5) for j in range(i))
        for i in range(5):
            expected=t[:,None]*[.15,0]+offsets[i];error=np.linalg.norm(positions[i]-expected,axis=1)
            rows=list(csv.DictReader((out/('tb3_%d.csv'%i)).open()))
            faults=sum(5<=float(r.get('evaluation_time',r['time']))<=29 and r['status'] not in ('ok','feasible_iterate','validated_one_step') for r in rows)
            metrics[str(i)]=dict(max_position_error_after_warmup=float(error[selected].max()),final_position_error=float(error[endpoint]),fault_stops=faults,
                max_external_constraints=max(int(r['static_constraints']) for r in rows),
                max_peer_constraints=max(int(r['peer_constraints']) for r in rows))
        checks=dict(full_duration=bool(finished) and abs(t[endpoint]-30)<=.15,peer_proxy_positive=minimum_peer>0,
            pair_shape_error_within_frozen_limit=maximum_pair_error<=.10,
            robot_errors_within_frozen_limit=all(m['max_position_error_after_warmup']<=.10 and m['final_position_error']<=.05 for m in metrics.values()),
            no_fault_stop=all(m['fault_stops']==0 for m in metrics.values()),no_external_robot_duplicates=all(m['max_external_constraints']==0 for m in metrics.values()),
            own_scan_only=all(sorted(topic for topic,nodes in audit.items() if '/tb3_controller_cbf_'+str(i) in nodes and topic.endswith('/scan'))==['/tb3_%d/scan'%i] for i in range(5)),
            no_truth_input=bool(audit) and not any(t in audit for t in ('/gazebo/model_states','/cbf/scenario/truth')))
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(passed=failure is None and bool(checks) and all(checks.values()),failure=failure,checks=checks,robots=metrics,
        minimum_peer_proxy_clearance=minimum_peer,maximum_relative_shape_error_after_warmup=maximum_pair_error,
        frozen_limits=dict(duration=30.,window=[5.,29.],maximum_position_and_pair_error=.10,maximum_final_position_error=.05),
        source_hashes=hashes,scope=__doc__)
    (out/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
