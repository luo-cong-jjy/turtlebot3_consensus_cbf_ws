#!/usr/bin/env python3
"""Run the actual CBF launch on isolated ROS/Gazebo masters and record truth.

Gazebo/world truth is used ONLY here for evaluation, never sent to controllers.
Only this script's own process groups are stopped. GUI is disabled in the test.
"""
import argparse
import csv
import json
import hashlib
import shutil
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import xmlrpc.client
import xml.etree.ElementTree as ET

import numpy as np


def free_port():
    with socket.socket() as stream:
        stream.bind(('127.0.0.1',0));return stream.getsockname()[1]


def stop(process):
    if process.poll() is not None:return
    for sig,timeout in [(signal.SIGINT,10),(signal.SIGTERM,5),(signal.SIGKILL,2)]:
        try:os.killpg(process.pid,sig);process.wait(timeout=timeout);return
        except subprocess.TimeoutExpired:continue
        except ProcessLookupError:return


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--duration',type=float,default=180.)
    parser.add_argument('--output',required=True)
    parser.add_argument('--wall-timeout',type=float,default=1000.)
    parser.add_argument('--default-output',action='store_true',help='Test automatic data_collect directory creation')
    parser.add_argument('--geometry-debug',action='store_true')
    parser.add_argument('--geometry-observer',type=int,default=0)
    parser.add_argument('--legacy-solver',action='store_true')
    parser.add_argument('--nlp-budget-ms',type=float,default=65.)
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    package=Path(__file__).resolve().parents[2]
    sources=out/'sources';sources.mkdir(exist_ok=True)
    manifest={}
    for relative in ['scripts/tb3_controller_node_cbf.py','scripts/cbf_mpc_core.py',
                     'scripts/leader_node_cbf.py','scripts/cbf_experiment_results.py','纯仿真代码/mpc_dcbf_cached_solver.py',
                     '纯仿真代码/mpc_dcbf_static_obstacles.py','rviz/tb3_cbf.rviz',
                     'launch/multi_turtlebot3_consensus_gazebo_cbf.launch']:
        path=package/relative;shutil.copy2(path,sources/path.name)
        manifest[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    (sources/'sha256.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    env=os.environ.copy();env.pop('ROS_HOSTNAME',None);env.pop('ROS_NAMESPACE',None)
    port=free_port()
    env.update(ROS_MASTER_URI='http://127.0.0.1:'+str(port),ROS_IP='127.0.0.1',
        ROS_HOME=str(out/'ros'),ROS_LOG_DIR=str(out/'roslog'),
        GAZEBO_MASTER_URI='http://127.0.0.1:'+str(free_port()),GAZEBO_MODEL_DATABASE_URI='',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg')
    os.environ.pop('ROS_HOSTNAME',None);os.environ.pop('ROS_NAMESPACE',None);os.environ.update(env)
    processes=[];streams=[];truth=[];latest={};completed=[];leader_start=[]
    marker_evidence={str(i):dict(messages=0,max_raw_ellipses=0,max_effective_envelopes=0)
                     for i in range(5)}
    marker_samples={}
    display_evidence={name:dict(messages=0,namespaces=[]) for name in ('ellipses','peers')}
    def launch(command,name):
        stream=(out/(name+'.log')).open('w');streams.append(stream)
        process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process);return process
    begin=time.monotonic();failure=None;graph_audit=None;actual_dir=None;completion_wall=None
    try:
        launch(['roscore','-p',str(port)],'master')
        server=xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
        for unused in range(200):
            try:
                if server.getPid('/cbf_test')[0]==1:break
            except (OSError,xmlrpc.client.Error):pass
            time.sleep(.1)
        else:raise RuntimeError('ROS master did not start')
        import rospy
        rospy.set_param('/use_sim_time',True)
        from gazebo_msgs.msg import ModelStates
        from std_msgs.msg import Bool,String
        from visualization_msgs.msg import MarkerArray
        from tb3_distributed_control.msg import LeaderHistoryData
        rospy.init_node('cbf_gazebo_evaluator',anonymous=True,disable_signals=True)
        def lead(msg):
            if not leader_start:leader_start.append(msg.ros_timestamp-msg.trajectory_time)
        def diagnostic(msg,i):
            latest[i]=json.loads(msg.data)
        def peer_markers(msg,i):
            evidence=marker_evidence[str(i)]
            current=[m for m in msg.markers if m.action==0]
            raw=sum(m.ns=='peer_raw' for m in current)
            bodies=sum(m.ns=='peer_body' for m in current)
            evidence['messages']+=1
            if raw>evidence['max_raw_ellipses'] or str(i) not in marker_samples:
                marker_samples[str(i)]=[dict(namespace=m.ns,id=m.id,stamp=m.header.stamp.to_sec(),
                    points=[[p.x,p.y] for p in m.points],text=m.text,
                    position=[m.pose.position.x,m.pose.position.y]) for m in current]
            evidence['max_raw_ellipses']=max(evidence['max_raw_ellipses'],raw)
            evidence['max_effective_envelopes']=max(evidence['max_effective_envelopes'],bodies)
        def display_markers(msg,name):
            item=display_evidence[name];item['messages']+=1
            item['namespaces']=sorted(set(item['namespaces'])|{m.ns for m in msg.markers if m.action==0})
        def models(msg):
            stamp=rospy.get_time()
            if truth and stamp-truth[-1]['ros_time']<.095:return
            if not all('tb3_'+str(i) in msg.name for i in range(5)):return
            row={'ros_time':stamp}
            for i in range(5):
                p=msg.pose[msg.name.index('tb3_'+str(i))].position
                row['x'+str(i)]=p.x;row['y'+str(i)]=p.y
            truth.append(row)
        rospy.Subscriber('/leader/history',LeaderHistoryData,lead,queue_size=100)
        rospy.Subscriber('/gazebo/model_states',ModelStates,models,queue_size=1)
        rospy.Subscriber('/system/shutdown',Bool,lambda m:completed.append(True) if m.data else None)
        for i in range(5):
            rospy.Subscriber('/tb3_'+str(i)+'/cbf_diagnostics',String,diagnostic,callback_args=i,queue_size=5)
            rospy.Subscriber('/tb3_'+str(i)+'/cbf_geometry/peers',MarkerArray,peer_markers,callback_args=i,queue_size=1)
        for name in display_evidence:
            rospy.Subscriber('/cbf/geometry/'+name,MarkerArray,display_markers,callback_args=name,queue_size=1)
        command=['roslaunch','tb3_distributed_control','multi_turtlebot3_consensus_gazebo_cbf.launch',
            'gui:=false','open_rviz:=false','total_time:='+str(args.duration)]
        if not args.default_output:command.append('output_dir:='+str(out))
        command.extend(['geometry_debug:='+str(args.geometry_debug).lower(),
                        'geometry_observer:='+str(args.geometry_observer),
                        'realtime_solver:='+str(not args.legacy_solver).lower(),
                        'nlp_budget_ms:='+str(args.nlp_budget_ms)])
        process=launch(command,'launch')
        last_print=0.
        while time.monotonic()-begin<args.wall_timeout:
            if actual_dir is None and rospy.has_param('/cbf/run_id'):
                key='/cbf/results/'+rospy.get_param('/cbf/run_id')+'/directory'
                if rospy.has_param(key):actual_dir=Path(rospy.get_param(key))
            if leader_start and graph_audit is None:
                _,_,state=server.getSystemState('/cbf_test')
                subscribers=dict(state[1])
                nodes=sorted({node for section in state for _,members in section for node in members})
                scans={name:subscribers.get('/'+name+'/scan',[]) for name in
                       ['tb3_'+str(i) for i in range(5)]}
                graph_audit=dict(scan_subscribers=scans,
                    geometry_display_publishers={name:dict(state[0]).get(name,[]) for name in
                        ('/cbf/geometry/ellipses','/cbf/geometry/peers')},
                    standalone_geometry_nodes=[name for name in nodes if name.endswith('/laser_geometry')])
                graph_audit['passed']=(not graph_audit['standalone_geometry_nodes'] and all(
                    '/tb3_controller_cbf_'+str(i) in scans['tb3_'+str(i)] for i in range(5)))
            if completed:
                if completion_wall is None:completion_wall=time.monotonic()
                if actual_dir is not None and (actual_dir/'RESULTS_SAVED').exists():break
                if time.monotonic()-completion_wall>40:raise RuntimeError('Result export did not finish; inspect recorder logs')
            if process.poll() is not None:raise RuntimeError('roslaunch exited unexpectedly')
            if not leader_start and time.monotonic()-begin>75:
                raise RuntimeError('Startup readiness timeout; inspect launch.log')
            if time.monotonic()-last_print>10:
                print('CBF progress sim=%.1f leader=%s robots=%s' % (rospy.get_time(),
                    'started' if leader_start else 'waiting',
                    {i:(round(d['time'],1),d['status'],round(d['consensus_error_x'],2)) for i,d in latest.items()}),flush=True)
                last_print=time.monotonic()
            time.sleep(.2)
        else:raise RuntimeError('Gazebo test wall timeout')
    except Exception as error:
        failure=str(error)
    finally:
        for process in reversed(processes):stop(process)
        for stream in streams:stream.close()
    if actual_dir is not None and actual_dir!=out:
        for file in actual_dir.iterdir():
            if file.is_file():shutil.copy2(file,out/file.name)
    if truth:
        with (out/'gazebo_truth.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(truth[0]));writer.writeheader();writer.writerows(truth)
    result=dict(duration=args.duration,wall_seconds=time.monotonic()-begin,completed=bool(completed),
        failure=failure,leader_start_ros=leader_start[0] if leader_start else None,
        geometry_graph=graph_audit,peer_marker_evidence=marker_evidence,display_evidence=display_evidence,
        experiment_output_dir=str(actual_dir) if actual_dir is not None else None,
        result_export_passed=(actual_dir is not None and (actual_dir/'RESULTS_SAVED').exists()
            and bool(list(actual_dir.glob('consensus_data_*.pkl')))
            and bool(list(actual_dir.glob('trajectory_*.png')))
            and (actual_dir/'geometry_statistics.json').exists()),robots={})
    (out/'peer_marker_samples.json').write_text(json.dumps(marker_samples,indent=2))
    for i in range(5):
        path=out/('tb3_'+str(i)+'.csv')
        if path.exists():
            rows=list(csv.DictReader(path.open()));statuses={}
            for row in rows:statuses[row['status']]=statuses.get(row['status'],0)+1
            if rows:
                result['robots'][str(i)]=dict(rows=len(rows),statuses=statuses,last_time=float(rows[-1]['time']),
                    mean_position_error=float(np.mean([np.hypot(float(x['consensus_error_x']),float(x['consensus_error_y'])) for x in rows])),
                    final_position_error=float(np.hypot(float(rows[-1]['consensus_error_x']),float(rows[-1]['consensus_error_y']))),
                    laser_peer_constraint_cycles=sum(float(x.get('laser_peer_constraints',0))>0 for x in rows),
                    laser_peer_constraint_count=sum(int(x.get('laser_peer_constraints',0)) for x in rows),
                    odometry_peer_constraint_count=sum(int(x.get('odometry_peer_constraints',0)) for x in rows),
                    solve_ms_mean=float(np.mean([float(x['solve_ms']) for x in rows])))
    if truth and leader_start:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        package=Path(__file__).resolve().parents[2]
        world=ET.parse(package/'worlds/consensus_obstacle_1.world').getroot()
        obstacles=[]
        for link in world.findall('.//model[@name="obstacles"]/link'):
            pose=np.array([float(x) for x in link.findtext('pose').split()]);g=link.find('collision/geometry')
            if g.find('cylinder') is not None:obstacles.append(('circle',pose[:2],float(g.findtext('cylinder/radius')),0.,0.))
            else:
                a,b,_=[float(x) for x in g.findtext('box/size').split()];obstacles.append(('box',pose[:2],a,b,pose[5]))
        samples=[r for r in truth if r['ros_time']>=leader_start[0]]
        times=np.array([r['ros_time']-leader_start[0] for r in samples])
        pos=np.array([[[r['x'+str(i)],r['y'+str(i)]] for r in samples] for i in range(5)])
        minimum=1000.;pair=1000.
        for i in range(5):
            for kind,center,a,b,theta in obstacles:
                delta=pos[i]-center
                if kind=='circle':distance=np.linalg.norm(delta,axis=1)-a
                else:
                    c,s=np.cos(theta),np.sin(theta)
                    local=delta@np.array([[c,-s],[s,c]])
                    q=np.abs(local)-np.array([a/2,b/2]);distance=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)
                minimum=min(minimum,float(distance.min()-.13))
            for j in range(i):pair=min(pair,float(np.linalg.norm(pos[i]-pos[j],axis=1).min()-.26))
        result.update(evaluation_body_radius=.13,physical_obstacle_clearance_min=minimum,
                      pair_body_clearance_min=pair)
        final_leader=np.array([2.5*np.sin(times[-1]/15),.3+2.5*np.sin(times[-1]/30)])
        offsets=.4*np.column_stack((np.cos(np.arange(5)*2*np.pi/5),np.sin(np.arange(5)*2*np.pi/5)))
        result['truth_final_position_error_max']=float(np.linalg.norm(pos[:,-1,:]-final_leader-offsets,axis=1).max())
        fig,ax=plt.subplots(figsize=(8,7))
        from matplotlib.patches import Circle,Rectangle
        from matplotlib.transforms import Affine2D
        for i in range(5):ax.plot(pos[i,:,0],pos[i,:,1],label='Robot '+str(i))
        for kind,center,a,b,theta in obstacles:
            if kind=='circle':patch=Circle(center,a,color='gray',alpha=.5)
            else:
                patch=Rectangle((-a/2,-b/2),a,b,color='gray',alpha=.5)
                patch.set_transform(Affine2D().rotate(theta).translate(*center)+ax.transData)
            ax.add_patch(patch)
        ax.plot(2.5*np.sin(times/15),.3+2.5*np.sin(times/30),'k--',label='Leader')
        ax.set_aspect('equal');ax.grid(alpha=.25);ax.legend();ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
        fig.tight_layout();fig.savefig(out/'gazebo_trajectories.png',dpi=160);plt.close(fig)
    result['peer_envelopes_passed']=(len(result['robots'])==5
        and all(v['laser_peer_constraint_cycles']>0 for v in result['robots'].values())
        and all(v['max_raw_ellipses']>0 and v['max_effective_envelopes']==4
                for v in marker_evidence.values()))
    categories_file=actual_dir/'chart_categories.json' if actual_dir is not None else None
    categories=json.loads(categories_file.read_text()) if categories_file is not None and categories_file.exists() else {}
    result['original_categories_passed']=(len(categories)==12 and all(
        (actual_dir/entry['file']).exists() for entry in categories.values())
        and categories.get('avoidance_forces',{}).get('status')=='not_applicable')
    expected='/tb3_controller_cbf_'+str(args.geometry_observer)
    result['display_passed']=(graph_audit is not None
        and all(nodes==[expected] for nodes in graph_audit['geometry_display_publishers'].values())
        and all(v['messages']>0 for v in display_evidence.values())
        and (args.geometry_debug or (display_evidence['ellipses']['namespaces']==['ellipse']
             and display_evidence['peers']['namespaces']==['peer_body','peer_raw'])))
    result['basic_checks_passed']=(failure is None and bool(completed) and len(result['robots'])==5
        and graph_audit is not None and graph_audit['passed']
        and result['peer_envelopes_passed']
        and result['result_export_passed']
        and result['display_passed']
        and result['original_categories_passed']
        and all(v['last_time']>args.duration-.5 for v in result['robots'].values())
        and result.get('physical_obstacle_clearance_min',-1)>-.005
        and result.get('pair_body_clearance_min',-1)>-.005)
    # Startup smoke tests need not converge in six seconds. Full formation
    # trials must actually finish tracking, not merely produce files/clearance.
    result['tracking_goal_passed']=(args.duration<120. or
        result.get('truth_final_position_error_max',float('inf'))<.05)
    result['acceptance_passed']=result['basic_checks_passed'] and result['tracking_goal_passed']
    (out/'evaluation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
    if failure:raise SystemExit(1)


if __name__=='__main__':main()
