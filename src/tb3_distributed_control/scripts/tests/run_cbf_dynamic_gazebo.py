#!/usr/bin/env python3
"""Isolated real Gazebo test; ground truth is evaluated, never fed to control."""
import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time
import xmlrpc.client
import xml.etree.ElementTree as ET
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_cbf_active_motion import evaluate_run as evaluate_active_run
from evaluate_cbf_team_motion import evaluate_team_run
from run_cbf_gazebo import free_port,stop


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-prediction-markers',action='store_true')
    parser.add_argument('--log-cluster-points',action='store_true',help='Local single-robot diagnostics only')
    parser.add_argument('--motion-geometry-observer',choices=['mvee','circle'],default='mvee')
    parser.add_argument('--circle-prior-retention',choices=['true','false'],default='false')
    parser.add_argument('--open-rviz',action='store_true')
    parser.add_argument('--capture-rviz',action='store_true',help='Capture only this test RViz window via its PID')
    parser.add_argument('--scenario',default='crossing',choices=['crossing','head_on','overtaking','maneuver','mixed','accelerating_crossing','decelerating_crossing','stop_go_crossing','active_head_on','active_overtaking','active_oblique_crossing'])
    parser.add_argument('--robots',type=int,default=1,choices=[1,2,5])
    parser.add_argument('--gazebo-seed',type=int,default=None)
    parser.add_argument('--duration',type=float,default=32.)
    parser.add_argument('--spatial-pass',choices=['true','false'],default='false')
    parser.add_argument('--spatial-lateral-extras',default='[0.12,0.22,0.32]')
    parser.add_argument('--spatial-shift-durations',default='[4.5,6.0,7.5]')
    parser.add_argument('--spatial-predictive-return',choices=['true','false'],default='false')
    parser.add_argument('--spatial-activation-threshold',type=float,default=-1.)
    parser.add_argument('--spatial-activation-preview-seconds',type=float,default=1.)
    parser.add_argument('--spatial-compiled-rollout',choices=['true','false'],default='false')
    parser.add_argument('--spatial-timeout-carry',choices=['true','false'],default='false')
    parser.add_argument('--spatial-runtime-guard',choices=['true','false'],default='false')
    parser.add_argument('--spatial-prefix-retention',choices=['true','false'],default='false')
    parser.add_argument('--local-reference-mode',choices=['timed','nearest_line'],default='timed')
    parser.add_argument('--prediction-mode',choices=['cv','frozen','ca','singer','adaptive'],default='cv')
    parser.add_argument('--progress-hypotheses',choices=['true','false'],default='true')
    parser.add_argument('--adaptive-recovery',choices=['true','false'],default='false')
    parser.add_argument('--transport-geometry',choices=['true','false'],default='false')
    parser.add_argument('--predictive-bypass',choices=['true','false'],default='false')
    parser.add_argument('--active-guidance-only',choices=['true','false'],default='false')
    parser.add_argument('--prediction-risk',choices=['true','false'],default='false')
    parser.add_argument('--suppress-guidance',choices=['true','false'],default='false')
    parser.add_argument('--risk-weight',type=float,default=3.)
    parser.add_argument('--risk-clearance',type=float,default=.35)
    parser.add_argument('--risk-discount',type=float,default=.5)
    parser.add_argument('--risk-tail-seconds',type=float,default=0.)
    parser.add_argument('--constraint-mode',choices=['dcbf','ellipse'],default='dcbf')
    parser.add_argument('--risk-penalty',choices=['clearance_quartic','wang_inverse'],default='clearance_quartic')
    parser.add_argument('--wang-alpha',type=float,default=1.)
    parser.add_argument('--wang-mu',type=float,default=.1)
    parser.add_argument('--wang-iota',type=float,default=.02)
    parser.add_argument('--wang-radius',type=float,default=1.)
    parser.add_argument('--mpc-horizon',type=int,default=10)
    parser.add_argument('--pending-seed-with-backup',choices=['true','false'],default='false')
    parser.add_argument('--log-mpc-inputs',choices=['true','false'],default='false')
    parser.add_argument('--rear-pass',choices=['true','false'],default='false')
    parser.add_argument('--rear-preview-seconds',type=float,default=6.)
    parser.add_argument('--infeasible-escape',choices=['true','false'],default='false')
    parser.add_argument('--execution-backup',choices=['true','false'],default='false')
    parser.add_argument('--progress-abstain',choices=['true','false'],default='true')
    parser.add_argument('--progress-guidance',choices=['true','false'],default='false')
    parser.add_argument('--moving-frame-projection',choices=['true','false'],default='false')
    parser.add_argument('--singer-tau',type=float,default=1.)
    parser.add_argument('--acceleration-buffer',choices=['true','false'],default='true')
    parser.add_argument('--stage-velocity-guidance',choices=['true','false'],default='true')
    parser.add_argument('--output',required=True)
    parser.add_argument('--wall-timeout',type=float,default=240.)
    args=parser.parse_args();out=Path(args.output).resolve()
    active=args.scenario.startswith('active_')
    if args.log_cluster_points and args.robots!=1:parser.error('Point recording is single-robot diagnostic only')
    try:
        extras=json.loads(args.spatial_lateral_extras)
        if np.shape(extras)!=(3,) or not np.isfinite(extras).all() or min(extras)<0 or np.any(np.diff(extras)<0):raise ValueError()
    except (ValueError,TypeError):parser.error('Expected three finite nonnegative ordered lateral extras')
    if args.local_reference_mode=='nearest_line' and args.robots!=1:parser.error('Nearest reference is single-robot only')
    if active and args.robots not in (1,2):parser.error('Active-motion protocol requires single or two-robot experiment')
    if args.robots==2 and not active:parser.error('Two-robot entry currently requires a qualified active scene')
    if active and args.duration!=40.:parser.error('Frozen active-motion protocol requires duration=40')
    if any(out.glob('tb3_*.csv')) or (out/'evaluation.json').exists():
        raise FileExistsError('Existing experiment: choose a new output directory')
    try:
        shift_durations=json.loads(args.spatial_shift_durations)
        if (np.shape(shift_durations)!=(3,) or not np.isfinite(shift_durations).all()
                or min(shift_durations)<=0 or np.any(np.diff(shift_durations)<0)):raise ValueError()
    except (ValueError,TypeError):parser.error('Expected three finite positive ordered shift durations')
    out.mkdir(parents=True,exist_ok=True)
    package=Path(__file__).resolve().parents[2]
    sources=out/'sources';sources.mkdir(exist_ok=True)
    manifest={}
    for relative in ['scripts/tb3_controller_node_cbf.py','scripts/cbf_mpc_core.py','scripts/leader_node_cbf.py','scripts/leader_node.py',
            'scripts/cbf_dynamic_scenario.py','scripts/cbf_spawn_checked.py','scripts/cbf_experiment_results.py','config/cbf_dynamic_scenarios.yaml',
            'scripts/cbf_active_monitor.py','scripts/evaluate_cbf_team_motion.py','config/cbf_two_robot_protocol.yaml','launch/two_turtlebot3_cbf_dynamic.launch','launch/single_turtlebot3_cbf_compact.launch','scripts/evaluate_cbf_active_motion.py','config/cbf_active_avoidance_scenarios.yaml','config/cbf_active_avoidance_protocol.yaml',
            '纯仿真代码/mpc_dcbf_cached_solver.py','launch/multi_turtlebot3_consensus_gazebo_cbf.launch',
            'launch/single_turtlebot3_cbf_dynamic.launch','launch/multi_turtlebot3_consensus_gazebo_cbf_dynamic.launch',
            'launch/single_turtlebot3_cbf_trigger022.launch','launch/single_turtlebot3_cbf_motion_geometry.launch','rviz/tb3_cbf.rviz','rviz/tb3_cbf_single.rviz','scripts/tests/run_cbf_dynamic_gazebo.py']:
        p=package/relative;shutil.copy2(p,sources/p.name);manifest[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
    (sources/'sha256.json').write_text(json.dumps(manifest,indent=2))
    env=os.environ.copy();env.pop('ROS_HOSTNAME',None);env.pop('ROS_NAMESPACE',None)
    env.update(ROS_MASTER_URI='http://127.0.0.1:'+str(free_port()),ROS_IP='127.0.0.1',
        GAZEBO_MASTER_URI='http://127.0.0.1:'+str(free_port()),ROS_HOME=str(out/'ros'),ROS_LOG_DIR=str(out/'roslog'),
        GAZEBO_MODEL_DATABASE_URI='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg')
    os.environ.update(env);processes=[];streams=[];truth=[];begins=[];completed=[];last={};audit={};seed_info={}
    marker_messages=[];capture_error=None;captured=False
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
        from visualization_msgs.msg import MarkerArray
        from tb3_distributed_control.msg import LeaderHistoryData
        rospy.init_node('cbf_dynamic_evaluator',anonymous=True,disable_signals=True)
        rospy.set_param('/cbf/dynamic/log_cluster_points',args.log_cluster_points)
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
        def prediction_markers(msg,observer):
            marker_messages.append(dict(observer=observer,received=rospy.get_time(),markers=[dict(
                stamp=m.header.stamp.to_sec(),frame=m.header.frame_id,ns=m.ns,id=m.id,type=m.type,action=m.action,
                position=[m.pose.position.x,m.pose.position.y,m.pose.position.z],
                orientation=[m.pose.orientation.x,m.pose.orientation.y,m.pose.orientation.z,m.pose.orientation.w],
                scale=[m.scale.x,m.scale.y,m.scale.z],color=[m.color.r,m.color.g,m.color.b,m.color.a],
                lifetime=m.lifetime.to_sec(),text=m.text,points=[[p.x,p.y,p.z] for p in m.points]) for m in msg.markers]))
        subscribers=[rospy.Subscriber('/leader/history',LeaderHistoryData,lead,queue_size=1),
            rospy.Subscriber('/gazebo/model_states',ModelStates,models,queue_size=1),
            rospy.Subscriber('/system/shutdown',Bool,lambda m:completed.append(time.monotonic()) if m.data else None)]
        for i in range(args.robots):subscribers.append(rospy.Subscriber('/tb3_%d/cbf_diagnostics'%i,String,diagnostic,callback_args=i,queue_size=1))
        if args.check_prediction_markers:
            for i in range(args.robots):subscribers.append(rospy.Subscriber('/tb3_%d/cbf_geometry/prediction_regions'%i,MarkerArray,prediction_markers,callback_args=str(i),queue_size=1))
            subscribers.append(rospy.Subscriber('/cbf/geometry/prediction_regions',MarkerArray,prediction_markers,callback_args='selected',queue_size=1))
        name=('single_turtlebot3_cbf_dynamic.launch' if args.robots==1 else 'two_turtlebot3_cbf_dynamic.launch'
              if args.robots==2 else 'multi_turtlebot3_consensus_gazebo_cbf_dynamic.launch')
        process=launch(['roslaunch','tb3_distributed_control',name,'gui:=false','open_rviz:='+str(args.open_rviz or args.capture_rviz).lower(),
            'scenario:='+args.scenario,'total_time:='+str(args.duration),'output_dir:='+str(out),'active_experiment:='+str(active).lower(),'local_reference_mode:='+args.local_reference_mode,'spatial_pass:='+args.spatial_pass,'spatial_lateral_extras:='+args.spatial_lateral_extras,'spatial_shift_durations:='+args.spatial_shift_durations,'spatial_predictive_return:='+args.spatial_predictive_return,'spatial_activation_threshold:='+str(args.spatial_activation_threshold),'spatial_activation_preview_seconds:='+str(args.spatial_activation_preview_seconds),'spatial_compiled_rollout:='+args.spatial_compiled_rollout,'spatial_timeout_carry:='+args.spatial_timeout_carry,'spatial_runtime_guard:='+args.spatial_runtime_guard,'spatial_prefix_retention:='+args.spatial_prefix_retention,
            'gazebo_seed:='+('' if args.gazebo_seed is None else str(args.gazebo_seed)),
            'prediction_mode:='+args.prediction_mode,'motion_geometry_observer:='+args.motion_geometry_observer,'circle_prior_retention:='+args.circle_prior_retention,'progress_guidance:='+args.progress_guidance,'progress_abstain:='+args.progress_abstain,'execution_backup:='+args.execution_backup,'infeasible_escape:='+args.infeasible_escape,'predictive_bypass:='+args.predictive_bypass,'active_guidance_only:='+args.active_guidance_only,'prediction_risk:='+args.prediction_risk,'suppress_guidance:='+args.suppress_guidance,'risk_weight:='+str(args.risk_weight),'risk_clearance:='+str(args.risk_clearance),'risk_discount:='+str(args.risk_discount),'risk_tail_seconds:='+str(args.risk_tail_seconds),'constraint_mode:='+args.constraint_mode,'risk_penalty:='+args.risk_penalty,'wang_alpha:='+str(args.wang_alpha),'wang_mu:='+str(args.wang_mu),'wang_iota:='+str(args.wang_iota),'wang_radius:='+str(args.wang_radius),'mpc_horizon:='+str(args.mpc_horizon),'pending_seed_with_backup:='+args.pending_seed_with_backup,'log_mpc_inputs:='+args.log_mpc_inputs,'rear_pass:='+args.rear_pass,'rear_preview_seconds:='+str(args.rear_preview_seconds),'transport_geometry:='+args.transport_geometry,'adaptive_recovery:='+args.adaptive_recovery,'progress_hypotheses:='+args.progress_hypotheses,'moving_frame_projection:='+args.moving_frame_projection,'singer_tau:='+str(args.singer_tau),
            'acceleration_buffer:='+args.acceleration_buffer,'stage_velocity_guidance:='+args.stage_velocity_guidance],'launch')
        report=0.
        while time.monotonic()-wall<args.wall_timeout:
            if args.capture_rviz and not captured and begins and rospy.get_time()-begins[0]>8.:
                try:
                    uri=server.lookupNode('/cbf_dynamic_test','/rviz')[2]
                    pid=xmlrpc.client.ServerProxy(uri).getPid('/cbf_dynamic_test')[2]
                    tree=subprocess.check_output(['xwininfo','-root','-tree'],text=True,timeout=2)
                    candidates=[re.search(r'(0x[0-9a-f]+)',line).group(1) for line in tree.splitlines()
                                if 'rviz' in line.lower() and re.search(r'0x[0-9a-f]+',line)]
                    window=None
                    for candidate in candidates:
                        prop=subprocess.check_output(['xprop','-id',candidate,'_NET_WM_PID'],text=True,timeout=2)
                        if re.search(r'=\s*'+str(pid)+r'\s*$',prop):window=candidate;break
                    if window is None:raise RuntimeError('This test RViz window was not found')
                    subprocess.run(['xwd','-silent','-id',window,'-out',str(out/'rviz_prediction.xwd')],check=True,timeout=3)
                    captured=True
                except Exception as exc:capture_error=str(exc);captured=True
            if begins and not audit:
                _,_,graph=server.getSystemState('/cbf_dynamic_test')
                audit={topic:members for topic,members in graph[1] if any('/tb3_controller_cbf_' in n for n in members)}
                try:
                    uri=server.lookupNode('/cbf_dynamic_test','/gazebo')[2]
                    pid=xmlrpc.client.ServerProxy(uri).getPid('/cbf_dynamic_test')[2]
                    seed_info['gazebo_command']=[s.decode('utf-8',errors='replace') for s in Path('/proc/%d/cmdline'%pid).read_bytes().split(b'\0') if s]
                except Exception as exc:seed_info['seed_check_error']=str(exc)
            if completed and (out/'RESULTS_SAVED').exists():break
            if completed and time.monotonic()-completed[0]>45:raise RuntimeError('Result export timeout')
            if process.poll() is not None:raise RuntimeError('roslaunch exited')
            if 'Segmentation fault' in (out/'launch.log').read_text(errors='replace'):
                raise RuntimeError('Native simulation process crashed (segmentation fault); see preserved launch log')
            if not begins and time.monotonic()-wall>75:raise RuntimeError('Sensor/MPC/scenario readiness timeout')
            if time.monotonic()-report>10:
                print('Dynamic Gazebo %s: %s'%(args.scenario,{i:(round(d['time'],1),d['status']) for i,d in last.items()}),flush=True);report=time.monotonic()
            time.sleep(.2)
        else:raise RuntimeError('Experiment wall timeout')
    except Exception as exc:failure=str(exc)
    finally:
        for p in reversed(processes):stop(p)
        for stream in streams:stream.close()
    result=dict(circle_prior_retention=args.circle_prior_retention=='true',geometry_observer=args.motion_geometry_observer,spatial_predictive_return=args.spatial_predictive_return=='true',spatial_shift_durations=shift_durations,spatial_activation_threshold=args.spatial_activation_threshold,spatial_activation_preview_seconds=args.spatial_activation_preview_seconds,spatial_lateral_extras=extras,spatial_runtime_guard=args.spatial_runtime_guard=='true',spatial_prefix_retention=args.spatial_prefix_retention=='true' or args.robots==2 and args.spatial_pass=='true',spatial_compiled_rollout=args.spatial_compiled_rollout=='true',spatial_timeout_carry=args.spatial_timeout_carry=='true',spatial_pass=args.spatial_pass=='true',local_reference_mode=args.local_reference_mode,rear_pass=args.rear_pass=='true',rear_preview_seconds=args.rear_preview_seconds,mpc_horizon=args.mpc_horizon,pending_seed_with_backup=args.pending_seed_with_backup=='true',log_mpc_inputs=args.log_mpc_inputs=='true',constraint_mode=args.constraint_mode,risk_penalty=args.risk_penalty,wang_alpha=args.wang_alpha,wang_mu=args.wang_mu,wang_iota=args.wang_iota,wang_radius=args.wang_radius,prediction_risk=args.prediction_risk=='true',suppress_guidance=args.suppress_guidance=='true',risk_weight=args.risk_weight,risk_clearance=args.risk_clearance,risk_discount=args.risk_discount,risk_tail_seconds=args.risk_tail_seconds,active_guidance_only=args.active_guidance_only=='true',adaptive_recovery=args.adaptive_recovery=='true',transport_geometry=args.transport_geometry=='true',predictive_bypass=args.predictive_bypass=='true',infeasible_escape=args.infeasible_escape=='true',execution_backup=args.execution_backup=='true',scenario=args.scenario,robots=args.robots,gazebo_seed=args.gazebo_seed,progress_guidance=args.progress_guidance=='true',progress_abstain=args.progress_abstain=='true',progress_hypotheses=args.progress_hypotheses=='true',prediction_mode=args.prediction_mode,failure=failure,
        completed=bool(completed),controller_subscriptions=audit,robot_statistics={})
    if args.check_prediction_markers:
        with (out/'prediction_regions.jsonl').open('w') as f:
            for row in marker_messages:f.write(json.dumps(row)+'\n')
        hard=sum(m['type']==3 and m['action']==0 and m['ns'].startswith('prediction_hard_') for r in marker_messages for m in r['markers'])
        soft=sum(m['type']==5 and m['action']==0 and m['ns'].startswith('prediction_soft_') for r in marker_messages for m in r['markers'])
        observers={r['observer'] for r in marker_messages if r['markers']}
        result['prediction_markers']=dict(messages=len(marker_messages),hard_cylinders=hard,soft_dashed_ellipses=soft,
            observers=sorted(observers),passed=hard>0 and {'selected',*map(str,range(args.robots))}<=observers
            and (args.spatial_activation_threshold<0 or args.spatial_activation_preview_seconds<=args.mpc_horizon*.1 or soft>0))
    if args.capture_rviz:
        if (out/'rviz_prediction.xwd').exists():
            subprocess.run(['convert',str(out/'rviz_prediction.xwd'),str(out/'rviz_prediction.png')],check=True)
        result['rviz_capture']=dict(passed=(out/'rviz_prediction.png').exists(),error=capture_error)
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
        exported=out/'experiment_statistics.json'
        window_final=(json.loads(exported.read_text()).get(str(i),{}).get('position_error_summary',{}).get('final',float('nan'))
                      if exported.exists() else float('nan'))
        result['robot_statistics'][str(i)]=dict(window_final_error=window_final,rows=len(rows),last_time=float(rows[-1]['time']),
            statuses=statuses,timing_p95_ms=float(np.percentile(times,95)),timing_max_ms=float(times.max()),
            final_error=float(np.hypot(float(rows[-1]['consensus_error_x']),float(rows[-1]['consensus_error_y']))),
            dynamic_cycles=sum(float(r['dynamic_tracks'])>0 for r in rows),
            max_estimated_speed=max(float(r['max_obstacle_speed']) for r in rows))
    result.update(seed_info)
    argv=seed_info.get('gazebo_command',[])
    metadata_path=out/'run_metadata.json'
    seed_metadata=json.loads(metadata_path.read_text()).get('gazebo_seed','') if metadata_path.exists() else ''
    result['seed_configured']=(args.gazebo_seed is None or (str(seed_metadata)==str(args.gazebo_seed)
        and '--seed' in argv and argv.index('--seed')+1<len(argv) and argv[argv.index('--seed')+1]==str(args.gazebo_seed)))
    result['progress_configured']=(bool(json.loads(metadata_path.read_text()).get('progress_guidance',False))==result['progress_guidance']) if metadata_path.exists() else False
    if metadata_path.exists():result['progress_configured'] &= bool(json.loads(metadata_path.read_text()).get('progress_hypotheses',True))==result['progress_hypotheses']
    if metadata_path.exists():result['progress_configured'] &= bool(json.loads(metadata_path.read_text()).get('progress_abstain',True))==result['progress_abstain']
    result['execution_backup_configured']=(bool(json.loads(metadata_path.read_text()).get('execution_backup',False))==result['execution_backup']) if metadata_path.exists() else False
    result['infeasible_escape_configured']=(bool(json.loads(metadata_path.read_text()).get('infeasible_escape',False))==result['infeasible_escape']) if metadata_path.exists() else False
    result['predictive_bypass_configured']=(bool(json.loads(metadata_path.read_text()).get('predictive_bypass',False))==result['predictive_bypass']) if metadata_path.exists() else False
    result['active_guidance_only_configured']=(bool(json.loads(metadata_path.read_text()).get('active_guidance_only',False))==result['active_guidance_only']) if metadata_path.exists() else False
    meta=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    result['risk_configured']=(meta.get('prediction_risk')==result['prediction_risk'] and meta.get('suppress_guidance')==result['suppress_guidance']
        and all(meta.get('risk_parameters',{}).get(k)==value for k,value in [('weight',args.risk_weight),('clearance',args.risk_clearance),('discount',args.risk_discount),('tail_seconds',args.risk_tail_seconds)]))
    result['core_mode_configured']=(meta.get('constraint_mode')==args.constraint_mode and meta.get('risk_parameters',{}).get('penalty')==args.risk_penalty
        and all(meta.get('risk_parameters',{}).get(k)==v for k,v in [('wang_alpha',args.wang_alpha),('wang_mu_per_node',args.wang_mu),('wang_iota',args.wang_iota),('wang_radius',args.wang_radius)]))
    result['rear_configured']=(meta.get('rear_pass')==result['rear_pass'] and meta.get('rear_preview_seconds')==args.rear_preview_seconds)
    result['horizon_configured']=(meta.get('mpc_horizon')==args.mpc_horizon)
    result['local_reference_configured']=(meta.get('local_reference_mode','timed')==args.local_reference_mode)
    result['spatial_pass_configured']=(meta.get('spatial_pass',False)==(args.spatial_pass=='true'))
    result['circle_prior_configured']=(meta.get('circle_prior_retention',False)==result['circle_prior_retention'])
    result['geometry_observer_configured']=(meta.get('geometry_observer','mvee')==args.motion_geometry_observer)
    result['return_configured']=(meta.get('spatial_predictive_return',False)==result['spatial_predictive_return'])
    result['shift_durations_configured']=(meta.get('spatial_shift_durations',[4.5,6.,7.5])==shift_durations)
    result['lateral_extras_configured']=(meta.get('spatial_lateral_extras',[.12,.22,.32])==extras)
    result['activation_configured']=(meta.get('spatial_activation_threshold',-1.)==args.spatial_activation_threshold and meta.get('spatial_activation_preview_seconds',1.)==args.spatial_activation_preview_seconds)
    result['spatial_backend_configured']=(meta.get('spatial_compiled_rollout',False)==result['spatial_compiled_rollout'] and meta.get('spatial_timeout_carry',False)==result['spatial_timeout_carry'] and meta.get('spatial_prefix_retention',False)==result['spatial_prefix_retention'] and meta.get('spatial_runtime_guard',False)==result['spatial_runtime_guard'])
    result['seed_policy_configured']=(meta.get('pending_seed_with_backup')==result['pending_seed_with_backup'] and meta.get('log_mpc_inputs')==result['log_mpc_inputs'])
    result['transport_geometry_configured']=(bool(json.loads(metadata_path.read_text()).get('transport_geometry',False))==result['transport_geometry']) if metadata_path.exists() else False
    result['adaptive_recovery_configured']=(bool(json.loads(metadata_path.read_text()).get('adaptive_recovery',False))==result['adaptive_recovery']) if metadata_path.exists() else False
    result['no_truth_input_passed']=bool(audit) and not any(topic in audit for topic in ('/gazebo/model_states','/cbf/scenario/truth','/gazebo/set_model_state'))
    result['own_scan_only']=all(sorted(topic for topic,members in audit.items()
        if '/tb3_controller_cbf_'+str(i) in members and topic.endswith('/scan'))==['/tb3_'+str(i)+'/scan'] for i in range(args.robots))
    result['passed']=(failure is None and bool(completed) and (out/'RESULTS_SAVED').exists()
        and (not args.check_prediction_markers or result['prediction_markers']['passed'])
        and (not args.capture_rviz or result['rviz_capture']['passed'])
        and len(result['robot_statistics'])==args.robots and result.get('dynamic_clearance_min',-1)>0.
        and result.get('peer_clearance_min',-1)>0. and result.get('static_clearance_min',-1)>0.
        and result['circle_prior_configured'] and result['geometry_observer_configured'] and result['return_configured'] and result['shift_durations_configured'] and result['activation_configured'] and result['lateral_extras_configured'] and result['spatial_backend_configured'] and result['own_scan_only'] and result['spatial_pass_configured'] and result['local_reference_configured'] and result['no_truth_input_passed'] and result['seed_configured'] and result['progress_configured'] and result['execution_backup_configured'] and result['infeasible_escape_configured'] and result['predictive_bypass_configured'] and result['active_guidance_only_configured'] and result['risk_configured'] and result['core_mode_configured'] and result['seed_policy_configured'] and result['horizon_configured'] and result['rear_configured'] and result['transport_geometry_configured'] and result['adaptive_recovery_configured']
        and all(s['last_time']>args.duration-.5 and s['dynamic_cycles']>0 and np.isfinite(s['window_final_error']) and s['window_final_error']<.25
                for s in result['robot_statistics'].values()))
    if active:
        result['base_passed']=result['passed']
        try:continuous=(evaluate_team_run(out,base_passed=result['base_passed']) if args.robots==2
                        else evaluate_active_run(out,base_passed=result['base_passed']))
        except Exception as exc:continuous=dict(passed=False,failed_checks=['active_evaluation_error'],error=str(exc))
        (out/'active_evaluation.json').write_text(json.dumps(continuous,indent=2,allow_nan=False))
        result.update(acceptance_protocol='two_robot_motion_v1' if args.robots==2 else 'active_motion_v1',active_motion=continuous,passed=continuous['passed'])
    (out/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
