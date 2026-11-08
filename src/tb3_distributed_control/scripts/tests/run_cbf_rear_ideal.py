#!/usr/bin/env python3
"""Ideal discrete-unicycle diagnostic, exact obstacle shape/future supplied.

Offline planning budget is 0.5s per simulated 0.1s tick, NOT a realtime claim.
No ROS/perception. Oracle information is confined to this labeled diagnostic;
the real controller continues to use its own scan/KF input.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import yaml
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
from cbf_mpc_core import default_config
from mpc_dcbf_cached_solver import MPCSolverManager,CachedMPCProblem,rollout


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scenario',choices=['crossing','decelerating_crossing'],required=True)
    p.add_argument('--method',choices=['legacy','plain','rear'],required=True);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    source=PACKAGE/'scripts/cbf_dynamic_scenario.py';tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='trajectory')
    scope={'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope);trajectory=scope['trajectory']
    spec=yaml.safe_load((PACKAGE/'config/cbf_dynamic_scenarios.yaml').read_text())[args.scenario][0]
    manifest={}
    for path in [PACKAGE/'纯仿真代码/mpc_dcbf_cached_solver.py',PACKAGE/'scripts/cbf_mpc_core.py',source,Path(__file__),PACKAGE/'config/cbf_dynamic_scenarios.yaml']:
        shutil.copy2(path,out/path.name);manifest[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    cfg=default_config();cfg.update(realtime=True,nlp_wall_time=.4,manager_wall_time=.48,execution_backup=True,
        rear_pass=args.method=='rear',suppress_guidance=args.method=='plain',log_mpc_inputs=True)
    dt=cfg['dt'];n=cfg['horizon'];duration=40.;steps=round(duration/dt);state=np.array([.4,0.,0.]);previous=np.array([.15,0.]);last=None
    manager=MPCSolverManager(cfg);rows=[];traces=[];minimum=math.inf
    def obstacle(t):
        values=np.array([trajectory(spec,t+j*dt) for j in range(61)]);centers=values[:,:2];velocity=values[0,2:]
        moving=bool(np.linalg.norm(velocity)>.03)
        # Same deterministic execution buffer as the existing CV controller.
        # Perfect shape/forecast removes stochastic inflation, not this margin.
        model_radius=spec['radius']+.02+.35*np.linalg.norm(velocity)
        return dict(name='ideal_actor',center=centers[0].copy(),a=model_radius,b=model_radius,theta=0.,dynamic=moving,guidance=not moving,
            prediction_centers=centers[:n+1].copy(),prediction_axes=np.tile([model_radius]*2,(n+1,1)),prediction_model='cv',
            rear_prediction_centers=centers.copy(),rear_prediction_axes=np.tile([model_radius]*2,(61,1)),
            guidance_motion=dict(stamp=t,weights=[1.],models=[dict(mode='cv',position=centers[0].tolist(),velocity=velocity.tolist(),acceleration=[0.,0.],tau=1.)]))
    problem=CachedMPCProblem(cfg,1,False);zero=np.zeros((n,2));problem.solve(state,np.tile(state,(n+1,1)),[obstacle(0.)],[-.22,.22],None,zero,zero)
    manager.problems[(1,False)]=problem
    for k in range(steps):
        t=k*dt;goals=np.zeros((n+1,3));goals[:,0]=.4+.15*(t+dt*np.arange(n+1));obs=obstacle(t);begin=time.perf_counter()
        try:
            answer=manager.solve(state,goals,[obs],[-.22,.22],last,'tb3_0',deadline=begin+.5,applied_previous=previous,cycle_token=k)
            command=answer['controls'][0].copy();last=answer
            status='ok' if answer['success'] else 'feasible_iterate' if answer['validated_steps']==n else 'validated_one_step'
        except RuntimeError:command=np.zeros(2);status='solver_failed_stop'
        elapsed=1000*(time.perf_counter()-begin);heading=math.atan2(math.sin(-state[2]),math.cos(-state[2]));error=state[:2]-goals[0,:2]
        clearance=float(np.linalg.norm(state[:2]-obs['center'])-.13-spec['radius']);minimum=min(minimum,clearance)
        rows.append(dict(time=t,ros_time=t,evaluation_time=t,cycle_id=k,status=status,x=state[0],y=state[1],theta=state[2],v=command[0],omega=command[1],
            consensus_error_x=error[0],consensus_error_y=error[1],heading_error=heading,leader_reference_heading=0.,clearance=clearance,solve_ms=elapsed))
        traces.append(dict(cycle_id=k,trajectory_time=t,status=status,command=command.tolist(),manager=manager.last_records.get('tb3_0',{})))
        state=rollout(state,command[None,:],dt)[1];previous=command
    final=float(np.linalg.norm(state[:2]-np.array([.4+.15*duration,0.])))
    final_actor=np.array(trajectory(spec,duration)[:2]);minimum=min(minimum,float(np.linalg.norm(state[:2]-final_actor)-.13-spec['radius']))
    with (out/'tb3_0.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'tb3_0_solver_trace.jsonl').open('w') as f:
        for trace in traces:f.write(json.dumps(trace)+'\n')
    (out/'run_metadata.json').write_text(json.dumps(dict(simulator='ideal_discrete_unicycle',forecast='oracle',realtime=False,offline_budget=.5,
        total_time=duration,control_dt=dt,robot_names=['tb3_0'],method=args.method,scenario=args.scenario,configuration=cfg,
        safety_model='h=d-(actor_radius+0.02+0.35*current_actor_speed)-0.12',evaluation_body_radius=.13,source_sha256=manifest),indent=2))
    result=dict(method=args.method,scenario=args.scenario,forecast='oracle',proxy_min_clearance=minimum,final_position_error=final,
        passed=minimum>0 and final<.25,position_rmse=float(np.sqrt(np.mean([r['consensus_error_x']**2+r['consensus_error_y']**2 for r in rows]))),
        command_low_speed_seconds=dt*sum(abs(r['v'])<.02 for r in rows),omega_total_variation=float(np.abs(np.diff([r['omega'] for r in rows])).sum()))
    (out/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
