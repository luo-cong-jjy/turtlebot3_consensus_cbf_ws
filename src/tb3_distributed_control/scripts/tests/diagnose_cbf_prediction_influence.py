#!/usr/bin/env python3
"""Paired offline constraint-only diagnostic on identical recorded inputs.

Freezes only external obstacle future centers. Current centers, uncertainty
axes, peer forecasts, goals, bounds and the initial guess stay identical.
Soft guidance is disabled in BOTH solves. This is not a closed-loop ablation
or a global-optimality/safety proof; solves have no realtime deadline.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
from cbf_mpc_core import default_config
from mpc_dcbf_cached_solver import CachedMPCProblem,barrier_values


def diagnose(path,start,end):
    results=[]
    for line_number,line in enumerate(path.open(),1):
        record=json.loads(line);t=record['trajectory_time']
        d=record.get('manager',{}).get('progress_decision')
        if not d or not start<=t<=end:continue
        cfg=dict(default_config(),**d['config']);cfg.update(realtime=False)
        state=np.array(d['state']);goals=np.array(d['goals']);previous=np.array(d['previous'])
        obstacles=copy.deepcopy(d['obstacles']);frozen=copy.deepcopy(obstacles)
        n=cfg['horizon'];guess=np.tile(previous,(n,1))
        plan_guess=np.tile(previous[0]*np.array([np.cos(state[2]),np.sin(state[2])]),(n,1))
        displacements={}
        for original,hold in zip(obstacles,frozen):
            if original.get('peer',False):continue
            centers=np.asarray(original['prediction_centers'])
            displacements[original['name']]=float(np.linalg.norm(centers[-1]-centers[0]))
            hold['prediction_centers']=np.tile(centers[0],(n+1,1))
        item=dict(time=t,line=line_number,decision_id=d['decision_id'],external_window_displacements=displacements)
        answers={}
        problem=CachedMPCProblem(cfg,len(obstacles),False)
        for label,obs in [('moving',obstacles),('frozen_external_centers',frozen)]:
            try:
                answer=problem.solve(state,goals,obs,d['bounds'],previous,guess,plan_guess)
                answers[label]=answer
                h=barrier_values(answer['states'],obstacles,cfg['margin'])
                gammas=np.array([o.get('gamma',cfg['gamma']) for o in obstacles])[:,None]
                slack=h[:,1:]-(1-gammas)*h[:,:-1]
                item[label]=dict(solved=True,first_control=answer['controls'][0].tolist(),cost=float(answer['cost']),
                    moving_forecast_min_slack=float(slack.min()) if slack.size else None,
                    moving_forecast_min_h=float(h.min()) if h.size else None)
            except RuntimeError:
                item[label]=dict(solved=False,status=problem.last_attempt.get('status'))
        if len(answers)==2:
            item['first_control_difference']=(answers['moving']['controls'][0]-answers['frozen_external_centers']['controls'][0]).tolist()
        results.append(item)
    return dict(trace=str(path.resolve()),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        solver_sha256=hashlib.sha256((PACKAGE/'纯仿真代码/mpc_dcbf_cached_solver.py').read_bytes()).hexdigest(),
        interval=[start,end],interpretation=__doc__,frames=results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('--start',type=float,default=6.)
    p.add_argument('--end',type=float,default=12.);p.add_argument('--output',required=True);args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(diagnose(Path(args.trace),args.start,args.end),indent=2,allow_nan=False))
