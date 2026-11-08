#!/usr/bin/env python3
"""Matched-input 2x2 comparison of Wang eq14 and geometric vs DCBF safety.

Uses every recorded decision within the declared interval, three common
initial guesses, no runtime deadline. This is an offline local-solver
diagnostic, NOT a full paper reproduction, closed-loop result or global proof.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import casadi as ca
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
from cbf_mpc_core import default_config
from mpc_dcbf_cached_solver import CachedMPCProblem,barrier_values,safety_residuals


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('trace')
    parser.add_argument('--start',type=float,default=6.);parser.add_argument('--end',type=float,default=16.)
    parser.add_argument('--beta',type=float,default=.3);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    trace=Path(args.trace);shutil.copy2(trace,out/'input_trace.jsonl')
    manifest={}
    for source in [PACKAGE/'纯仿真代码/mpc_dcbf_cached_solver.py',PACKAGE/'scripts/cbf_mpc_core.py',Path(__file__)]:
        shutil.copy2(source,out/source.name);manifest[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
    results=[];problems={}
    for line_number,line in enumerate(trace.open(),1):
        record=json.loads(line);manager=record.get('manager',{})
        d=manager.get('input_snapshot') or manager.get('progress_decision');t=record['trajectory_time']
        if not d or not args.start<=t<=args.end:continue
        state=np.asarray(d['state']);goals=np.asarray(d['goals']);previous=np.asarray(d['previous'])
        obstacles=copy.deepcopy(d['obstacles']);bounds=d['bounds']
        cfg=dict(default_config(),**d['config']);cfg.update(realtime=False,execution_backup=True,suppress_guidance=True,
            risk_penalty='wang_inverse',risk_weight=args.beta,risk_tail_seconds=0.)
        n=cfg['horizon'];initial_h=barrier_values(state[None,:],obstacles,cfg['margin'])[:,0]
        entry=dict(time=t,line=line_number,input=d,initial_h_min=float(initial_h.min()),safe_initial=bool(initial_h.min()>=-2e-6),profiles={})
        q=np.clip(np.diff(goals[:,:2],axis=0)/cfg['dt'],-cfg['plan_v_max'],cfg['plan_v_max'])
        tracking=np.tile(previous,(n,1));tracking[:,0]=np.clip(tracking[:,0],*bounds)
        tracking[:,1]=np.clip(tracking[:,1],-cfg['omega_max'],cfg['omega_max'])
        seeds={'previous':tracking}
        for label,sign in [('left',1),('right',-1)]:
            seed=np.tile([np.clip(.12,*bounds),0.],(n,1));seed[:min(n,5),1]=sign*min(.8,cfg['omega_max']);seeds[label]=seed
        for mode in ('dcbf','ellipse'):
            for risk in (False,True):
                profile=mode+('_inverse' if risk else '_hard')
                configured=dict(cfg,constraint_mode=mode,prediction_risk=risk)
                key=json.dumps(configured,sort_keys=True)+str(len(obstacles))
                if key not in problems:problems[key]=CachedMPCProblem(configured,len(obstacles),bool(cfg['delta_limit']))
                problem=problems[key];attempts=[];best=None;begin=time.perf_counter()
                for name,guess in seeds.items():
                    try:
                        answer=problem.solve(state,goals,obstacles,bounds,previous,guess,q)
                        h=barrier_values(answer['states'],obstacles,cfg['margin'])
                        result=dict(seed=name,valid=True,cost=answer['cost'],risk_cost=answer.get('risk_cost',0.),
                            first_control=answer['controls'][0].tolist(),states=answer['states'].tolist(),controls=answer['controls'].tolist(),
                            min_h=float(h.min()),original_dcbf_min_residual=float(safety_residuals(h,obstacles,dict(cfg,constraint_mode='dcbf')).min()))
                        if best is None or result['cost']<best['cost']:best=result
                    except RuntimeError:result=dict(seed=name,valid=False,status=problem.last_attempt.get('status'))
                    attempts.append(result)
                entry['profiles'][profile]=dict(best=best,attempts=attempts,wall_ms=1000*(time.perf_counter()-begin))
        results.append(entry)
    names=['dcbf_hard','dcbf_inverse','ellipse_hard','ellipse_inverse'];summary={}
    for name in names:
        summary[name]=dict(valid_frames=sum(r['profiles'][name]['best'] is not None for r in results),
            valid_safe_start_frames=sum(r['safe_initial'] and r['profiles'][name]['best'] is not None for r in results),
            geometric_rate_violation_frames=sum(r['profiles'][name]['best'] is not None and r['profiles'][name]['best']['original_dcbf_min_residual']<-2e-6 for r in results))
    result=dict(interpretation=__doc__,trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),sources=manifest,
        versions=dict(casadi=ca.__version__,numpy=np.__version__),interval=[args.start,args.end],
        beta=args.beta,wang_alpha=1.,wang_mu_per_node=.1,wang_iota=.02,wang_radius=1.,
        frames=len(results),safe_start_frames=sum(r['safe_initial'] for r in results),summary=summary,results=results)
    (out/'benchmark.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:result[k] for k in ['frames','safe_start_frames','summary']},indent=2))


if __name__=='__main__':main()
