#!/usr/bin/env python3
"""Recompute optional numerical model-gap evidence; not physical safety."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[2];sys.path[:0]=[str(P/'scripts'),str(P/'纯仿真代码')]
from mpc_dcbf_cached_solver import CachedMPCProblem,feasible_objective_certificate


def cost(candidate,goals,previous,cfg):
    x=np.asarray(candidate['states']);u=np.asarray(candidate['controls']);q=np.asarray(candidate['plan_controls']);n=len(u)
    r=np.vstack([x[0,:2],x[0,:2]+cfg['dt']*np.cumsum(q,axis=0)])
    angle=np.arctan2(np.sin(x[:,2]-goals[:,2]),np.cos(x[:,2]-goals[:,2]));e=np.column_stack([x[:,:2]-goals[:,:2],angle])
    value=np.sum(e[:-1]**2*cfg['q_state'])+np.sum(u*u*cfg['r_control'])
    value+=np.sum((r[:-1]-goals[:-1,:2])**2*cfg['q_plan'])+np.sum((x[:-1,:2]-r[:-1])**2*cfg['q_coupling'])+np.sum(q*q*cfg['r_plan'])
    delta=np.diff(u,axis=0)
    if previous is not None:delta=np.vstack([u[0]-previous,delta])
    value+=np.sum(delta*delta*cfg['r_delta'])+np.sum(np.diff(q,axis=0)**2*cfg['r_delta_plan'])
    value+=np.sum(e[-1]**2*cfg['q_terminal'])+np.sum((r[-1]-goals[-1,:2])**2*cfg['q_plan'])+np.sum((x[-1,:2]-r[-1])**2*cfg['q_coupling'])
    return float(value)


def audit(directory):
    p=Path(directory);checked=applied=dropped=0;errors=[];maximum_difference=0.;problem=None
    for f in sorted(p.glob('tb3_*_solver_trace.jsonl')):
        for row in map(json.loads,f.open()):
            manager=row['manager'];certificate=manager.get('cycle_budget',{}).get('objective_certificate')
            if certificate is None:continue
            snapshot=manager['input_snapshot'];cfg=snapshot['config'];checked+=1
            candidate={k:np.asarray(certificate[k]) for k in ('states','controls','plan_controls')}
            candidate.update(cost=certificate['gap_upper_bound'],validated_steps=cfg['horizon'])
            previous=None if snapshot['previous'] is None else np.asarray(snapshot['previous'])
            if problem is None:problem=CachedMPCProblem(cfg,len(snapshot['obstacles']),cfg['delta_limit'])
            problem.config=cfg
            expected=cost(candidate,np.asarray(snapshot['goals']),previous,cfg);difference=abs(expected-certificate['gap_upper_bound']);maximum_difference=max(maximum_difference,difference)
            valid=problem.validate(candidate,np.asarray(snapshot['state']),snapshot['obstacles'],snapshot['bounds'],previous)
            guide=snapshot['guidance'];eligible=feasible_objective_certificate(candidate,cfg,(np.asarray(guide['vectors']),guide['level']))
            if (not valid or eligible is None or certificate['lower_bound']!=0 or difference>1e-9
                    or expected>certificate['tolerance']+1e-10 or manager['attempts']!=0):errors.append(dict(cycle=row['cycle_id'],error='Invalid numerical prefix/cost/gap evidence'))
            if row['status'] in ('ok','feasible_iterate','validated_one_step'):
                applied+=1
                if not np.allclose(row['command'],candidate['controls'][0],atol=1e-12,rtol=0):errors.append(dict(cycle=row['cycle_id'],error='Applied command differs'))
            else:dropped+=1
    return dict(passed=not errors,applicable=checked>0,certificates=checked,applied=applied,dropped_by_controller=dropped,
        maximum_cost_difference=maximum_difference,error_count=len(errors),first_error=errors[0] if errors else None,
        note='Full numerical model feasibility and nonnegative-objective gap; no exact global optimizer or physical safety claim.')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory');a.add_argument('--output',required=True);r=a.parse_args();p=Path(r.output)
    if p.exists():raise FileExistsError(p)
    result=audit(r.directory);p.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
