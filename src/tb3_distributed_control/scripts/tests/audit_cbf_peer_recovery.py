#!/usr/bin/env python3
"""Check worker recovery references on original frozen hard-window inputs.

Only the logged first N steps are reconstructible here. Long peer extension is
a soft hypothesis, not a safety certificate. Stale/carried summaries are counted
separately rather than attributed to a newly generated recovery reference.
"""
from pathlib import Path
import argparse,json,sys,collections
import numpy as np
P=Path(__file__).resolve().parents[2];sys.path[:0]=[str(P/'scripts'),str(P/'纯仿真代码')]
from mpc_dcbf_cached_solver import CachedMPCProblem,rollout


def audit(directory):
    errors=[];seen=set();checked=0;reverse=0;stale=0;used=0;modes=collections.Counter();difference=0.
    for file in Path(directory).glob('tb3_*_solver_trace.jsonl'):
        for event in map(json.loads,file.open()):
            current=event['manager'];local=current.get('local_worker',{});record=local.get('record',{})
            summary=record.get('peer_recovery',{});mode=summary.get('mode');key=(file.name,local.get('sequence'))
            if mode in ('return','hold_lane','outer_lane') and current.get('seed')=='local_worker_current':used+=1
            if key in seen or not record:continue
            seen.add(key);modes[mode]+=1
            if mode not in ('return','hold_lane','outer_lane'):continue
            saved=record['input_snapshot'];stamp=max(o.get('recovery_stamp',0.) for o in saved['obstacles'] if o.get('peer',False))
            if abs(summary.get('stamp',-1)-stamp)>1e-8 or record.get('spatial_timeout_carry',False):stale+=1;continue
            cfg=saved['config'];guide=saved['guidance'];n=cfg['horizon'];dt=cfg['dt']
            vector=np.asarray(guide['vectors']);angle=np.asarray(guide['headings']);state=np.asarray(saved['state'])
            direction=np.column_stack((np.cos(angle),np.sin(angle)))
            speed=np.sum(vector*direction,axis=1)
            control=np.column_stack((speed,np.diff(np.r_[state[2],angle])/dt))
            candidate=dict(controls=control,states=rollout(state,control,dt),plan_controls=vector)
            problem=CachedMPCProblem.__new__(CachedMPCProblem);problem.config=cfg;problem.dt=dt;problem.constraint_mode=cfg.get('constraint_mode','dcbf')
            if not problem.validate(candidate,state,saved['obstacles'],saved['bounds'],np.asarray(saved['previous'])):
                errors.append(dict(worker=key,error='Recovery reference fails original hard prefix'))
            error=float(np.abs(vector-speed[:,None]*direction).max());difference=max(difference,error)
            if error>1e-9 or guide['level']!=1.:errors.append(dict(worker=key,error='Guide is not recorded recovery velocity'))
            if summary['selected_peer_clearance']<.03-1e-12 or summary['minimum_speed'][summary['selected']]<.09-1e-12:
                errors.append(dict(worker=key,error='Selected soft criteria differ from predeclared values'))
            checked+=1;reverse+=int(summary.get('reverse_chart',False))
    return dict(passed=checked>0 and not errors,checked_original_hard_prefix_references=checked,
        reverse_chart_references=reverse,stale_or_carried_summaries_not_attributed=stale,
        foreground_cycles_using_worker_with_recovery_summary=used,unique_worker_modes=dict(modes),
        maximum_velocity_reconstruction_error=difference,error_count=len(errors),first_error=errors[0] if errors else None,scope=__doc__)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(audit(a.directory),indent=2))
