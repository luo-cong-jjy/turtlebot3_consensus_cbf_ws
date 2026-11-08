#!/usr/bin/env python3
"""Replay recorded first-command bounds/selected safety model on exact inputs.

Only statuses with an adopted validated command are checked. Deadline stops
are counted separately; zero commands are not silently certified as safe.
This checks the sampled model, not continuous or physical safety.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
from mpc_dcbf_cached_solver import first_step_mask


def audit(directory):
    path=Path(directory);checked=0;missing=0;stops=0;errors=[];pose_max=0.
    for trace in sorted(path.glob('tb3_*_solver_trace.jsonl')):
        name=trace.name.replace('_solver_trace.jsonl','')
        rows={int(r['cycle_id']):r for r in csv.DictReader((path/(name+'.csv')).open())}
        for line in trace.open():
            r=json.loads(line);manager=r.get('manager',{});snapshot=manager.get('input_snapshot')
            if snapshot is None:missing+=1;continue
            row=rows[r['cycle_id']];state=np.asarray(snapshot['state'])
            delta=float(np.max(np.abs(state-np.array([float(row[k]) for k in ('x','y','theta')]))))
            pose_max=max(pose_max,delta)
            if delta>1e-10:errors.append(dict(cycle_id=r['cycle_id'],error='Input state differs from metric pose',difference=delta))
            if r['status'] not in ('ok','feasible_iterate','validated_one_step'):
                stops+=1;continue
            valid,_,_=first_step_mask(state,np.asarray(r['command'])[None,:],snapshot['obstacles'],
                snapshot['bounds'],np.asarray(snapshot['previous']),snapshot['config'])
            checked+=1
            if not valid[0]:errors.append(dict(cycle_id=r['cycle_id'],error='Adopted command fails selected first-step model'))
    return dict(applicable=checked+stops>0,passed=not errors and checked>0,checked_adopted_first_steps=checked,
        uncertified_other_statuses=stops,records_without_snapshot=missing,maximum_input_metric_pose_difference=pose_max,
        first_error=errors[0] if errors else None,error_count=len(errors),interpretation=__doc__)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(audit(args.directory),indent=2))
