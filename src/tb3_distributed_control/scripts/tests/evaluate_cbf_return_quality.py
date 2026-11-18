#!/usr/bin/env python3
"""Frozen offline recovery metrics; never fed into the controller."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def evaluate(directory):
    p=Path(directory);meta=json.loads((p/'run_metadata.json').read_text())
    protocol=meta['active_protocol'];spec=json.loads((p/'dynamic_scenario.json').read_text())['specifications'][0]
    rows=list(csv.DictReader((p/'gazebo_truth.csv').open()))
    t=np.array([float(r['time']) for r in rows]);ego=np.array([[float(r['tb3_0_x']),float(r['tb3_0_y'])] for r in rows])
    actor=np.array([[float(r[spec['name']+'_x']),float(r[spec['name']+'_y'])] for r in rows])
    reference=np.asarray(protocol['reference_start'])+t[:,None]*np.asarray(protocol['reference_velocity'])
    delta=ego-reference;y=abs(delta[:,1]);error=np.linalg.norm(delta,axis=1)
    encounter=np.flatnonzero((t>=4.)&(t<=25.));cpa=int(encounter[np.argmin(np.linalg.norm(ego[encounter]-actor[encounter],axis=1))])
    post=np.flatnonzero((t>=t[cpa])&(t<=39.))
    def settled(values):
        bad=np.flatnonzero(values[post]>.05)
        next_index=bad[-1]+1 if len(bad) else 0
        if next_index>=len(post):return None
        stamp=float(t[post[next_index]])
        return stamp if 39.-stamp>=1. else None
    y_settle=settled(y);xy_settle=settled(error)
    controls=[r for r in csv.DictReader((p/'tb3_0.csv').open()) if 4<=float(r['time'])<=39]
    omega=np.array([float(r['omega']) for r in controls]);returns=[];early=[]
    for r in map(json.loads,(p/'tb3_0_solver_trace.jsonl').open()):
        s=r.get('manager',{}).get('spatial_pass',{})
        if s.get('mode')=='return':returns.append(dict(time=r['trajectory_time'],duration=s['duration']))
        if s.get('early_return_started'):
            early.append(dict(time=r['trajectory_time'],duration=s['duration'],qualified=s.get('selected_return_qualified'),scope=s.get('preview_scope')))
    assessment=json.loads((p/'evaluation.json').read_text())
    return dict(source=str(p),controller_passed=assessment['passed'],closest_approach_time=float(t[cpa]),
        settled_y_time=y_settle,settled_y_delay=None if y_settle is None else y_settle-float(t[cpa]),
        settled_xy_time=xy_settle,settled_xy_delay=None if xy_settle is None else xy_settle-float(t[cpa]),
        post_cpa_integrated_abs_y=float(np.trapz(y[post],t[post])),maximum_abs_y=float(y[t<=39].max()),
        omega_total_variation=float(abs(np.diff(omega)).sum()),
        first_return=returns[0] if returns else None,early_return_starts=early,
        all_early_starts_full_qualified=all(r['qualified'] and r['scope']=='extended_preview_qualified' for r in early),
        motion=assessment['active_motion'],statistics=assessment['robot_statistics']['0'],
        definition='CPA: actual centres in [4,25] s. Settled: thereafter through 39 s continuously <=0.05 m, at least 1 s remaining. XY uses original timed reference, not nearest path.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');args=p.parse_args()
    output=Path(args.directory)/'return_quality.json'
    if output.exists():raise FileExistsError(output)
    result=evaluate(args.directory);output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
