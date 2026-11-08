#!/usr/bin/env python3
"""Same nominal target / mission window comparison of two static Gazebo runs.

Uses independent sampled model positions for both versions, not their legacy
error columns. A single pair is a regression check, not statistical evidence.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def evaluate(directory):
    path=Path(directory);e=json.loads((path/'evaluation.json').read_text());meta=json.loads((path/'run_metadata.json').read_text())
    if meta['trajectory_type']!='eight_slow_slow':raise ValueError('Expected original slow-eight task')
    rows=list(csv.DictReader((path/'gazebo_truth.csv').open()));start=e['leader_start_ros'];duration=e['duration']
    times=np.array([float(r['ros_time']) for r in rows])-start;valid=(times>=0)&(times<=duration)
    poses=np.array([[[float(r['x'+str(i)]),float(r['y'+str(i)])] for i in range(5)] for r in rows])[valid];t=times[valid]
    leader=np.column_stack((2.5*np.sin(t/15),.3+2.5*np.sin(t/30)))
    radius=float(meta['formation_radius']);offset=radius*np.column_stack((np.cos(np.arange(5)*2*np.pi/5),np.sin(np.arange(5)*2*np.pi/5)))
    errors=np.linalg.norm(poses-leader[:,None,:]-offset[None,:,:],axis=2)
    return dict(source=str(path.resolve()),radius=radius,truth_samples=len(t),last_mission_sample=float(t[-1]),
        mean_position_error=float(errors.mean()),position_rmse=float(np.sqrt(np.mean(errors**2))),
        final_position_error_max=float(errors[-1].max()),mean_position_error_per_robot=errors.mean(axis=0).tolist(),
        physical_proxy_clearance=e['physical_obstacle_clearance_min'],peer_proxy_clearance=e['pair_body_clearance_min'],
        source_sha256=json.loads((path/'sources/sha256.json').read_text()),
        completed=e['completed'],failure=e['failure'],reported_basic_checks=e['basic_checks_passed'],
        truth_sha256=hashlib.sha256((path/'gazebo_truth.csv').read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--teacher',required=True);p.add_argument('--current',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    teacher=evaluate(a.teacher);current=evaluate(a.current)
    if teacher['radius']!=current['radius']:raise ValueError('Formation radius differs')
    result=dict(interpretation=__doc__,teacher=teacher,current=current,mean_error_change=current['mean_position_error']-teacher['mean_position_error'])
    out.write_text(json.dumps(result,indent=2));print(json.dumps({k:{m:v[m] for m in ['mean_position_error','position_rmse','final_position_error_max','physical_proxy_clearance','peer_proxy_clearance']} for k,v in [('teacher',teacher),('current',current)]},indent=2))
