#!/usr/bin/env python3
"""Conditional forecast errors against offline Gazebo truth, one actor only.

Nearest current center within 0.45m associates tracks for EVALUATION ONLY.
All horizons share the <=T-max_requested_horizon current-frame window. Position errors include
partial-view ellipse-center bias; displacement errors remove current offset.
Rejected/unmatched frames are counted; this is not unconditional accuracy,
confidence coverage, a controller input, or a cross-trajectory model ranking.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def analyze(directory):
    path=Path(directory);meta=json.loads((path/'run_metadata.json').read_text())
    specs=json.loads((path/'dynamic_scenario.json').read_text())['specifications']
    if len(specs)!=1:raise ValueError('This diagnostic requires exactly one actor')
    actor=specs[0]['name'];rows=list(csv.DictReader((path/'gazebo_truth.csv').open()))
    ros=np.array([float(r['ros_time']) for r in rows]);mission=np.array([float(r['time']) for r in rows])
    truth=np.array([[float(r[actor+'_x']),float(r[actor+'_y'])] for r in rows])
    origin=ros[0]-mission[0];end=float(meta['total_time']);nodes=[0,5,10,15,25];dt=float(meta['control_dt'])
    errors={j:[] for j in nodes};early={j:[] for j in nodes};displacements={j:[] for j in nodes}
    examined=unmatched=0
    def position(stamp):return np.array([np.interp(stamp,ros,truth[:,a]) for a in (0,1)])
    for line in (path/'obstacle_predictions.jsonl').open():
        frame=json.loads(line);stamp=frame['stamp'];time=stamp-origin
        if frame['observer']!='tb3_0' or not ros[0]<=stamp<=ros[-1] or not 0<=time<=end-max(nodes)*dt:continue
        examined+=1;actual=position(stamp);tracks=frame['tracks']
        track=min(tracks,key=lambda t:np.linalg.norm(np.asarray(t['centers'][0])-actual)) if tracks else None
        if track is None or np.linalg.norm(np.asarray(track['centers'][0])-actual)>.45:
            unmatched+=1;continue
        predicted=np.asarray(track['centers'])
        for j in nodes:
            future=stamp+j*frame['dt']
            if j>=len(predicted) or future>ros[-1]:continue
            actual_future=position(future);err=float(np.linalg.norm(predicted[j]-actual_future))
            delta=float(np.linalg.norm((predicted[j]-predicted[0])-(actual_future-actual)))
            errors[j].append(err);displacements[j].append(delta)
            if time<=5.:early[j].append(delta)
    def stats(values):
        return dict(count=len(values),rmse=float(np.sqrt(np.mean(np.square(values)))) if values else None,
            p95=float(np.percentile(values,95)) if values else None)
    assert all(x==0. for x in displacements[0])
    return dict(interpretation=__doc__,source=str(path.resolve()),mpc_horizon=meta.get('mpc_horizon',10),
        examined_current_frames=examined,unmatched_current_frames=unmatched,
        horizons={str(round(j*dt,10)):dict(position=stats(errors[j]),displacement=stats(displacements[j]),
            first_5s_displacement=stats(early[j])) for j in nodes})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(analyze(a.directory),indent=2,allow_nan=False))
