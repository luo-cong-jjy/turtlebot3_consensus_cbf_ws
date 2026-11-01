#!/usr/bin/env python3
"""Same logged fitted ellipses -> independent CV/CA/Singer replay.

Ground truth is accessed only AFTER production estimators generate forecasts.
This isolates the estimator on each recorded viewpoint; it does not rerun the
controller or raw-scan clustering and cannot establish closed-loop safety.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from test_cbf_dynamic import inline
from test_cbf_ca_prediction import scenario


def replay(directory):
    p=Path(directory);meta=json.loads((p/'run_metadata.json').read_text())
    if len(meta['robot_names'])!=1:raise ValueError('This replay protocol requires one robot')
    truth=list(csv.DictReader((p/'gazebo_truth.csv').open()))
    stamps=np.array([float(r['ros_time']) for r in truth]);times=np.array([float(r['time']) for r in truth])
    origin=float(np.median(stamps-times));end=float(meta['total_time'])
    specification=json.loads((p/'dynamic_scenario.json').read_text())['specifications'][0]
    names=sorted({k[:-2] for k in truth[0] if k.startswith('cbf_dynamic_') and k.endswith('_x')})
    if len(names)!=1:raise ValueError('This replay protocol requires one external obstacle')
    target=np.array([[float(r[names[0]+'_x']),float(r[names[0]+'_y'])] for r in truth])
    frames=[json.loads(s) for s in (p/'obstacle_predictions.jsonl').open()]
    if not frames or any('measurements' not in f for f in frames):
        raise ValueError('Raw fitted ellipse measurements not recorded; do not reconstruct them from filtered predictions')
    rows=[]
    for mode in ('cv','ca','singer'):
        tracker=inline.DynamicEllipseTracker(dt=float(meta['control_dt']),mode=mode,singer_tau=1.)
        buckets={};available={};eligible={}
        for frame in frames:
            stamp=float(frame['stamp'])
            fits=[inline.Ellipse(*m['center'],m['a'],m['b'],m['theta'],m['point_count']) for m in frame['measurements']]
            tracker.update(fits,stamp)
            predictions=tracker.obstacles(tracker.snapshot(stamp),stamp)
            if not 0<=stamp-origin<=end or not stamps[0]<=stamp<=stamps[-1]:continue
            now=np.array([np.interp(stamp,stamps,target[:,a]) for a in (0,1)])
            t=stamp-origin
            vb=np.array(scenario.trajectory(specification,max(0.,t-.01))[2:])
            va=np.array(scenario.trajectory(specification,t+.01)[2:])
            phase='accelerating' if np.linalg.norm(va-vb)/.02>.005 else 'constant_or_stopped'
            # Offline identity assignment ONLY. It never feeds back to update
            # or forecast. Report association coverage, not just gated scores.
            obs=min(predictions,key=lambda o:np.linalg.norm(o['center']-now)) if predictions else None
            for j in (0,5,10):
                future=stamp+j*float(meta['control_dt'])
                if future>min(stamps[-1],origin+end):continue
                for group in ('all',phase):
                    key=(group,j);eligible[key]=eligible.get(key,0)+1
                    if obs is None:continue
                    available[key]=available.get(key,0)+1
                    actual=np.array([np.interp(future,stamps,target[:,a]) for a in (0,1)])
                    absolute=np.linalg.norm(obs['prediction_centers'][j]-actual)
                    displacement=np.linalg.norm(obs['prediction_centers'][j]-obs['center']-(actual-now))
                    buckets.setdefault(key,[]).append([absolute,displacement])
        for phase,j in eligible:
            key=(phase,j);values=np.asarray(buckets.get(key,[])).reshape((-1,2))
            rows.append(dict(input_directory=p.name,source_prediction_mode=meta['prediction_mode'],
                scenario=meta['dynamic_scenario'],mode=mode,phase=phase,horizon_s=j*float(meta['control_dt']),eligible_frames=eligible[key],
                forecast_frames=available.get(key,0),coverage=available.get(key,0)/eligible[key],
                tracks_created=tracker.next_id-1,
                absolute_rmse=float(np.sqrt(np.mean(values[:,0]**2))) if len(values) else None,
                displacement_rmse=float(np.sqrt(np.mean(values[:,1]**2))) if len(values) else None))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directories',nargs='+');parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    metrics=[];sources={}
    for directory in args.directories:
        metrics.extend(replay(directory));p=Path(directory)
        sources[p.name]={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in
            ('run_metadata.json','gazebo_truth.csv','obstacle_predictions.jsonl')}
    with (out/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    package=Path(__file__).resolve().parents[2]
    code_sha={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__).resolve(),package/'scripts/tb3_controller_node_cbf.py',package/'scripts/cbf_dynamic_scenario.py']}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=sources,code_sha256=code_sha,singer_tau=1.,
        note='Identical timestamped raw fitted ellipses per recorded viewpoint. Same scan-stamp forecast epoch, no added latency. Ground truth is scoring only; nearest current-truth identity assignment without error threshold. Coverage retained; no score fabrication when no forecast. Not raw-scan processing replay or closed-loop evidence.'),indent=2))
    for row in metrics:
        if row['horizon_s']==1. and row['phase']=='all':print(row['input_directory'],row['mode'],round(row['displacement_rmse'],4),row['coverage'])


if __name__=='__main__':main()
