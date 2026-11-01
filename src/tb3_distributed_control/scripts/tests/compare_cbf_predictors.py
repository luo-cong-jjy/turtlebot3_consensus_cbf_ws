#!/usr/bin/env python3
"""Paired common-input surrogate replay, not a closed-loop/sensor benchmark.

All modes get identical timestamped noisy ellipse measurements/dropouts. Truth
is used ONLY here for scoring, never for the production estimator. Parameters
are fixed before evaluation; per-seed errors are retained, including CA losses.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import yaml
from test_cbf_dynamic import inline
from test_cbf_ca_prediction import scenario
from test_cbf_inline_geometry import PACKAGE


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--seeds',type=int,default=10)
    parser.add_argument('--seed-start',type=int,default=0)
    parser.add_argument('--modes',nargs='+',choices=['frozen','cv','ca','singer'],default=['frozen','cv','ca'])
    parser.add_argument('--singer-tau',type=float,default=1.)
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    specs=yaml.safe_load((PACKAGE/'config/cbf_dynamic_scenarios.yaml').read_text())
    cases={name:specs[name][0] for name in ('crossing','accelerating_crossing','decelerating_crossing','stop_go_crossing','maneuver')}
    cases['stationary']=dict(start=[2.,1.],velocity=[0.,0.])
    rows=[];inputs=[]
    for case,spec in cases.items():
        for noise in (.005,.015):
            for seed in range(args.seed_start,args.seed_start+args.seeds):
                rng=np.random.default_rng(seed+2811)
                times=np.arange(301)*.1+rng.uniform(-.009,.009,301);times[0]=0.
                measurements=[]
                for t in times:
                    x,y,_,_=scenario.trajectory(spec,t)
                    center=np.array([x,y])+rng.normal(0,noise,2)
                    present=bool(rng.random()>.10 and not 8.<t<8.3)
                    measurements.append((t,center,present))
                    inputs.append(dict(case=case,noise=noise,seed=seed,stamp=t,center=center.tolist(),present=present))
                for mode in args.modes:
                    tracker=inline.DynamicEllipseTracker(mode=mode,singer_tau=args.singer_tau)
                    buckets={}
                    for t,center,present in measurements:
                        fits=[inline.Ellipse(*center,.20,.12,0.,20)] if present else []
                        tracker.update(fits,t)
                        snapshot=tracker.snapshot(t)
                        if t<3. or not snapshot:continue
                        # A one-target surrogate: identity switches are counted
                        # separately; they are not hidden by nearest-truth gating.
                        track=max(snapshot,key=lambda item:item['seen'])
                        obs=tracker.forecast(track,t+.03)
                        if obs is None:continue
                        now=t+.03;p0=np.array(scenario.trajectory(spec,now)[:2])
                        v_before=np.array(scenario.trajectory(spec,max(0.,now-.01))[2:])
                        v_after=np.array(scenario.trajectory(spec,now+.01)[2:])
                        phase='accelerating' if np.linalg.norm(v_after-v_before)/.02>.005 else 'constant_or_stopped'
                        for j in (5,10):
                            future=now+j*.1
                            if future>30.:continue
                            target=np.array(scenario.trajectory(spec,future)[:2])
                            prediction=obs['prediction_centers'][j]
                            abs_error=np.linalg.norm(prediction-target)
                            disp_error=np.linalg.norm((prediction-obs['center'])-(target-p0))
                            key=(phase,j)
                            buckets.setdefault(key,[]).append((abs_error,disp_error,np.linalg.norm(obs['acceleration'])))
                    for (phase,j),values in buckets.items():
                        a=np.asarray(values)
                        rows.append(dict(case=case,noise=noise,seed=seed,mode=mode,phase=phase,horizon_s=j*.1,
                            samples=len(a),absolute_rmse=float(np.sqrt(np.mean(a[:,0]**2))),
                            displacement_rmse=float(np.sqrt(np.mean(a[:,1]**2))),
                            absolute_max=float(a[:,0].max()),mean_acceleration=float(a[:,2].mean()),tracks_created=tracker.next_id-1))
    with (out/'measurements.jsonl').open('w') as stream:
        for item in inputs:stream.write(json.dumps(item)+'\n')
    with (out/'per_seed_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for case in cases:
        for noise in (.005,.015):
            for phase in ('accelerating','constant_or_stopped'):
                for j in (.5,1.):
                    modes={}
                    for mode in args.modes:
                        group=[r for r in rows if (r['case'],r['noise'],r['mode'],r['phase'],r['horizon_s'])==(case,noise,mode,phase,j)]
                        if not group:continue
                        modes[mode]={metric:float(np.mean([r[metric] for r in group])) for metric in ('absolute_rmse','displacement_rmse')}
                        modes[mode]['seed_count']=len(group)
                    if modes:summary.append(dict(case=case,noise=noise,phase=phase,horizon_s=j,modes=modes))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    sources={str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [PACKAGE/'scripts/tb3_controller_node_cbf.py',Path(__file__).resolve(),PACKAGE/'scripts/cbf_dynamic_scenario.py',
         PACKAGE/'config/cbf_dynamic_scenarios.yaml']}
    (out/'protocol.json').write_text(json.dumps(dict(seeds=args.seeds,seed_start=args.seed_start,modes=args.modes,singer_tau=args.singer_tau,source_sha256=sources,dt=.1,horizon=10,
        ca_jerk_density=.01,ca_acceleration_variance=.04,parameters='Fixed before replay; no evaluation-set tuning',
        note='Synthetic full-ellipse centers, independent Gaussian noise, 10% random misses plus a fixed gap. No visible-contour bias, ego-pose errors or real laser clustering. Means of per-seed RMSE, not pooled sample RMSE or physical safety evidence.'),indent=2))
    for item in summary:
        if item['horizon_s']==1. and item['noise']==.005:
            print(item['case'],item['phase'],{k:round(v['displacement_rmse'],4) for k,v in item['modes'].items()})


if __name__=='__main__':main()
