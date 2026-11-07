#!/usr/bin/env python3
"""Observed motion quality, separate from geometric scenario acceptance.

Reads recorded samples without changing them. Low speed refers to commands,
not measured velocity; a stationary command is not a dynamic-safety proof.
Intervals with missing samples (>1.5 control periods) are excluded from
duration integrals, and are reported explicitly.
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np


def analyze(directory):
    directory=Path(directory)
    meta=json.loads((directory/'run_metadata.json').read_text())
    dt=float(meta.get('control_dt',.1));result={}
    for path in sorted(directory.glob('tb3_*.csv')):
        rows=list(csv.DictReader(path.open()))
        if not rows:continue
        def column(name):return np.array([float(r[name]) for r in rows])
        t=column('evaluation_time');v=column('v');w=column('omega')
        error=np.hypot(column('consensus_error_x'),column('consensus_error_y'))
        delta=np.diff(t);valid=(delta>0)&(delta<=1.5*dt)
        intervals=np.where(valid,delta,0.)
        low=np.abs(v[:-1])<.02;lag=error[:-1]>.25
        longest=current=0.
        for i in range(len(intervals)):
            current=current+intervals[i] if valid[i] and low[i] and lag[i] else 0.
            longest=max(longest,current)
        evalid=np.isfinite(error[:-1])&valid
        elapsed=intervals[evalid].sum()
        statuses=collections.Counter(r['status'] for r in rows)
        bad=np.flatnonzero(error>.25)
        total_var=float(np.abs(np.diff(w))[valid].sum())
        item=dict(samples=len(rows),first_time=float(t[0]),last_time=float(t[-1]),
            excluded_gap_count=int((~valid).sum()),observed_duration=float(intervals.sum()),
            position_rmse=float(np.sqrt(np.mean(error[np.isfinite(error)]**2))),
            time_mean_position_error=float((error[:-1][evalid]*intervals[evalid]).sum()/elapsed) if elapsed else None,
            final_position_error=float(error[-1]),peak_position_error=float(np.nanmax(error)),
            last_error_above_25cm=float(t[bad[-1]]) if len(bad) else None,
            command_low_speed_seconds=float(intervals[low].sum()),
            command_low_speed_with_lag_seconds=float(intervals[low&lag].sum()),
            longest_command_low_speed_with_lag_seconds=float(longest),
            reverse_seconds=float(intervals[v[:-1]<-.02].sum()),
            omega_total_variation=total_var,
            max_omega_command_step=float(np.max(np.abs(np.diff(w))[valid])) if valid.any() else None,
            measured_pose_path_length=float(np.hypot(np.diff(column('x')),np.diff(column('y')))[valid].sum()),
            status_counts=dict(statuses))
        trace=directory/(path.stem+'_solver_trace.jsonl')
        decisions=[];inactive=active=0
        if trace.exists():
            for line in trace.open():
                manager=json.loads(line).get('manager',{})
                progress=manager.get('progress',{})
                if manager.get('guide_level',0)>0:
                    if progress.get('circulation_active') is True:active+=1
                    elif progress.get('circulation_active') is False:inactive+=1
                d=manager.get('progress_decision')
                if d:
                    preview=d['previews'].get(d['selected'],{})
                    decisions.append(bool(preview.get('circulation_active',False)))
        item.update(reference_decisions=len(decisions),active_circulation_decisions=sum(decisions),
            active_guide_cycles=active,explicit_inactive_guide_cycles=inactive)
        item['inactive_guidance_gate_check']=(inactive==0) if meta.get('active_guidance_only',False) else None
        result[path.stem]=item
    return dict(source=str(directory.resolve()),thresholds=dict(low_command_speed=.02,lag=.25,max_interval=1.5*dt),
        interpretation='Descriptive command/measurement metrics; not a safety or prediction-benefit test.',
        parameters={k:meta.get(k) for k in ('prediction_mode','transport_geometry','adaptive_recovery','predictive_bypass','active_guidance_only','prediction_risk','suppress_guidance','risk_parameters')},
        robots=result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',required=True);args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(analyze(args.directory),indent=2,allow_nan=False))
