#!/usr/bin/env python3
"""Replay scan-time measurements to explain geometry used at control time.

First verify the configured forecast against the saved control inputs. Then
compare both contour-time conventions on the identical snapshot. No truth
or scenario specification enters this replay; it is not a safety test.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from test_cbf_dynamic import inline
from mpc_dcbf_cached_solver import barrier_values


def replay(directory,observer='tb3_0'):
    p=Path(directory);m=json.loads((p/'run_metadata.json').read_text());mode=m['prediction_mode']
    model=inline.make_dynamic_tracker(mode=mode,dt=m['control_dt'],
        ca_jerk_density=m['ca_jerk_density'],ca_acceleration_variance=m['ca_acceleration_variance'],
        singer_tau=m['singer_tau'],reaction_time=m['dynamic_reaction_time'],extra_margin=m['dynamic_extra_margin'],
        acceleration_buffer=m['acceleration_buffer'],stage_velocity_guidance=m['stage_velocity_guidance'],
        transport_geometry=False,**(m['reliability_parameters'] if mode=='adaptive' else {}))
    frames=[f for f in map(json.loads,(p/'obstacle_predictions.jsonl').open()) if f['observer']==observer]
    rows={int(r['cycle_id']):r for r in csv.DictReader((p/(observer+'.csv')).open())}
    def convention(value):
        model.transport_geometry=value
        if mode=='adaptive':
            for bank in model.filters.values():bank.transport_geometry=value
    index=0;records=[];maximum=0.;failure=None
    for event in map(json.loads,(p/(observer+'_solver_trace.jsonl')).open()):
        d=event.get('manager',{}).get('progress_decision')
        if d is None:continue
        row=rows[event['cycle_id']];scan=float(row['ros_time'])-float(row['geometry_age']);now=d['decision_time']
        while index<len(frames) and frames[index]['stamp']<=scan+1e-7:
            f=frames[index];fits=[inline.Ellipse(*v['center'],v['a'],v['b'],v['theta'],v['point_count']) for v in f['measurements']]
            model.update(fits,f['stamp']);index+=1
        expected={o['name']:o for o in d['obstacles'] if not o.get('peer',False)};found=set()
        for track in model.snapshot(scan):
            convention(False);old=model.forecast(track,now)
            if old is None or old['name'] not in expected:continue
            convention(True);new=model.forecast(track,now);saved=expected[old['name']];found.add(old['name'])
            configured=new if m.get('transport_geometry',False) else old
            difference=max(float(np.abs(configured['prediction_centers']-saved['prediction_centers']).max()),
                float(np.abs(configured['prediction_axes']-saved['prediction_axes']).max()))
            maximum=max(maximum,difference)
            if difference>1e-8:
                failure=dict(cycle=event['cycle_id'],obstacle=old['name'],difference=difference);break
            state=np.array(d['state'])[None,:];margin=d['config']['margin']
            records.append(dict(time=event['trajectory_time'],cycle=event['cycle_id'],obstacle=old['name'],
                age=now-track['seen'],old_axes=old['prediction_axes'][0].tolist(),transported_axes=new['prediction_axes'][0].tolist(),
                old_h0=float(barrier_values(state,[old],margin)[0,0]),transported_h0=float(barrier_values(state,[new],margin)[0,0]),
                old_guard_center=old['guard_measurement_center'].tolist(),transported_guard_center=new['guard_measurement_center'].tolist()))
        if failure is None and found!=set(expected):failure=dict(cycle=event['cycle_id'],missing=sorted(set(expected)-found))
        if failure is not None:break
    return dict(observer=observer,verified_forecasts=len(records),max_configured_forecast_difference=maximum,
        passed=failure is None and bool(records),first_mismatch=failure,records=records,
        note='Each configured forecast matched recorded inputs before counterfactual geometry comparison. The transported contour assumes the same estimated translation; it does not certify full-object coverage or physical safety.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--observer',default='tb3_0');parser.add_argument('--output',required=True);a=parser.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError('Choose a new replay output')
    result=replay(a.directory,a.observer);output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    if not result['passed']:raise SystemExit(1)
