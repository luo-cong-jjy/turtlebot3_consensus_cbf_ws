#!/usr/bin/env python3
"""Identical-input observer replay. Gazebo truth is OFFLINE evaluation only."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from test_cbf_inline_geometry import inline
from mpc_dcbf_static_obstacles import reference_covariance_matrices,reference_axis_inflation
inline.reference_covariance_matrices=reference_covariance_matrices
inline.reference_axis_inflation=reference_axis_inflation


def replay(directory):
    p=Path(directory);meta=json.loads((p/'run_metadata.json').read_text())
    if meta['prediction_mode']!='cv':raise ValueError('This stage validates CV geometry only')
    truth=list(csv.DictReader((p/'gazebo_truth.csv').open()))
    times=np.array([float(r['time']) for r in truth]);begin=float(np.median([float(r['ros_time'])-float(r['time']) for r in truth]))
    spec=json.loads((p/'dynamic_scenario.json').read_text())['specifications']
    if len(spec)!=1:raise ValueError('Single external target only in this offline audit')
    name=spec[0]['name'];radius=spec[0]['radius']
    positions=np.array([[float(r[name+'_x']),float(r[name+'_y'])] for r in truth])
    def actual(t):return np.array([np.interp(t,times,positions[:,j]) for j in (0,1)])
    args=dict(dt=meta['control_dt'],horizon=meta['mpc_horizon'],mode='cv',
        reaction_time=meta['dynamic_reaction_time'],extra_margin=meta['dynamic_extra_margin'],
        transport_geometry=meta['transport_geometry'],acceleration_buffer=meta['acceleration_buffer'],
        stage_velocity_guidance=meta['stage_velocity_guidance'])
    models={kind:inline.DynamicEllipseTracker(**args,geometry_observer=kind) for kind in ('mvee','circle')}
    frames=[r for r in map(json.loads,(p/'obstacle_predictions.jsonl').open()) if r['observer']=='tb3_0']
    records=[];replay_error=0.;replay_checks=0;gates={};holds=0;resets=0
    for frame in frames:
        stamp=frame['stamp'];t=stamp-begin;results={};ids={}
        for kind,model in models.items():
            fits=[]
            for measurement in frame['measurements']:
                raw=measurement.get('base_enclosure',measurement)
                e=inline.Ellipse(*raw['center'],raw['a'],raw['b'],raw['theta'],raw['point_count'])
                if kind=='circle':
                    if 'points' not in measurement:raise ValueError('No saved cluster points; do not fabricate observations')
                    e.circle_candidate=inline.circle_observation(measurement['points'],e,frame['sensor_origin'])
                    reason=e.circle_candidate['reason'];gates[reason]=gates.get(reason,0)+1
                fits.append(e)
            model.update(fits,stamp);current=model.obstacles(model.snapshot(stamp),stamp)
            results[kind]={o['id']:o for o in current};ids[kind]=[e.track_id for e in fits]
        configured=meta.get('geometry_observer','mvee')
        saved={r['id']:r for r in frame['tracks']};found=results[configured]
        if set(saved)!=set(found):raise AssertionError('Recorded track set cannot be reproduced')
        for key,old in saved.items():
            new=found[key]
            err=max(float(abs(new['prediction_centers']-old['centers']).max()),float(abs(new['prediction_axes']-old['axes']).max()))
            replay_error=max(replay_error,err);replay_checks+=1
        if not 4<=t<=20 or len(frame['measurements'])!=1:continue
        row=dict(time=t,truth_position=actual(t).tolist(),truth_velocity=(actual(t+.5)-actual(t-.5)).tolist())
        for kind in models:
            o=results[kind][ids[kind][0]];velocity=np.asarray(o['velocity']);center=o['center']
            entry=dict(velocity_error=float(np.linalg.norm(velocity-(actual(t+.5)-actual(t-.5)))),
                centre_error=float(np.linalg.norm(center-actual(t))),observation_kind=o['observation_kind'],
                position_observed=o['position_observed'],observer_resets=o['observer_resets'],
                velocity=velocity.tolist(),center=center.tolist(),displacement_error={},absolute_prediction_error={},
                occupied_circle_coverage={},collision_proxy_coverage={})
            if kind=='circle':holds+=int(not o['position_observed']);resets=max(resets,o['observer_resets'])
            for horizon in (0.,1.,2.):
                stage=min(int(round(horizon/meta['control_dt'])),meta['mpc_horizon'])
                prediction=center+horizon*velocity
                if horizon<=meta['mpc_horizon']*meta['control_dt']:prediction=o['prediction_centers'][stage]
                expected=actual(t+horizon)
                entry['displacement_error'][str(horizon)]=float(np.linalg.norm((prediction-center)-(expected-actual(t))))
                entry['absolute_prediction_error'][str(horizon)]=float(np.linalg.norm(prediction-expected))
                a,b=o['prediction_axes'][stage];theta=o['theta'];c,s=np.cos(theta),np.sin(theta)
                angles=np.arange(256)*(2*np.pi/256)
                for field,body in [('occupied_circle_coverage',radius),('collision_proxy_coverage',radius+.13)]:
                    axes=np.array([a,b])
                    if field=='collision_proxy_coverage':axes*=1+.12/b
                    delta=expected+body*np.column_stack((np.cos(angles),np.sin(angles)))-prediction
                    local=delta@np.array([[c,-s],[s,c]])
                    upper=float(np.linalg.norm(local/axes,axis=1).max())+body/min(axes)*np.pi/256
                    entry[field][str(horizon)]=bool(upper<=1.)
            row[kind]=entry
        records.append(row)
    summary={}
    for kind in models:
        rows=[r[kind] for r in records]
        summary[kind]=dict(frames=len(rows),velocity_rmse=float(np.sqrt(np.mean([r['velocity_error']**2 for r in rows]))),
            current_centre_rmse=float(np.sqrt(np.mean([r['centre_error']**2 for r in rows]))),
            displacement_rmse={str(h):float(np.sqrt(np.mean([r['displacement_error'][str(h)]**2 for r in rows]))) for h in (1.,2.)},
            absolute_centre_prediction_rmse={str(h):float(np.sqrt(np.mean([r['absolute_prediction_error'][str(h)]**2 for r in rows]))) for h in (1.,2.)},
            occupied_circle_coverage={str(h):float(np.mean([r['occupied_circle_coverage'][str(h)] for r in rows])) for h in (0.,1.,2.)},
            collision_proxy_coverage={str(h):float(np.mean([r['collision_proxy_coverage'][str(h)] for r in rows])) for h in (0.,1.,2.)})
    return dict(source=str(p),configured_observer=meta.get('geometry_observer','mvee'),
        configured_replay_checks=replay_checks,configured_maximum_difference=replay_error,
        configured_replay_passed=replay_checks>0 and replay_error<1e-8,window=[4.,20.],
        circle_gate_counts=gates,circle_hold_frames=holds,circle_maximum_resets=resets,summary=summary,records=records,
        interpretation='Same measurements, not a closed-loop counterfactual. Primary velocity/displacement errors cancel fixed centre offsets; absolute centre error changes target semantics. Coverage is for known evaluation cylinders with a conservative angular remainder; 2 s uses held hard-window axes. Truth never enters either observer.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory');parser.add_argument('--output',required=True);a=parser.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    result=replay(a.directory);out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    if not result['configured_replay_passed']:raise SystemExit(1)
