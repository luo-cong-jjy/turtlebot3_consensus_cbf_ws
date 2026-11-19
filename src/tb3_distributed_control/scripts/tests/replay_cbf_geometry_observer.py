#!/usr/bin/env python3
"""Identical-input observer replay. Gazebo truth is OFFLINE evaluation only."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from test_cbf_inline_geometry import inline
from mpc_dcbf_static_obstacles import reference_covariance_matrices,reference_axis_inflation
inline.reference_covariance_matrices=reference_covariance_matrices
inline.reference_axis_inflation=reference_axis_inflation


def replay(directory,compare_retention=False,reject_circle_interval=None,observer='tb3_0'):
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
    kinds=('mvee','circle','circle_prior') if compare_retention else ('mvee','circle')
    models={kind:inline.DynamicEllipseTracker(**args,geometry_observer='mvee' if kind=='mvee' else 'circle',
        circle_prior_retention=(kind=='circle_prior' if compare_retention else
            kind=='circle' and meta.get('circle_prior_retention',False))) for kind in kinds}
    configured=meta.get('geometry_observer','mvee')
    configured_key=('circle_prior' if compare_retention and configured=='circle'
        and meta.get('circle_prior_retention',False) else configured)
    evaluated_kinds=tuple(models)
    if reject_circle_interval is not None:
        if len(reject_circle_interval)!=2 or reject_circle_interval[0]>=reject_circle_interval[1]:
            raise ValueError('Expected ordered offline quality-rejection interval')
        # Validate the UNMODIFIED recorded model separately from the stressed
        # counterfactual observers. No original prediction is certified after
        # silently changing its input validity.
        models['_recorded']=inline.DynamicEllipseTracker(**args,geometry_observer=configured,
            circle_prior_retention=meta.get('circle_prior_retention',False))
        configured_key='_recorded'
    private=p/(observer+'_local_clusters.jsonl')
    frame_source=private if private.exists() else p/'obstacle_predictions.jsonl'
    frames=[r for r in map(json.loads,frame_source.open()) if r['observer']==observer]
    records=[];replay_error=0.;replay_checks=0;gates={};holds=0;resets=0;snapshots={}
    for frame in frames:
        stamp=frame['stamp'];t=stamp-begin;results={};ids={}
        for kind,model in models.items():
            fits=[]
            for measurement in frame['measurements']:
                raw=measurement.get('base_enclosure',measurement)
                e=inline.Ellipse(*raw['center'],raw['a'],raw['b'],raw['theta'],raw['point_count'])
                if model.geometry_observer=='circle':
                    if 'points' not in measurement:raise ValueError('No saved cluster points; do not fabricate observations')
                    e.circle_candidate=inline.circle_observation(measurement['points'],e,frame['sensor_origin'])
                    if (reject_circle_interval is not None and kind!='_recorded'
                            and reject_circle_interval[0]<=t<=reject_circle_interval[1]):
                        e.circle_candidate=dict(accepted=False,reason='offline_injected_quality_loss')
                    reason=e.circle_candidate['reason'];gates[reason]=gates.get(reason,0)+1
                fits.append(e)
            model.update(fits,stamp);current=model.obstacles(model.snapshot(stamp),stamp)
            results[kind]={o['id']:o for o in current};ids[kind]=[e.track_id for e in fits]
        snapshots[round(stamp,8)]=models[configured_key].snapshot(stamp)
        saved={r['id']:r for r in frame['tracks']};found=results[configured_key]
        if set(saved)!=set(found):raise AssertionError('Recorded track set cannot be reproduced')
        for key,old in saved.items():
            new=found[key]
            err=max(float(abs(new['prediction_centers']-old['centers']).max()),float(abs(new['prediction_axes']-old['axes']).max()))
            replay_error=max(replay_error,err);replay_checks+=1
        if not 4<=t<=20:continue
        # OFFLINE ground-truth labels identify the one external evaluation
        # actor, rather than quietly scoring a missed peer as that actor.
        matched=[]
        for index,measurement in enumerate(frame['measurements']):
            points=np.asarray(measurement.get('points',[]),float)
            if len(points) and np.mean(abs(np.linalg.norm(points-actual(t),axis=1)-radius)<=.04)>=.8:
                matched.append(index)
        if len(matched)!=1:continue
        measurement_index=matched[0]
        row=dict(time=t,truth_position=actual(t).tolist(),truth_velocity=(actual(t+.5)-actual(t-.5)).tolist())
        for kind in evaluated_kinds:
            o=results[kind][ids[kind][measurement_index]];velocity=np.asarray(o['velocity']);center=o['center']
            entry=dict(velocity_error=float(np.linalg.norm(velocity-(actual(t+.5)-actual(t-.5)))),
                centre_error=float(np.linalg.norm(center-actual(t))),observation_kind=o['observation_kind'],
                position_observed=o['position_observed'],observer_resets=o['observer_resets'],
                prior_shape_retained=o.get('prior_shape_retained',False),
                circle_quality_degraded=o.get('circle_quality_degraded',False),
                motion_observation_age=o.get('motion_observation_age',0.),
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
    for kind in evaluated_kinds:
        rows=[r[kind] for r in records]
        summary[kind]=dict(frames=len(rows),velocity_rmse=float(np.sqrt(np.mean([r['velocity_error']**2 for r in rows]))),
            current_centre_rmse=float(np.sqrt(np.mean([r['centre_error']**2 for r in rows]))),
            displacement_rmse={str(h):float(np.sqrt(np.mean([r['displacement_error'][str(h)]**2 for r in rows]))) for h in (1.,2.)},
            absolute_centre_prediction_rmse={str(h):float(np.sqrt(np.mean([r['absolute_prediction_error'][str(h)]**2 for r in rows]))) for h in (1.,2.)},
            occupied_circle_coverage={str(h):float(np.mean([r['occupied_circle_coverage'][str(h)] for r in rows])) for h in (0.,1.,2.)},
            collision_proxy_coverage={str(h):float(np.mean([r['collision_proxy_coverage'][str(h)] for r in rows])) for h in (0.,1.,2.)})
    control_checks=0;control_error=0.;missing=[]
    rows={int(r['cycle_id']):r for r in csv.DictReader((p/(observer+'.csv')).open())}
    for event in map(json.loads,(p/(observer+'_solver_trace.jsonl')).open()):
        saved=event.get('manager',{}).get('input_snapshot')
        if saved is None:continue
        row=rows[event['cycle_id']];now=float(row['ros_time']);scan=now-float(row['geometry_age'])
        state=snapshots.get(round(scan,8))
        if state is None:missing.append(dict(cycle=event['cycle_id'],scan=scan));continue
        expected={o['name']:o for o in models[configured_key].obstacles(state,now)}
        for obs in saved['obstacles']:
            if obs['name']=='inactive' or obs.get('peer',False):continue
            if obs['name'] not in expected:missing.append(dict(cycle=event['cycle_id'],track=obs['name']));continue
            found=expected[obs['name']]
            difference=max(float(abs(np.asarray(obs['prediction_centers'])-found['prediction_centers']).max()),
                           float(abs(np.asarray(obs['prediction_axes'])-found['prediction_axes']).max()))
            control_error=max(control_error,difference);control_checks+=1
    return dict(source=str(p),observer=observer,frame_source=frame_source.name,
        metric_target_label='Offline radial cluster-to-actor match only; never used online',
        metric_labelled_frames=len(records),configured_observer=meta.get('geometry_observer','mvee'),
        configured_circle_prior_retention=meta.get('circle_prior_retention',False),
        comparison_includes_circle_prior=compare_retention,
        offline_quality_rejection_interval=reject_circle_interval,
        configured_replay_checks=replay_checks,configured_maximum_difference=replay_error,
        configured_replay_passed=replay_checks>0 and replay_error<1e-8,window=[4.,20.],
        control_replay_checks=control_checks,control_maximum_difference=control_error,
        control_replay_passed=control_checks>0 and not missing and control_error<1e-8,
        control_replay_missing=missing,
        circle_gate_counts=gates,circle_hold_frames=holds,circle_maximum_resets=resets,summary=summary,records=records,
        interpretation='Same measurements, not a closed-loop counterfactual. Primary velocity/displacement errors cancel fixed centre offsets; absolute centre error changes target semantics. Coverage is for known evaluation cylinders with a conservative angular remainder; 2 s uses held hard-window axes. Truth never enters either observer.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory');parser.add_argument('--output',required=True)
    parser.add_argument('--compare-retention',action='store_true')
    parser.add_argument('--observer',default='tb3_0')
    parser.add_argument('--reject-circle-interval',type=float,nargs=2,help='OFFLINE counterfactual only');a=parser.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    result=replay(a.directory,compare_retention=a.compare_retention,reject_circle_interval=a.reject_circle_interval,observer=a.observer);out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    if not result['configured_replay_passed']:raise SystemExit(1)
