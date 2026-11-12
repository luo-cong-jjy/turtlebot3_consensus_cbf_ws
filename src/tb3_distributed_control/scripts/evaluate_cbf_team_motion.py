#!/usr/bin/env python3
"""Independent two-robot motion/formation acceptance; no control publishers."""
import csv
import json
from pathlib import Path
import numpy as np
from evaluate_cbf_active_motion import evaluate_motion


def evaluate_team(times, positions, actor, actor_radius, protocol, scenario, fault_counts=(0, 0)):
    t=np.asarray(times,float);p=np.asarray(positions,float);actor=np.asarray(actor,float)
    if (len(t)<3 or p.shape!=(2,len(t),2) or actor.shape!=(len(t),2)
            or not np.isfinite(p).all() or not np.isfinite(t).all() or not np.isfinite(actor).all()
            or not np.all(np.diff(t)>0)):
        return dict(passed=False,failed_checks=['invalid_team_truth'])
    required=protocol['spatial_required'][scenario]
    reports={}
    for i in range(2):
        local=dict(protocol,reference_start=(np.asarray(protocol['reference_start'])+protocol['robot_offsets'][i]).tolist())
        motion=evaluate_motion(t,p[i],actor,actor_radius,local,fault_counts[i])
        checks=dict(motion['checks'])
        # Preserve the observed behavior fact. An unthreatened robot need not
        # invent a lateral manoeuvre; roles were fixed before controller trials.
        observed=checks.pop('spatial_detour_observed')
        checks['spatial_detour_required_satisfied']=not required[i] or observed
        reports['tb3_'+str(i)]=dict(motion=motion,spatial_required=bool(required[i]),
            checks=checks,accepted=bool(all(checks.values())),failed_checks=[k for k,v in checks.items() if not v])
    delta=p[0]-p[1];radii=2*protocol['body_radius'];sample=float(np.linalg.norm(delta,axis=1).min()-radii)
    change=np.diff(delta,axis=0)
    fraction=np.clip(-np.sum(delta[:-1]*change,axis=1)/np.maximum(np.sum(change**2,axis=1),1e-15),0.,1.)
    swept=float(np.linalg.norm(delta[:-1]+fraction[:,None]*change,axis=1).min()-radii)
    endpoint=int(np.argmin(np.abs(t-protocol['duration'])))
    fresh=abs(t[endpoint]-protocol['duration'])<=protocol['endpoint_freshness']
    ideal=np.asarray(protocol['robot_offsets'][0])-protocol['robot_offsets'][1]
    pair_error=float(np.linalg.norm(delta[endpoint]-ideal)) if fresh else None
    checks=dict(all_robot_motion_accepted=all(r['accepted'] for r in reports.values()),
        sampled_peer_proxy_clearance_positive=sample>0.,linear_swept_peer_proxy_clearance_positive=swept>0.,
        relative_formation_recovered=fresh and pair_error<=protocol['maximum_final_pair_error'])
    return dict(passed=bool(all(checks.values())),checks=checks,failed_checks=[k for k,v in checks.items() if not v],
        robots=reports,peer_proxy_clearance_min=sample,peer_linear_swept_proxy_clearance_min=swept,
        final_relative_formation_error=pair_error,endpoint_sample_time=float(t[endpoint]),protocol=protocol,
        note='Actual positions with circular proxies and linear interpolation, not continuous mesh safety proof')


def evaluate_team_run(directory,base_passed=None):
    path=Path(directory);meta=json.loads((path/'run_metadata.json').read_text());protocol=meta['active_protocol']
    if (meta['robot_names']!=['tb3_0','tb3_1'] or not meta.get('two_robot_experiment')
            or meta['trajectory_type']!='line' or protocol['version']!='two_robot_motion_v1_2026_11_12'):
        raise ValueError('Explicit frozen two-robot line protocol required')
    specs=json.loads((path/'dynamic_scenario.json').read_text())['specifications']
    if len(specs)!=1:raise ValueError('Two-robot protocol requires one actor')
    spec=specs[0];source=path/'gazebo_truth.csv'
    if not source.exists():source=path/'active_truth.csv'
    if not source.exists():return dict(passed=False,failed_checks=['missing_independent_truth'])
    rows=list(csv.DictReader(source.open()));t=[float(r['time']) for r in rows]
    positions=[[[float(r[name+'_x']),float(r[name+'_y'])] for r in rows] for name in meta['robot_names']]
    actor=[[float(r[spec['name']+'_x']),float(r[spec['name']+'_y'])] for r in rows]
    faults=[];lo,hi=protocol['motion_window']
    for name in meta['robot_names']:
        f=path/(name+'.csv')
        if not f.exists():faults.append(None);continue
        commands=list(csv.DictReader(f.open()))
        faults.append(sum(lo<=float(r.get('evaluation_time',r['time']))<=hi
            and r['status'] not in ('ok','feasible_iterate','validated_one_step') for r in commands))
    result=evaluate_team(t,positions,actor,spec['radius'],protocol,meta['dynamic_scenario'],faults)
    result.update(independent_truth_file=source.name,base_passed=base_passed,
        controller_records_finalized=(path/'RESULTS_SAVED').exists())
    if not result['controller_records_finalized']:
        result['passed']=False;result['failed_checks'].append('controller_records_unfinalized')
    if base_passed is not None:result['passed']=bool(result['passed'] and base_passed)
    return result
