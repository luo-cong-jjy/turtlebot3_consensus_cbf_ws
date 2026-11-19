#!/usr/bin/env python3
"""Independent five-robot motion/formation acceptance; no control publishers."""
import csv
import json
from pathlib import Path
import numpy as np
from evaluate_cbf_active_motion import evaluate_motion


def evaluate_five(times, positions, actor, actor_radius, protocol, scenario, fault_counts=(0, 0, 0, 0, 0)):
    t=np.asarray(times,float);p=np.asarray(positions,float);actor=np.asarray(actor,float)
    if (len(t)<3 or p.shape!=(5,len(t),2) or actor.shape!=(len(t),2)
            or not np.isfinite(p).all() or not np.isfinite(t).all() or not np.isfinite(actor).all()
            or not np.all(np.diff(t)>0)):
        return dict(passed=False,failed_checks=['invalid_team_truth'])
    required=protocol['spatial_required'][scenario]
    reports={}
    for i in range(5):
        local=dict(protocol,reference_start=(np.asarray(protocol['reference_start'])+protocol['robot_offsets'][i]).tolist())
        motion=evaluate_motion(t,p[i],actor,actor_radius,local,fault_counts[i])
        checks=dict(motion['checks'])
        # Preserve the observed behavior fact. An unthreatened robot need not
        # invent a lateral manoeuvre; roles were fixed before controller trials.
        observed=checks.pop('spatial_detour_observed')
        checks['spatial_detour_required_satisfied']=not required[i] or observed
        reports['tb3_'+str(i)]=dict(motion=motion,spatial_required=bool(required[i]),
            checks=checks,accepted=bool(all(checks.values())),failed_checks=[k for k,v in checks.items() if not v])
    endpoint=int(np.argmin(np.abs(t-protocol['duration'])))
    fresh=abs(t[endpoint]-protocol['duration'])<=protocol['endpoint_freshness']
    pairs={}
    for i in range(5):
        for j in range(i+1,5):
            delta=p[i]-p[j];radii=2*protocol['body_radius']
            sample=float(np.linalg.norm(delta,axis=1).min()-radii)
            change=np.diff(delta,axis=0)
            fraction=np.clip(-np.sum(delta[:-1]*change,axis=1)/np.maximum(np.sum(change**2,axis=1),1e-15),0.,1.)
            swept=float(np.linalg.norm(delta[:-1]+fraction[:,None]*change,axis=1).min()-radii)
            ideal=np.asarray(protocol['robot_offsets'][i])-protocol['robot_offsets'][j]
            error=np.linalg.norm(delta-ideal,axis=1)
            pairs[str(i)+'_'+str(j)]=dict(sampled_proxy_clearance_min=sample,linear_swept_proxy_clearance_min=swept,
                final_relative_formation_error=float(error[endpoint]) if fresh else None,
                maximum_relative_formation_error=float(error.max()))
    sample=min(v['sampled_proxy_clearance_min'] for v in pairs.values())
    swept=min(v['linear_swept_proxy_clearance_min'] for v in pairs.values())
    pair_error=max(v['final_relative_formation_error'] for v in pairs.values()) if fresh else None
    checks=dict(all_robot_motion_accepted=all(r['accepted'] for r in reports.values()),
        sampled_peer_proxy_clearance_positive=sample>0.,linear_swept_peer_proxy_clearance_positive=swept>0.,
        relative_formation_recovered=fresh and pair_error<=protocol['maximum_final_pair_error'])
    return dict(passed=bool(all(checks.values())),checks=checks,failed_checks=[k for k,v in checks.items() if not v],
        robots=reports,pairs=pairs,peer_proxy_clearance_min=sample,peer_linear_swept_proxy_clearance_min=swept,
        final_relative_formation_error=pair_error,endpoint_sample_time=float(t[endpoint]),protocol=protocol,
        note='Ten pairs, sampled/linearly swept circular proxies; temporary deformation allowed, no mesh-safety proof')


def validate_scene(protocol,spec,scenario):
    """Freeze physical size and threatened roles; independent evaluation only."""
    expected=.12 if protocol['version']=='five_robot_bin012_v1_2026_11_19' else .18
    if spec['radius']!=expected:
        raise ValueError('Actor radius disagrees with frozen protocol')
    offsets=np.asarray(protocol['robot_offsets']);delta=np.asarray(protocol['reference_start'])+offsets-np.asarray(spec['start'])
    velocity=np.asarray(protocol['reference_velocity'])-np.asarray(spec['velocity'])
    norm=float(velocity@velocity)
    closest=np.clip(-delta@velocity/max(norm,1e-15),0.,protocol['duration']) if norm else np.zeros(5)
    distance=np.linalg.norm(delta+closest[:,None]*velocity,axis=1)
    roles=(distance<protocol['body_radius']+expected).tolist()
    if roles!=protocol['spatial_required'][scenario]:
        raise ValueError('Spatial roles disagree with pretrial nominal CPA geometry')
    return dict(closest_times=closest.tolist(),closest_center_distances=distance.tolist(),spatial_required=roles)


def evaluate_five_run(directory,base_passed=None):
    path=Path(directory);meta=json.loads((path/'run_metadata.json').read_text());protocol=meta['active_protocol']
    if (meta['robot_names']!=['tb3_'+str(i) for i in range(5)] or not meta.get('five_robot_experiment')
            or meta['trajectory_type']!='line' or protocol['version'] not in ('five_robot_motion_v1_2026_11_19','five_robot_bin012_v1_2026_11_19')):
        raise ValueError('Explicit frozen five-robot line protocol required')
    expected=np.array([[.4*np.cos(2*np.pi*i/5),.4*np.sin(2*np.pi*i/5)] for i in range(5)])
    if meta.get('formation_radius')!=.4 or not np.allclose(protocol['robot_offsets'],expected,atol=1e-9):
        raise ValueError('Frozen radius=.4 pentagon required')
    specs=json.loads((path/'dynamic_scenario.json').read_text())['specifications']
    if len(specs)!=1:raise ValueError('Five-robot protocol requires one actor')
    spec=specs[0];validate_scene(protocol,spec,meta['dynamic_scenario'])
    source=path/'gazebo_truth.csv'
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
    result=evaluate_five(t,positions,actor,spec['radius'],protocol,meta['dynamic_scenario'],faults)
    result.update(independent_truth_file=source.name,base_passed=base_passed,
        controller_records_finalized=(path/'RESULTS_SAVED').exists())
    if not result['controller_records_finalized']:
        result['passed']=False;result['failed_checks'].append('controller_records_unfinalized')
    if base_passed is not None:result['passed']=bool(result['passed'] and base_passed)
    return result
