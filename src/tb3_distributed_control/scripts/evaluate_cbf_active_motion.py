#!/usr/bin/env python3
"""Independent active-avoidance acceptance; stopping/time-yield is not success.

Truth is read ONLY for evaluation. Speed thresholds are test criteria, not
positive lower bounds added to the controller. Linear swept clearances are a
sampled interpolation proxy, not mesh contact or continuous safety proof.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import yaml


def evaluate_motion(times,ego,actor,actor_radius,protocol,fault_stop_count=0):
    t=np.asarray(times,float);p=np.asarray(ego,float);o=np.asarray(actor,float)
    if len(t)<3 or p.shape!=(len(t),2) or o.shape!=p.shape or not all(np.isfinite(v).all() for v in (t,p,o)):
        return dict(passed=False,failed_checks=['insufficient_or_invalid_truth'])
    if not np.all(np.diff(t)>0):return dict(passed=False,failed_checks=['nonmonotonic_truth'])
    start,end=protocol['motion_window'];duration=float(protocol['duration']);delta=np.diff(t)
    movement=np.linalg.norm(np.diff(p,axis=0),axis=1);speed=movement/delta
    overlap=np.maximum(0.,np.minimum(t[1:],end)-np.maximum(t[:-1],start))
    selected=overlap>0;valid=delta<=protocol['maximum_truth_gap']
    covered=bool(t[0]<=start and t[-1]>=end and valid[selected].all())
    stopped=selected&valid&(speed<protocol['stop_speed']);total_stop=float(overlap[stopped].sum())
    longest=current=0.
    for i in range(len(delta)):
        current=current+overlap[i] if stopped[i] else 0.;longest=max(longest,current)
    cumulative=np.r_[0.,np.cumsum(movement)];window=float(protocol['rolling_window'])
    grid=np.arange(start+window,end+1e-8,.1)
    rolling=(np.interp(grid,t,cumulative)-np.interp(grid-window,t,cumulative))/window
    minimum_rolling=float(rolling.min()) if len(rolling) and covered else None
    relative=p-o;combined=float(protocol['body_radius'])+actor_radius
    sample_min=float(np.linalg.norm(relative,axis=1).min()-combined)
    change=np.diff(relative,axis=0);fraction=np.clip(-np.sum(relative[:-1]*change,axis=1)/np.maximum(np.sum(change**2,axis=1),1e-15),0.,1.)
    swept=float(np.linalg.norm(relative[:-1]+fraction[:,None]*change,axis=1).min()-combined)
    plane=relative[:,0];crossings=np.flatnonzero((plane[:-1]<0)&(plane[1:]>=0));event=None
    for i in crossings:
        alpha=-plane[i]/(plane[i+1]-plane[i]);stamp=t[i]+alpha*delta[i]
        if start<=stamp<=end:
            point=p[i]+alpha*(p[i+1]-p[i]);event=dict(time=float(stamp),position=point.tolist(),
                lateral_offset=float(abs(point[1]-protocol['reference_start'][1])));break
    endpoint=int(np.argmin(np.abs(t-duration)));fresh=bool(abs(t[endpoint]-duration)<=protocol['endpoint_freshness'])
    reference=np.asarray(protocol['reference_start'])+duration*np.asarray(protocol['reference_velocity'])
    final_error=float(np.linalg.norm(p[endpoint]-reference)) if fresh else None
    checks=dict(truth_covers_motion_window=covered,task_endpoint_fresh=fresh,
        sampled_proxy_clearance_positive=sample_min>0.,linear_swept_proxy_clearance_positive=swept>0.,
        no_sustained_standstill=longest<=protocol['maximum_stop_episode'],
        total_standstill_within_measurement_tolerance=total_stop<=protocol['maximum_total_stop'],
        no_crawling_time_avoidance=minimum_rolling is not None and minimum_rolling>=protocol['minimum_rolling_speed'],
        passed_actor=event is not None,
        spatial_detour_observed=event is not None and event['lateral_offset']>=protocol['minimum_lateral_at_pass'],
        tracking_recovered=fresh and final_error<protocol['maximum_final_position_error'],
        no_fault_stop=fault_stop_count==0 if protocol.get('require_no_fault_stop',True) else True)
    checks={k:bool(v) for k,v in checks.items()}
    return dict(passed=bool(all(checks.values())),checks=checks,failed_checks=[k for k,v in checks.items() if not v],
        motion_window=[start,end],sampled_proxy_clearance_min=sample_min,linear_swept_proxy_clearance_min=swept,
        total_standstill_seconds=total_stop,longest_standstill_seconds=float(longest),minimum_rolling_speed=minimum_rolling,
        fault_stop_count=fault_stop_count,first_forward_pass=event,endpoint_sample_time=float(t[endpoint]),final_position_error=final_error,
        observed_max_planar_speed=float(speed[selected].max()) if selected.any() else None)


def evaluate_run(directory,protocol=None,base_passed=None):
    path=Path(directory);meta=json.loads((path/'run_metadata.json').read_text())
    protocol=protocol or meta.get('active_protocol')
    if not protocol:raise ValueError('No frozen active protocol supplied or recorded')
    specs=json.loads((path/'dynamic_scenario.json').read_text())['specifications']
    if len(specs)!=1 or len(meta['robot_names'])!=1 or meta['trajectory_type']!='line':raise ValueError('Single actor/robot and straight reference required')
    source=path/'gazebo_truth.csv'
    if not source.exists():source=path/'active_truth.csv'
    if not source.exists():return dict(passed=False,failed_checks=['missing_independent_truth'])
    rows=list(csv.DictReader(source.open()));name=specs[0]['name']
    t=[float(r['time']) for r in rows];p=[[float(r['tb3_0_x']),float(r['tb3_0_y'])] for r in rows]
    o=[[float(r[name+'_x']),float(r[name+'_y'])] for r in rows]
    control_file=path/'tb3_0.csv';faults=None
    if control_file.exists():
        commands=list(csv.DictReader(control_file.open()));lo,hi=protocol['motion_window']
        faults=sum(lo<=float(r.get('evaluation_time',r['time']))<=hi and r['status'] not in ('ok','feasible_iterate','validated_one_step') for r in commands)
    result=evaluate_motion(t,p,o,specs[0]['radius'],protocol,faults)
    result.update(protocol=protocol,source=str(path.resolve()),independent_truth_file=source.name,
        controller_records_finalized=(path/'RESULTS_SAVED').exists(),base_passed=base_passed)
    if not result['controller_records_finalized']:
        result['passed']=False;result['failed_checks'].append('controller_records_unfinalized')
    if base_passed is not None:result['passed']=bool(result['passed'] and base_passed)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--protocol');p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    protocol=yaml.safe_load(Path(a.protocol).read_text()) if a.protocol else None
    result=evaluate_run(a.directory,protocol);out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
