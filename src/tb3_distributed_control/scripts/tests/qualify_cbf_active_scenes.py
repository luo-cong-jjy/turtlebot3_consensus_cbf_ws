#!/usr/bin/env python3
"""Freeze/qualify scenarios before controller trials, without supplying a path.

Constructive quintic lane-change witness proves bounded geometric feasibility
under ideal geometry, not success of the real perception/controller pipeline.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import yaml
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
from evaluate_cbf_active_motion import evaluate_motion
from mpc_dcbf_cached_solver import rollout


def qualify():
    scene_file=PACKAGE/'config/cbf_active_avoidance_scenarios.yaml';protocol_file=PACKAGE/'config/cbf_active_avoidance_protocol.yaml'
    scenes=yaml.safe_load(scene_file.read_text());protocol=yaml.safe_load(protocol_file.read_text())
    source=PACKAGE/'scripts/cbf_dynamic_scenario.py';node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='trajectory')
    scope={'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    dt=.1;t=np.arange(round(protocol['duration']/dt)+1)*dt;w=protocol['witness']
    def smooth(s):s=np.clip(s,0.,1.);return 10*s**3-15*s**4+6*s**5
    lateral=w['lateral_offset']*(smooth((t-w['start_time'])/w['outward_duration'])-
        smooth((t-w['start_time']-w['outward_duration']-w['hold_duration'])/w['return_duration']))
    nominal=np.asarray(protocol['reference_start'])+t[:,None]*np.asarray(protocol['reference_velocity'])
    points=nominal.copy();points[:,1]+=lateral;diff=np.diff(points,axis=0)
    heading=np.r_[0.,np.unwrap(np.arctan2(diff[:,1],diff[:,0]))]
    controls=np.column_stack((np.linalg.norm(diff,axis=1)/dt,np.diff(heading)/dt))
    states=rollout(np.r_[points[0],0.],controls,dt)
    np.testing.assert_allclose(states[:,:2],points,atol=1e-11)
    witness_bounds=bool(np.abs(controls[:,0]).max()<=protocol['maximum_speed'] and np.abs(controls[:,1]).max()<=protocol['maximum_yaw_rate'])
    results={}
    for name,specs in scenes.items():
        spec=specs[0];values=np.array([scope['trajectory'](spec,v) for v in t]);actor=values[:,:2]
        motion=evaluate_motion(t,points,actor,spec['radius'],protocol)
        distances=np.linalg.norm(nominal-actor,axis=1)-protocol['body_radius']-spec['radius']
        hit=np.flatnonzero(distances<=0.);stopped=np.tile(protocol['reference_start'],(len(t),1))
        stopped_result=evaluate_motion(t,stopped,actor,spec['radius'],protocol)
        model_h=np.linalg.norm(points-actor,axis=1)-spec['radius']-(.02+.35*np.linalg.norm(values[:,2:],axis=1))-.12
        residual=model_h[1:]-.7*model_h[:-1]
        item=dict(specification=spec,initial_detection_distance=float(np.linalg.norm(points[0]-actor[0])),
            first_nominal_proxy_collision=float(t[hit[0]]) if len(hit) else None,
            witness=motion,witness_speed_max=float(controls[:,0].max()),witness_speed_min=float(controls[:,0].min()),
            witness_yaw_rate_max=float(np.abs(controls[:,1]).max()),ideal_geometry_dcbf_residual_min=float(residual.min()),
            stopped_baseline=stopped_result)
        if name=='active_overtaking':
            following=nominal.copy();following[:,0]=np.minimum(nominal[:,0],actor[:,0]-protocol['body_radius']-spec['radius']-.05)
            item['slow_following_baseline']=evaluate_motion(t,following,actor,spec['radius'],protocol)
        item['qualified']=bool(len(hit) and witness_bounds and motion['passed'] and not stopped_result['passed'] and residual.min()>=-2e-6
            and item['initial_detection_distance']<=3.5)
        results[name]=item
    return dict(protocol=protocol,scenes=results,qualified=all(v['qualified'] for v in results.values()),
        sources={str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [scene_file,protocol_file,source]},
        scope=__doc__,witness_times=t.tolist(),witness_positions=points.tolist(),witness_controls=controls.tolist())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);result=qualify();out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:{m:v[m] for m in ['qualified','first_nominal_proxy_collision','witness_speed_max','witness_yaw_rate_max','ideal_geometry_dcbf_residual_min']} for k,v in result['scenes'].items()},indent=2))
    raise SystemExit(0 if result['qualified'] else 1)
