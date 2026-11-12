#!/usr/bin/env python3
"""Pre-trial two-robot geometric witness; never an input to the controller."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import yaml
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
from evaluate_cbf_team_motion import evaluate_team
from mpc_dcbf_cached_solver import rollout


def qualify():
    protocol_file=PACKAGE/'config/cbf_two_robot_protocol.yaml';scene_file=PACKAGE/'config/cbf_active_avoidance_scenarios.yaml'
    protocol=yaml.safe_load(protocol_file.read_text());scenes=yaml.safe_load(scene_file.read_text())
    dt=.1;t=np.arange(round(protocol['duration']/dt)+1)*dt;w=protocol['witness']
    def smooth(q):q=np.clip(q,0.,1.);return 10*q**3-15*q**4+6*q**5
    lateral=w['lateral_offset']*(smooth((t-w['start_time'])/w['outward_duration'])-
        smooth((t-w['start_time']-w['outward_duration']-w['hold_duration'])/w['return_duration']))
    leader=np.asarray(protocol['reference_start'])+t[:,None]*np.asarray(protocol['reference_velocity'])
    positions=np.array([leader+offset for offset in np.asarray(protocol['robot_offsets'])]);positions[:,:,1]+=lateral
    displacement=np.diff(positions[0],axis=0);heading=np.r_[0.,np.unwrap(np.arctan2(displacement[:,1],displacement[:,0]))]
    controls=np.column_stack((np.linalg.norm(displacement,axis=1)/dt,np.diff(heading)/dt))
    for path in positions:np.testing.assert_allclose(rollout(np.r_[path[0],0.],controls,dt)[:,:2],path,atol=1e-11)
    bounded=bool(abs(controls[:,0]).max()<=protocol['maximum_speed'] and abs(controls[:,1]).max()<=protocol['maximum_yaw_rate'])
    results={}
    for name,specs in scenes.items():
        spec=specs[0];velocity=np.array(spec['velocity']);actor=np.array(spec['start'])+t[:,None]*velocity
        result=evaluate_team(t,positions,actor,spec['radius'],protocol,name)
        nominal=positions.copy();nominal[:,:,1]-=lateral
        nominal_min=np.linalg.norm(nominal-actor[None],axis=2).min(axis=1)-protocol['body_radius']-spec['radius']
        roles=[bool(v<=0.) for v in nominal_min]
        h=np.linalg.norm(positions-actor[None],axis=2)-spec['radius']-.02-.35*np.linalg.norm(velocity)-.12
        residual=h[:,1:]-.7*h[:,:-1]
        results[name]=dict(qualified=bool(bounded and result['passed'] and roles==protocol['spatial_required'][name]
            and residual.min()>=-2e-6),witness_report=result,nominal_collision_roles=roles,
            nominal_min_proxy_clearance=nominal_min.tolist(),ideal_dcbf_residual_min=float(residual.min()))
    return dict(qualified=all(r['qualified'] for r in results.values()),scenes=results,protocol=protocol,
        maximum_speed=float(abs(controls[:,0]).max()),maximum_yaw_rate=float(abs(controls[:,1]).max()),
        times=t.tolist(),positions=positions.tolist(),controls=controls.tolist(),
        sources={str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [protocol_file,scene_file]},
        scope='Ideal constructive qualification only; shared lateral witness is not communicated or commanded')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);a=parser.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    result=qualify();out.write_text(json.dumps(result,indent=2,allow_nan=False));print('Qualified:',result['qualified'])
    print({name:(r['qualified'],r['nominal_collision_roles'],r['ideal_dcbf_residual_min']) for name,r in result['scenes'].items()})
    raise SystemExit(0 if result['qualified'] else 1)
