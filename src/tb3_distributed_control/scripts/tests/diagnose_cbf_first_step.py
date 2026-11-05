#!/usr/bin/env python3
"""Sufficient model-infeasibility certificate, not a physical safety test.

For fixed stage-1 axes, the smoothed ellipse h is L-Lipschitz in position,
L=b/min(a,b). Any bounded unicycle command displaces at most dt*|v|max.
If h1(hold)+L*radius is below the DCBF requirement minus tolerance, even
the larger unconstrained displacement ball cannot satisfy this barrier.
No certificate means undecided, not feasible. Only logged decisions replay.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

PACKAGE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PACKAGE/'纯仿真代码'))
from mpc_dcbf_cached_solver import barrier_values
from mpc_dcbf_static_obstacles import obstacle_axes


def certify(state,obstacles,bounds,config,tolerance=2e-6):
    state=np.asarray(state,float)
    if state.shape!=(3,) or not np.isfinite(state).all() or not np.isfinite(bounds).all() or config['dt']<=0:
        raise ValueError('Expected a finite state, input bounds and positive dt')
    radius=config['dt']*(max(abs(float(v)) for v in bounds)+tolerance)
    values=barrier_values(np.tile(state,(2,1)),obstacles,config['margin'])
    if not np.isfinite(values).all():raise ValueError('Nonfinite barrier input')
    certificates=[]
    for i,o in enumerate(obstacles):
        a,b=obstacle_axes(o,1)
        if not np.isfinite([a,b]).all() or min(a,b)<=0:raise ValueError('Expected positive finite axes')
        lipschitz=max(1.,b/a);upper=values[i,1]+lipschitz*radius
        required=(1-o.get('gamma',config['gamma']))*values[i,0]
        if upper<required-tolerance:
            certificates.append(dict(obstacle=o.get('name',str(i)),h0=float(values[i,0]),h1_hold=float(values[i,1]),
                lipschitz=lipschitz,displacement_radius=radius,reachable_upper=float(upper),required_h1=float(required),gap=float(upper-required)))
    return dict(certified_model_infeasible=bool(certificates),certificates=certificates,
        no_certificate_means='undecided',tolerance=tolerance)


def diagnose(directory):
    p=Path(directory);frames=[]
    for f in sorted(p.glob('tb3_*_solver_trace.jsonl')):
        for r in map(json.loads,f.open()):
            d=r.get('manager',{}).get('progress_decision')
            if d is None:continue
            result=certify(d['state'],d['obstacles'],d['bounds'],d['config'])
            frames.append(dict(robot=f.stem.split('_solver')[0],time=r['trajectory_time'],decision_id=d['decision_id'],**result))
    return dict(decisions_examined=len(frames),certified_frames=sum(r['certified_model_infeasible'] for r in frames),frames=frames,
        note='Sufficient infeasibility of frozen model first-step constraints only. No truth input; all robots with logged decisions. Absence of certificate is undecided. Does not establish physical impossibility, collision or safety.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError('Keep previous diagnostics; choose a new output')
    result=diagnose(args.directory);output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='frames'},indent=2))
