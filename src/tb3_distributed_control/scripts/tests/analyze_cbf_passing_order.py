#!/usr/bin/env python3
"""Offline actual crossing order for the single straight-reference scenarios.

Uses recorded ideal poses or independent Gazebo model poses. This does not
feed truth to the controller or infer passing order from a desired guide.
"""
import argparse
import ast
import csv
import json
import math
from pathlib import Path
import numpy as np
import yaml


def analyze(directory):
    path=Path(directory);meta=json.loads((path/'run_metadata.json').read_text())
    ideal=meta.get('simulator')=='ideal_discrete_unicycle'
    if ideal:
        root=Path(__file__).resolve().parents[2]
        spec_path=path/'cbf_dynamic_scenarios.yaml'
        if not spec_path.exists():spec_path=root/'config/cbf_dynamic_scenarios.yaml'
        spec=yaml.safe_load(spec_path.read_text())[meta['scenario']][0]
        source=path/'cbf_dynamic_scenario.py';tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='trajectory')
        scope={'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
        rows=list(csv.DictReader((path/'tb3_0.csv').open()));t=np.array([float(r['time']) for r in rows])
        ego=np.array([[float(r['x']),float(r['y'])] for r in rows]);actor=np.array([scope['trajectory'](spec,v)[:2] for v in t])
    else:
        specs=json.loads((path/'dynamic_scenario.json').read_text())['specifications']
        if len(specs)!=1 or len(meta['robot_names'])!=1 or meta['trajectory_type']!='line':raise ValueError('Single actor / robot straight-reference diagnostic only')
        spec=specs[0];name=spec['name'];rows=list(csv.DictReader((path/'gazebo_truth.csv').open()))
        t=np.array([float(r['time']) for r in rows]);ego=np.array([[float(r['tb3_0_x']),float(r['tb3_0_y'])] for r in rows])
        actor=np.array([[float(r[name+'_x']),float(r[name+'_y'])] for r in rows])
    relative_x=ego[:,0]-actor[:,0];indices=np.flatnonzero((relative_x[:-1]<0)&(relative_x[1:]>=0));events=[]
    for i in indices:
        fraction=-relative_x[i]/(relative_x[i+1]-relative_x[i]);stamp=t[i]+fraction*(t[i+1]-t[i])
        delta=(ego[i]-actor[i])+fraction*((ego[i+1]-actor[i+1])-(ego[i]-actor[i]))
        lo=max(0,i-3);hi=min(len(t)-1,i+4);velocity=(actor[hi]-actor[lo])/max(t[hi]-t[lo],1e-9)
        norm=np.linalg.norm(velocity)
        signed=float(np.dot(delta,velocity/norm)) if norm>.005 else None
        events.append(dict(time=float(stamp),relative_position=delta.tolist(),signed_motion_projection=signed,
            order='stationary_or_unknown' if signed is None else 'rear' if signed<0 else 'front',
            center_distance=float(np.linalg.norm(delta)),proxy_clearance=float(np.linalg.norm(delta)-.13-spec['radius'])))
    return dict(source=str(path.resolve()),input_kind='ideal' if ideal else 'independent_gazebo',forward_crossings=events,
        first_order=events[0]['order'] if events else 'no_forward_crossing',interpretation=__doc__)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(analyze(a.directory),indent=2,allow_nan=False))
