#!/usr/bin/env python3
"""Replay rear-candidate guidance from frozen source and recorded model inputs.

Recorded budget-abstain events skip the unfinished computation, preserving
the incumbent as online. Early snapshot schemas omitted the CV guidance flag;
restore its documented dynamic/inactive convention in a replay copy only.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
MODULES={}


def audit(directory):
    path=Path(directory);meta=json.loads((path/'run_metadata.json').read_text())
    if not (meta.get('rear_pass') or meta.get('configuration',{}).get('rear_pass')):return dict(applicable=False)
    source=path/'sources/mpc_dcbf_cached_solver.py'
    if source.exists():expected=json.loads((path/'sources/sha256.json').read_text())['纯仿真代码/mpc_dcbf_cached_solver.py']
    else:source=path/'mpc_dcbf_cached_solver.py';expected=meta['source_sha256'][source.name]
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=expected:raise ValueError('Runtime source hash mismatch')
    if digest not in MODULES:
        spec=importlib.util.spec_from_file_location('_rear_replay_'+digest[:12],source);module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module;spec.loader.exec_module(module);MODULES[digest]=module
    module=MODULES[digest];memory={};checked=prefixes=abstentions=normalized=0;maximum=0.;error=None;problems={}
    for line in (path/'tb3_0_solver_trace.jsonl').open():
        record=json.loads(line);manager=record.get('manager',{});snapshot=manager.get('input_snapshot')
        if snapshot is None:continue
        cfg=snapshot['config'];state=np.asarray(snapshot['state']);goals=np.asarray(snapshot['goals']);previous=np.asarray(snapshot['previous']);bounds=snapshot['bounds']
        obs=[dict(o) for o in snapshot['obstacles']]
        for o in obs:
            for key in ('center','prediction_centers','prediction_axes','prediction_angles','rear_prediction_centers','rear_prediction_axes'):
                if key in o:o[key]=np.asarray(o[key])
            if 'guidance' not in o:
                o['guidance']=False if o.get('name')=='inactive' or o.get('peer',False) else not o.get('dynamic',False);normalized+=1
        memory.update(progress_previous=previous,progress_bounds=bounds)
        try:
            if manager.get('rear_pass',{}).get('mode')=='budget_abstain':
                memory.pop('progress_plan',None);result=(np.zeros((cfg['horizon'],2)),0.);abstentions+=1
            else:
                result=module.continuous_guidance(state,goals,obs,cfg,memory)
                assert memory['rear_summary']['mode']==manager['rear_pass']['mode'],'Mode mismatch'
            expected=snapshot['guidance'];headings=result[2] if len(result)>2 else np.arctan2(result[0][:,1],result[0][:,0])
            differences=[np.max(np.abs(result[0]-np.asarray(expected['vectors']))),abs(result[1]-expected['level']),np.max(np.abs(headings-np.asarray(expected['headings'])))]
            maximum=max(maximum,*[float(x) for x in differences]);assert max(differences)<=1e-9,'Reference numerical mismatch'
            plan=memory.get('progress_plan')
            if plan is not None and result[1]>0:
                key=(len(obs),bool(cfg['delta_limit']))
                if key not in problems:problems[key]=module.CachedMPCProblem(cfg,*key)
                assert problems[key].validate(plan,state,obs,bounds,previous),'Prefix fails original validator';prefixes+=1
            checked+=1
        except (AssertionError,ValueError,KeyError) as exc:
            error=dict(cycle_id=record['cycle_id'],error=str(exc));break
    return dict(applicable=True,passed=error is None and checked>0,source_sha256=digest,checked_cycles=checked,
        validated_reference_prefixes=prefixes,recorded_budget_abstentions=abstentions,cv_guidance_fields_restored=normalized,
        maximum_numeric_difference=maximum,first_error=error,interpretation=__doc__)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(audit(a.directory),indent=2))
