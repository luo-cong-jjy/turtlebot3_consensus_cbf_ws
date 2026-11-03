#!/usr/bin/env python3
"""Replay logged reference decisions without obstacle truth or scenario scripts."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

PACKAGE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PACKAGE/'纯仿真代码'))
from mpc_dcbf_cached_solver import progress_preview,rollout,barrier_values

FROZEN_MODULES={}


def audit(directory):
    path=Path(directory);metadata=json.loads((path/'run_metadata.json').read_text())
    if not metadata.get('progress_guidance',False):return dict(applicable=False)
    hashes=json.loads((path/'sources/sha256.json').read_text())
    digest=hashes['纯仿真代码/mpc_dcbf_cached_solver.py'];snapshot=path/'sources/mpc_dcbf_cached_solver.py'
    if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=digest:raise ValueError('Saved source does not match runtime hash')
    if digest not in FROZEN_MODULES:
        spec=importlib.util.spec_from_file_location('_progress_frozen_'+digest[:12],snapshot)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        FROZEN_MODULES[digest]=module
    frozen=FROZEN_MODULES[digest]
    decisions=[]
    for line in (path/'tb3_0_solver_trace.jsonl').open():
        record=json.loads(line);decision=record.get('manager',{}).get('progress_decision')
        if decision is not None:decisions.append(decision)
    maximum=0.;selected_valid=0;fallbacks=0;error=None;preview_count=0
    for decision in decisions:
        try:
            state=np.array(decision['state']);goals=np.array(decision['goals']);cfg=decision['config']
            obstacles=decision['obstacles'];index=decision['selected_obstacle'];previous=np.array(decision['previous']);bounds=decision['bounds']
            previews={}
            for mode,recorded in decision['previews'].items():
                result=frozen.progress_preview(state,goals,obstacles,index,mode,cfg,previous,bounds);previews[mode]=result
                for key in ('cost','min_h','min_slack','terminal_error','estimated_recovery'):
                    delta=abs(result[key]-recorded[key]);maximum=max(maximum,delta)
                    np.testing.assert_allclose(result[key],recorded[key],rtol=0,atol=1e-9)
                assert result['feasible']==recorded['feasible']
                assert result['release_time']==recorded['release_time']
                np.testing.assert_allclose(result['states'],rollout(state,result['controls'],cfg['dt']),atol=1e-12)
                assert np.max(np.abs(result['controls'][:,1]))<=cfg['omega_max']+1e-12
                assert result['controls'][:,0].min()>=bounds[0]-1e-12 and result['controls'][:,0].max()<=bounds[1]+1e-12
                for hypothesis,value in recorded.get('hypothesis_min_h',{}).items():
                    np.testing.assert_allclose(value,result['hypothesis_min_h'][hypothesis],rtol=0,atol=1e-9)
                preview_count+=1
            valid=[p for p in previews.values() if p['feasible']]
            selected=min(valid,key=lambda p:p['cost'])['mode'] if valid else 'legacy'
            prior=decision.get('previous_decision')
            if prior is not None and prior['name']==decision['name']:
                incumbent=previews.get(prior['selected'])
                if incumbent is not None and incumbent['feasible'] and selected!=prior['selected']:
                    advantage=incumbent['cost']-previews[selected]['cost']
                    if decision['decision_time']-prior['switch_time']<.8 or advantage<max(.03,.15*incumbent['cost']):selected=prior['selected']
            if 'previous_decision' in decision:assert selected==decision['selected'],'Winner/hysteresis mismatch'
            else:selected=decision['selected']  # Development prototypes did not record prior hysteresis state.
            if selected=='legacy':fallbacks+=1
            else:
                assert previews[selected]['feasible'];selected_valid+=1
                states=previews[selected]['states'][:cfg['horizon']+1]
                original_h=barrier_values(states,obstacles,cfg['margin'])
                gammas=np.array([o.get('gamma',cfg['gamma']) for o in obstacles])[:,None]
                assert np.min(original_h[:,1:]-(1-gammas)*original_h[:,:-1])>=-2e-6
        except (AssertionError,ValueError,KeyError) as exc:
            error=dict(decision_id=decision['decision_id'],error=str(exc));break
    result=dict(applicable=True,passed=error is None and bool(decisions),source_sha256=digest,decisions=len(decisions),
        winner_checks=sum('previous_decision' in d for d in decisions),
        previews_reconstructed=preview_count,selected_valid=selected_valid,no_valid_previews=fallbacks,
        legacy_fallbacks=fallbacks if not metadata.get('progress_abstain',False) else 0,
        reference_withdrawals=fallbacks if metadata.get('progress_abstain',False) else 0,
        maximum_numeric_difference=maximum,first_error=error,
        note='Only logged decision inputs are replayed. Checks preview kinematics, original-window DCBF, retained hypotheses, scores and hysteresis. Does not prove actual commands follow previews or continuous physical safety.')
    (path/'progress_audit.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    result=audit(args.directory);print(json.dumps(result,indent=2))
    if result.get('passed') is False:raise SystemExit(1)
