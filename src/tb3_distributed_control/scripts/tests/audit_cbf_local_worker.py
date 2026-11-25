#!/usr/bin/env python3
"""Revalidate saved current candidates and link worker inputs to issued commands.

This certifies only the original sampled numerical model on recorded inputs.
It does not certify physical obstacles, all unsaved shutdown races or optimality.
"""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
P=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P/'scripts'),str(P/'纯仿真代码')]
from mpc_dcbf_cached_solver import CachedMPCProblem,delay_compensated_request


def audit(directory):
    root=Path(directory);errors=[];full=short=missing=worker_inputs=0;maximum=0.;decisions={}
    def compare(a,b,label,robot,cycle):
        nonlocal maximum
        a,b=np.asarray(a,float),np.asarray(b,float)
        error=float(np.max(np.abs(a-b))) if a.shape==b.shape and a.size else (0. if a.shape==b.shape else float('inf'))
        maximum=max(maximum,error)
        if not np.isfinite(error) or error>1e-9:errors.append(dict(robot=robot,cycle=cycle,error=label,difference=error))
    for file in root.glob('tb3_*_solver_trace.jsonl'):
        robot=file.name.replace('_solver_trace.jsonl','');events={r['cycle_id']:r for r in map(json.loads,file.open())}
        rows={int(r['cycle_id']):r for r in csv.DictReader((root/(robot+'.csv')).open())};seen=set()
        for cycle,event in events.items():
            record=event['manager'];local=record.get('local_worker')
            if local is None:continue
            decision=local['decision'];decisions[decision]=decisions.get(decision,0)+1
            saved=record.get('input_snapshot');candidate=record.get('current_candidate')
            adopted=event['status'] in ('ok','feasible_iterate','validated_one_step')
            if adopted and candidate is not None:
                cfg=saved['config'];problem=CachedMPCProblem.__new__(CachedMPCProblem)
                problem.config=cfg;problem.dt=cfg['dt'];problem.constraint_mode=cfg.get('constraint_mode','dcbf')
                current={k:np.asarray(candidate[k],float) for k in ('states','controls','plan_controls')}
                one=candidate['validated_steps']==1
                if candidate['validated_steps'] not in (1,cfg['horizon']):errors.append(dict(robot=robot,cycle=cycle,error='Unrecognised validation scope'))
                if not problem.validate(current,np.asarray(saved['state']),saved['obstacles'],saved['bounds'],np.asarray(saved['previous']),one_step=one):
                    errors.append(dict(robot=robot,cycle=cycle,error='Current candidate fails original validator'))
                compare(event['command'],current['controls'][0],'Applied command differs from checked candidate',robot,cycle)
                if one:short+=1
                else:full+=1
            elif adopted:missing+=1
            old=local.get('record',{});sequence=local.get('sequence')
            if not old.get('input_snapshot') or sequence in seen:continue
            seen.add(sequence)
            if sequence not in events or sequence>=cycle:
                errors.append(dict(robot=robot,cycle=cycle,error='Worker input has no earlier recorded source'));continue
            source=events[sequence];source_input=source['manager']['input_snapshot'];cfg=source_input['config']
            stamp=float(rows[sequence]['ros_time'])
            task=dict(state=np.asarray(source_input['state']),goals=np.asarray(source_input['goals']),
                obstacles=source_input['obstacles'],previous=np.asarray(source_input['previous']),stamp=stamp)
            if cfg.get('worker_delay_compensation',False):
                task=delay_compensated_request(task,np.asarray(source['command']),cfg)
            expected=old['input_snapshot'];worker_inputs+=1
            for k in ('state','goals','previous'):compare(task[k],expected[k],'Worker '+k+' not linked to source command',robot,cycle)
            for a,b in zip(task['obstacles'],expected['obstacles']):
                if a['name']!=b['name']:errors.append(dict(robot=robot,cycle=cycle,error='Worker obstacle identity mismatch'))
                for k in ('center','a','b','theta','prediction_centers','prediction_axes','prediction_angles'):
                    if k in a and k in b:compare(a[k],b[k],'Worker obstacle '+k,robot,cycle)
    return dict(passed=not errors and worker_inputs>0,full_current_candidates=full,
        first_step_only_candidates=short,current_candidates_unrecorded=missing,
        linked_worker_inputs=worker_inputs,maximum_input_difference=maximum,decisions=decisions,
        error_count=len(errors),first_error=errors[0] if errors else None,scope=__doc__)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(audit(args.directory),indent=2))
