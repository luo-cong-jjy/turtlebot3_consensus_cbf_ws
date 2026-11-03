#!/usr/bin/env python3
"""Separate post-counterexample validation: withdraw invalid soft guidance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--seeds',type=int,nargs=2,default=[211,239]);args=parser.parse_args()
    if len(set(args.seeds))!=2 or any(s<0 for s in args.seeds):raise ValueError('Two distinct nonnegative seeds required')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    package=Path(__file__).resolve().parents[2];schedule=[]
    cases=[('accelerating_crossing',32),('decelerating_crossing',32),('stop_go_crossing',40)]
    for si,seed in enumerate(args.seeds):
        for ci,(case,duration) in enumerate(cases):
            policies=['v3','abstain'] if (si+ci)%2==0 else ['abstain','v3']
            for policy in policies:schedule.append(dict(seed=seed,scenario=case,duration=duration,policy=policy,role='primary'))
    for policy in ('v3','abstain'):
        schedule.append(dict(seed=args.seeds[0],scenario='accelerating_crossing',duration=32,policy=policy,role='repeat',repeat=2))
    for case in ('head_on','overtaking'):
        schedule.append(dict(seed=args.seeds[0],scenario=case,duration=40,policy='abstain',role='transfer_regression'))
    paths=['scripts/tb3_controller_node_cbf.py','scripts/cbf_mpc_core.py','scripts/cbf_experiment_results.py',
        'scripts/leader_node.py','scripts/leader_node_cbf.py','scripts/cbf_dynamic_scenario.py',
        '纯仿真代码/mpc_dcbf_cached_solver.py','纯仿真代码/mpc_dcbf_static_obstacles.py',
        'config/cbf_dynamic_scenarios.yaml','worlds/consensus_dynamic_obstacles.world',
        'launch/multi_turtlebot3_consensus_gazebo_cbf.launch','launch/single_turtlebot3_cbf_dynamic.launch',
        'launch/multi_turtlebot3_consensus_gazebo_cbf_dynamic.launch',
        'scripts/tests/run_cbf_dynamic_gazebo.py','scripts/tests/run_cbf_abstention_matrix.py']
    sha={p:hashlib.sha256((package/p).read_bytes()).hexdigest() for p in paths}
    revision=subprocess.check_output(['git','-C',str(package.parents[1]),'rev-parse','HEAD'],text=True).strip()
    protocol=dict(source_commit=revision,source_sha256=sha,schedule=schedule,prediction_mode='adaptive',
        note='Separate cohort after V3 counterexample. 101/137 are now development data; new explicit seeds 211/239. Both policies keep the same predictor, preview generation, hard MPC and solver. Only invalid-reference fallback differs: old guide versus no extra soft guide. No validation-set retuning or algorithm-failure retries. Repeats/transfer tests separate from means.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'protocol.json').chmod(0o444)
    records=[]
    for i,spec in enumerate(schedule):
        for relative,digest in sha.items():
            if hashlib.sha256((package/relative).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen source changed: '+relative)
        name='gazebo_%s_%s_seed%d'%(spec['scenario'],spec['policy'],spec['seed'])
        if spec.get('repeat'):name+='_repeat'+str(spec['repeat'])
        path=out/name
        command=[sys.executable,str(package/'scripts/tests/run_cbf_dynamic_gazebo.py'),
            '--scenario',spec['scenario'],'--prediction-mode','adaptive','--gazebo-seed',str(spec['seed']),
            '--progress-guidance','true','--progress-hypotheses','true',
            '--progress-abstain','false' if spec['policy']=='v3' else 'true',
            '--duration',str(spec['duration']),'--output',str(path)]
        print('START %d/%d %s'%(i+1,len(schedule),name),flush=True)
        with (out/(name+'_runner.log')).open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        evaluation=json.loads((path/'evaluation.json').read_text()) if (path/'evaluation.json').exists() else {}
        records.append(dict(spec,directory=name,returncode=result.returncode,passed=evaluation.get('passed',False),
            failure=evaluation.get('failure'),clearance=evaluation.get('dynamic_clearance_min'),
            seed_configured=evaluation.get('seed_configured'),progress_configured=evaluation.get('progress_configured'),
            statistics=evaluation.get('robot_statistics',{})))
        (out/'runs.json').write_text(json.dumps(records,indent=2))
        print('FINISH',name,'passed=',records[-1]['passed'],'clearance=',records[-1]['clearance'],flush=True)
    print('COMPLETED',len(records),'trials',flush=True)


if __name__=='__main__':main()
