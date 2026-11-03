#!/usr/bin/env python3
"""Frozen reference-policy comparison; predictor and hard MPC remain fixed."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--seeds',type=int,nargs=2,default=[101,137]);args=parser.parse_args()
    if len(set(args.seeds))!=2 or any(s<0 for s in args.seeds):raise ValueError('Two distinct nonnegative seeds required')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    package=Path(__file__).resolve().parents[2];schedule=[]
    cases=[('accelerating_crossing',32),('decelerating_crossing',32),('stop_go_crossing',40)]
    for si,seed in enumerate(args.seeds):
        for ci,(case,duration) in enumerate(cases):
            policies=['legacy','progress'] if (si+ci)%2==0 else ['progress','legacy']
            for policy in policies:schedule.append(dict(seed=seed,scenario=case,duration=duration,policy=policy,role='primary'))
    for policy in ('legacy','progress'):
        schedule.append(dict(seed=args.seeds[0],scenario='accelerating_crossing',duration=32,policy=policy,role='repeat',repeat=2))
    for case in ('accelerating_crossing','decelerating_crossing'):
        schedule.append(dict(seed=args.seeds[0],scenario=case,duration=32,policy='mean_only',role='component_ablation'))
    for case in ('head_on','overtaking'):
        schedule.append(dict(seed=args.seeds[0],scenario=case,duration=40,policy='progress',role='transfer_regression'))
    paths=['scripts/tb3_controller_node_cbf.py','scripts/cbf_mpc_core.py','scripts/cbf_experiment_results.py',
        'scripts/leader_node.py','scripts/leader_node_cbf.py','scripts/cbf_dynamic_scenario.py',
        '纯仿真代码/mpc_dcbf_cached_solver.py','纯仿真代码/mpc_dcbf_static_obstacles.py',
        'config/cbf_dynamic_scenarios.yaml','worlds/consensus_dynamic_obstacles.world',
        'launch/multi_turtlebot3_consensus_gazebo_cbf.launch','launch/single_turtlebot3_cbf_dynamic.launch',
        'launch/multi_turtlebot3_consensus_gazebo_cbf_dynamic.launch',
        'scripts/tests/run_cbf_dynamic_gazebo.py','scripts/tests/run_cbf_progress_matrix.py']
    sha={p:hashlib.sha256((package/p).read_bytes()).hexdigest() for p in paths}
    revision=subprocess.check_output(['git','-C',str(package.parents[1]),'rev-parse','HEAD'],text=True).strip()
    protocol=dict(source_commit=revision,source_sha256=sha,schedule=schedule,prediction_mode='adaptive',
        policy_parameters=dict(preview_horizon=4.,wait_estimation_horizon=6.,decision_period=.5,
            minimum_switch_hold=.8,switch_relative=.15,switch_absolute=.03,motion_gate=.02),
        note='New explicit seeds; development used 41/73 only. Fixed predictor, goal, input bounds, CBFs, solver and mission durations. Two seeds are descriptive only. Repeats, one-seed ablations and transfer checks excluded from main means. No algorithm-failure retries or validation-set retuning.')
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
            '--progress-guidance','false' if spec['policy']=='legacy' else 'true',
            '--progress-hypotheses','false' if spec['policy']=='mean_only' else 'true',
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
