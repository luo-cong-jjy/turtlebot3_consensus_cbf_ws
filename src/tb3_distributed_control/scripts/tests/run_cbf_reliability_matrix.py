#!/usr/bin/env python3
"""Predeclared, serial, seed-paired CV/CA/adaptive Gazebo comparison.

Seed controls Gazebo randomness, not asynchronous ROS or wall-budget IPOPT.
Failed algorithm trials are retained and never retried to obtain a good score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[41,73])
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    package=Path(__file__).resolve().parents[2]
    modes=['cv','ca','adaptive'];cases=[('accelerating_crossing',32),('decelerating_crossing',32),('stop_go_crossing',40)]
    schedule=[]
    for si,seed in enumerate(args.seeds):
        for ci,(case,duration) in enumerate(cases):
            rotation=(si+ci)%len(modes);order=modes[rotation:]+modes[:rotation]
            for mode in order:schedule.append(dict(seed=seed,scenario=case,mode=mode,duration=duration))
    # One predeclared repeated identical-seed baseline quantifies residual
    # asynchronous variability; it is not a replacement of a failed trial.
    schedule.append(dict(seed=args.seeds[0],scenario='accelerating_crossing',mode='cv',duration=32,repeat=2))
    paths=['scripts/tb3_controller_node_cbf.py','scripts/cbf_experiment_results.py','scripts/leader_node.py',
        'scripts/cbf_dynamic_scenario.py','scripts/cbf_mpc_core.py','纯仿真代码/mpc_dcbf_static_obstacles.py','worlds/consensus_dynamic_obstacles.world','纯仿真代码/mpc_dcbf_cached_solver.py','config/cbf_dynamic_scenarios.yaml',
        'launch/multi_turtlebot3_consensus_gazebo_cbf.launch','launch/single_turtlebot3_cbf_dynamic.launch',
        'scripts/tests/run_cbf_dynamic_gazebo.py','scripts/tests/run_cbf_reliability_matrix.py']
    sha={rel:hashlib.sha256((package/rel).read_bytes()).hexdigest() for rel in paths}
    revision=subprocess.check_output(['git','-C',str(package.parents[1]),'rev-parse','HEAD'],text=True).strip()
    protocol=dict(schedule=schedule,source_sha256=sha,source_commit=revision,
        parameters=dict(reliability_horizon=1.,score_tau=.5,weight_tau=.3,score_floor=.015,switch_ratio=.8,switch_hold=.6),
        evaluation='metrics schema 2, fixed mission endpoint, positive 20Hz proxy clearance, no truth input, actual gzserver seed check',
        policy='All scheduled trials retained; no parameter retuning; no algorithm-failure retries. The two seeds are exploratory repeats, not statistical superiority evidence.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'protocol.json').chmod(0o444)
    records=[]
    for index,spec in enumerate(schedule):
        for relative,digest in sha.items():
            if hashlib.sha256((package/relative).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Source changed after protocol freeze: '+relative)
        name='gazebo_%s_%s_seed%d'%(spec['scenario'],spec['mode'],spec['seed'])
        if spec.get('repeat'):name+='_repeat'+str(spec['repeat'])
        directory=out/name
        print('START %d/%d %s'%(index+1,len(schedule),name),flush=True)
        cmd=[sys.executable,str(package/'scripts/tests/run_cbf_dynamic_gazebo.py'),
            '--scenario',spec['scenario'],'--prediction-mode',spec['mode'],'--gazebo-seed',str(spec['seed']),
            '--duration',str(spec['duration']),'--output',str(directory)]
        with (out/(name+'_runner.log')).open('w') as stream:
            result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
        evaluation=json.loads((directory/'evaluation.json').read_text()) if (directory/'evaluation.json').exists() else {}
        record=dict(spec,directory=name,returncode=result.returncode,passed=evaluation.get('passed',False),
            failure=evaluation.get('failure'),seed_configured=evaluation.get('seed_configured'),
            clearance=evaluation.get('dynamic_clearance_min'),statistics=evaluation.get('robot_statistics',{}))
        records.append(record);(out/'runs.json').write_text(json.dumps(records,indent=2))
        print('FINISH',name,'passed=',record['passed'],'clearance=',record['clearance'],flush=True)
    print('COMPLETED',len(records),'trials, passed',sum(r['passed'] for r in records),flush=True)


if __name__=='__main__':main()
