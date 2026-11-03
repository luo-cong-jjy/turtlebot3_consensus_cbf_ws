#!/usr/bin/env python3
"""Compare frozen reference policies; keep repeats and ablations separate."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    base=Path(args.directory);runs=[]
    for spec in json.loads((base/'runs.json').read_text()):
        p=base/spec['directory'];eval_path=p/'evaluation.json'
        evaluation=json.loads(eval_path.read_text()) if eval_path.exists() else {}
        data=json.loads((p/'experiment_statistics.json').read_text()).get('0',{}) if (p/'experiment_statistics.json').exists() else {}
        rows=list(csv.DictReader((p/'tb3_0.csv').open())) if (p/'tb3_0.csv').exists() else []
        result=dict(spec,nominal=data.get('position_error_summary'),heading=data.get('heading_error_summary'),
            statuses=data.get('statuses'),timing_max_ms=data.get('max_computation_ms'),overruns=data.get('overruns'),
            timing_p95_ms=evaluation.get('robot_statistics',{}).get('0',{}).get('timing_p95_ms'),
            no_truth_input=evaluation.get('no_truth_input_passed'),settling_time_005=None,
            preview_decisions=0,preview_valid=0,no_valid_preview=0,decision_ms_max=None)
        if rows:
            result.update(v_tv=float(np.abs(np.diff([float(r['v']) for r in rows])).sum()),
                omega_tv=float(np.abs(np.diff([float(r['omega']) for r in rows])).sum()),
                mode_cycles={str(code):sum(int(float(r['progress_mode']))==code for r in rows) for code in range(6)})
            window=[];last=-np.inf
            for r in rows:
                t=float(r['evaluation_time'])
                if 0<=t<=spec['duration'] and t>last+1e-9 and abs(float(r['odom_age']))<=.6:
                    window.append((t,np.hypot(float(r['consensus_error_x']),float(r['consensus_error_y']))));last=t
            if window:
                t,e=np.asarray(window).T;peak=int(np.nanargmax(e));bad=np.flatnonzero((e>.05)|~np.isfinite(e))
                start=max(peak,int(bad[-1])+1 if len(bad) else 0)
                if start<len(t) and t[-1]-t[start]>=2. and np.diff(t[start:]).max(initial=0.)<=.15+1e-9:
                    result['settling_time_005']=float(t[start])
        trace=p/'tb3_0_solver_trace.jsonl'
        if trace.exists():
            decisions=[r['manager']['progress_decision'] for r in map(json.loads,trace.open()) if 'progress_decision' in r.get('manager',{})]
            result['preview_decisions']=len(decisions)
            result['preview_valid']=sum(d['selected']!='legacy' for d in decisions)
            result['no_valid_preview']=sum(d['selected']=='legacy' for d in decisions)
            result['decision_ms_max']=max((d['decision_ms'] for d in decisions),default=None)
        runs.append(result)
    primary=[r for r in runs if r['role']=='primary'];groups=[];paired=[]
    cases=list(dict.fromkeys(r['scenario'] for r in primary));policies=list(dict.fromkeys(r['policy'] for r in primary))
    for case in cases:
        for policy in policies:
            group=[r for r in primary if (r['scenario'],r['policy'])==(case,policy)]
            rmse=[r['nominal']['rmse'] for r in group if r['nominal'] is not None]
            clear=[r['clearance'] for r in group if r['clearance'] is not None]
            groups.append(dict(scenario=case,policy=policy,trials=len(group),passed=sum(r['passed'] for r in group),
                mean_rmse=float(np.mean(rmse)) if rmse else None,rmse_range=[min(rmse),max(rmse)] if rmse else None,
                minimum_clearance=min(clear) if clear else None,
                mean_omega_tv=float(np.mean([r['omega_tv'] for r in group if 'omega_tv' in r])) if rmse else None,
                recovery_observed=sum(r['settling_time_005'] is not None for r in group)))
        for seed in sorted({r['seed'] for r in primary}):
            a,b=[next(r for r in primary if (r['scenario'],r['policy'],r['seed'])==(case,policy,seed)) for policy in policies]
            if a['nominal'] is not None and b['nominal'] is not None:
                paired.append(dict(scenario=case,seed=seed,comparison=policies[1]+' - '+policies[0],
                    rmse_difference=b['nominal']['rmse']-a['nominal']['rmse'],clearance_difference=b['clearance']-a['clearance']))
    (base/'progress_comparison.json').write_text(json.dumps(dict(runs=runs,groups=groups,paired_differences=paired,
        note='Main means use only primary trials. Same-seed repeats, single-seed component ablations and transfer checks remain separate. Two seeds are descriptive; all failures retained. Missing recovery is null. Heading/input variation and runtime are reported alongside position.'),indent=2))
    colors={'legacy':'tab:blue','v3':'tab:orange','progress':'tab:orange','abstain':'tab:purple'}
    fig,axes=plt.subplots(2,len(cases),figsize=(5*len(cases),7),squeeze=False)
    for col,case in enumerate(cases):
        for r in [r for r in primary if r['scenario']==case and r['nominal'] is not None]:
            p=base/r['directory'];data=list(csv.DictReader((p/'tb3_0.csv').open()))
            label=r['policy']+' seed '+str(r['seed'])+(' failed' if not r['passed'] else '')
            style='-' if r['seed']==min(a['seed'] for a in primary) else '--'
            axes[0,col].plot([float(a['evaluation_time']) for a in data],
                [np.hypot(float(a['consensus_error_x']),float(a['consensus_error_y'])) for a in data],
                color=colors[r['policy']],ls=style,lw=1.6,label=label)
            truth=list(csv.DictReader((p/'gazebo_truth.csv').open()));truth=[a for a in truth if 0<=float(a['time'])<=r['duration']]
            clear=[np.hypot(float(a['tb3_0_x'])-float(a['cbf_dynamic_0_x']),float(a['tb3_0_y'])-float(a['cbf_dynamic_0_y']))-.31 for a in truth]
            axes[1,col].plot([float(a['time']) for a in truth],clear,color=colors[r['policy']],ls=style,lw=1.6,label=label)
        axes[0,col].set_title(case.replace('_',' '));axes[0,col].set_ylabel('Nominal position error (m)')
        axes[1,col].set_ylabel('Body proxy clearance (m), y zoom');axes[1,col].set_ylim(-.08,.6)
        axes[1,col].axhline(0,color='red',ls=':',lw=1.)
        for ax in axes[:,col]:ax.set_xlabel('Time (s)');ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(base/('Reference_policy_comparison.'+ext),dpi=300)
    plt.close(fig)
    for group in groups:print(group,flush=True)


if __name__=='__main__':main()
