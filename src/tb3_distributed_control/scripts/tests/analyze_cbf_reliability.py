#!/usr/bin/env python3
"""Descriptive paired-seed results, retaining failures and censored recovery."""
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
    base=Path(args.directory);records=json.loads((base/'runs.json').read_text());results=[]
    for run in records:
        path=base/run['directory'];evaluation=json.loads((path/'evaluation.json').read_text())
        meta=json.loads((path/'run_metadata.json').read_text()) if (path/'run_metadata.json').exists() else {}
        statistics=json.loads((path/'experiment_statistics.json').read_text()) if (path/'experiment_statistics.json').exists() else {}
        stat=statistics.get('0',{});rows=list(csv.DictReader((path/'tb3_0.csv').open())) if (path/'tb3_0.csv').exists() else []
        result=dict(run,metadata_seed=meta.get('gazebo_seed'),source_commit=json.loads((base/'protocol.json').read_text())['source_commit'],
            nominal=stat.get('position_error_summary'),heading=stat.get('heading_error_summary'),
            statuses=stat.get('statuses'),overruns=stat.get('overruns'),timing_max_ms=stat.get('max_computation_ms'),
            timing_p95_ms=evaluation.get('robot_statistics',{}).get('0',{}).get('timing_p95_ms'),
            no_truth_input=evaluation.get('no_truth_input_passed'),settling_time_005=None)
        if rows:
            result['v_total_variation']=float(np.abs(np.diff([float(r['v']) for r in rows])).sum())
            result['omega_total_variation']=float(np.abs(np.diff([float(r['omega']) for r in rows])).sum())
            data=[];last=-np.inf
            for r in rows:
                t=float(r['evaluation_time'])
                if 0<=t<=run['duration'] and t>last+1e-9 and abs(float(r['odom_age']))<=meta['data_timeout']:
                    data.append((t,np.hypot(float(r['consensus_error_x']),float(r['consensus_error_y']))));last=t
            if data:
                times,error=np.asarray(data).T;peak=int(np.nanargmax(error));bad=np.flatnonzero((error>.05)|~np.isfinite(error))
                candidate=max(peak,int(bad[-1])+1 if len(bad) else 0)
                if candidate<len(times) and times[-1]-times[candidate]>=2. and np.diff(times[candidate:]).max(initial=0.)<=1.5*meta['control_dt']:
                    result['settling_time_005']=float(times[candidate])
        results.append(result)
    primary=[r for r in results if not r.get('repeat')];groups=[];paired=[]
    for case in dict.fromkeys(r['scenario'] for r in primary):
        for mode in ('cv','ca','adaptive'):
            group=[r for r in primary if (r['scenario'],r['mode'])==(case,mode)]
            if not group:continue
            values=[r['nominal']['rmse'] for r in group if r['nominal'] is not None]
            clear=[r['clearance'] for r in group if r['clearance'] is not None]
            groups.append(dict(scenario=case,mode=mode,trials=len(group),passed=sum(r['passed'] for r in group),
                rmse_samples=len(values),mean_rmse=float(np.mean(values)) if values else None,
                rmse_range=[float(min(values)),float(max(values))] if values else None,
                minimum_clearance=min(clear) if clear else None,
                recovery_observed=sum(r['settling_time_005'] is not None for r in group)))
        for seed in sorted({r['seed'] for r in primary if r['scenario']==case}):
            candidate=next(r for r in primary if (r['scenario'],r['seed'],r['mode'])==(case,seed,'adaptive'))
            for mode in ('cv','ca'):
                other=next(r for r in primary if (r['scenario'],r['seed'],r['mode'])==(case,seed,mode))
                if candidate['nominal'] is None or other['nominal'] is None:continue
                paired.append(dict(scenario=case,seed=seed,comparison='adaptive - '+mode,
                    rmse_difference=candidate['nominal']['rmse']-other['nominal']['rmse'],
                    clearance_difference=candidate['clearance']-other['clearance'],
                    omega_tv_difference=candidate['omega_total_variation']-other['omega_total_variation'],
                    candidate_passed=candidate['passed'],baseline_passed=other['passed']))
    summary=dict(runs=results,groups=groups,paired_differences=paired,
        note='Two distinct seeds are exploratory descriptions, not significance or safety-probability evidence. The predeclared same-seed repeat is secondary and excluded from group means. Failures remain; missing/unfinished recovery stays null. CSV command TV includes stop commands and terminal rows.')
    (base/'reliability_comparison.json').write_text(json.dumps(summary,indent=2))
    cases=list(dict.fromkeys(r['scenario'] for r in primary));colors={'cv':'tab:blue','ca':'tab:orange','adaptive':'tab:purple'}
    fig,axes=plt.subplots(2,len(cases),figsize=(5*len(cases),7),squeeze=False)
    for col,case in enumerate(cases):
        for run in [r for r in primary if r['scenario']==case and r['nominal'] is not None]:
            path=base/run['directory'];rows=list(csv.DictReader((path/'tb3_0.csv').open()))
            label=run['mode'].upper()+' seed '+str(run['seed'])+(' failed' if not run['passed'] else '')
            style='-' if run['seed']==min(r['seed'] for r in primary) else '--'
            axes[0,col].plot([float(r['evaluation_time']) for r in rows],
                [np.hypot(float(r['consensus_error_x']),float(r['consensus_error_y'])) for r in rows],
                color=colors[run['mode']],ls=style,lw=1.4,label=label)
            truth=list(csv.DictReader((path/'gazebo_truth.csv').open()));truth=[r for r in truth if 0<=float(r['time'])<=run['duration']]
            t=np.array([float(r['time']) for r in truth]);clear=np.array([np.hypot(float(r['tb3_0_x'])-float(r['cbf_dynamic_0_x']),float(r['tb3_0_y'])-float(r['cbf_dynamic_0_y']))-.31 for r in truth])
            axes[1,col].plot(t,clear,color=colors[run['mode']],ls=style,lw=1.4,label=label)
        axes[0,col].set_title(case.replace('_',' '));axes[0,col].set_ylabel('Nominal position error (m)')
        axes[1,col].set_ylabel('Near-encounter clearance (m), y zoom');axes[1,col].set_ylim(-.08,.6)
        axes[1,col].axhline(0,color='red',ls=':',lw=1.)
        for ax in axes[:,col]:ax.set_xlabel('Time (s)');ax.grid(alpha=.3);ax.legend(fontsize=6)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(base/('Paired_reliability_comparison.'+ext),dpi=300)
    plt.close(fig)
    adaptive=[r for r in primary if r['mode']=='adaptive' and r['nominal'] is not None]
    if adaptive:
        seeds=sorted({r['seed'] for r in primary})
        fig,axes=plt.subplots(len(seeds),len(cases),figsize=(5*len(cases),3*len(seeds)),squeeze=False)
        for run in adaptive:
            col=cases.index(run['scenario']);row=sorted({r['seed'] for r in primary}).index(run['seed']);ax=axes[row,col]
            path=base/run['directory'];frames=[json.loads(s) for s in (path/'obstacle_predictions.jsonl').open()]
            leader=list(csv.DictReader((path/'leader.csv').open()));origin=float(np.median([float(r['ros_time'])-float(r['trajectory_time']) for r in leader]))
            counts={}
            for frame in frames:
                for tr in frame['tracks']:counts[tr['id']]=counts.get(tr['id'],0)+1
            if not counts:continue
            key=max(counts,key=counts.get);samples=[(frame['stamp']-origin,tr['model_weights']) for frame in frames for tr in frame['tracks'] if tr['id']==key and 0<=frame['stamp']-origin<=run['duration']]
            if not samples:continue
            t=np.array([s[0] for s in samples]);weights=np.array([s[1] for s in samples])
            # Break curves across missing time intervals; never interpolate a hidden selection.
            for j,mode in enumerate(('cv','ca','singer')):
                values=weights[:,j].copy();values[np.r_[False,np.diff(t)>.3]]=np.nan
                ax.plot(t,values,lw=1.5,label=mode.upper())
            ax.set_title('%s seed %s track %s'%(run['scenario'].replace('_',' '),run['seed'],key),fontsize=9)
            ax.set_ylim(-.03,1.03);ax.set_xlabel('Time (s)');ax.set_ylabel('Weights (not probabilities)');ax.grid(alpha=.3);ax.legend(fontsize=7)
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(base/('Online_expert_weights.'+ext),dpi=300)
        plt.close(fig)
    repeats=[r for r in results if r.get('repeat') and r['nominal'] is not None]
    if repeats:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for repeat in repeats:
            original=next(r for r in primary if (r['scenario'],r['seed'],r['mode'])==(repeat['scenario'],repeat['seed'],repeat['mode']))
            for run,label in ((original,'primary'),(repeat,'predeclared repeat')):
                path=base/run['directory'];data=list(csv.DictReader((path/'tb3_0.csv').open()))
                axes[0].plot([float(r['evaluation_time']) for r in data],[np.hypot(float(r['consensus_error_x']),float(r['consensus_error_y'])) for r in data],lw=1.8,label=label)
                truth=list(csv.DictReader((path/'gazebo_truth.csv').open()));truth=[r for r in truth if 0<=float(r['time'])<=run['duration']]
                axes[1].plot([float(r['tb3_0_x']) for r in truth],[float(r['tb3_0_y']) for r in truth],lw=1.8,label=label)
        axes[0].set_xlabel('Time (s)');axes[0].set_ylabel('Nominal position error (m)')
        axes[1].set_xlabel('x (m)');axes[1].set_ylabel('y (m)');axes[1].set_aspect('equal',adjustable='datalim')
        for ax in axes:ax.grid(alpha=.3);ax.legend()
        fig.suptitle('Same Gazebo seed does not freeze ROS / wall-budget solver timing')
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(base/('Predeclared_same_seed_repeat.'+ext),dpi=300)
        plt.close(fig)
    for group in groups:print(group,flush=True)


if __name__=='__main__':main()
