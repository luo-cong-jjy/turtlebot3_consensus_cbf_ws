#!/usr/bin/env python3
"""Keep positive/negative Gazebo evidence and paired surrogate metrics together."""
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
    base=Path(args.directory);reports=[]
    for path in sorted(base.glob('gazebo_*')):
        file=path/'evaluation.json'
        if not file.exists():continue
        evaluation=json.loads(file.read_text())
        report=dict(directory=path.name,scenario=evaluation['scenario'],mode=evaluation['prediction_mode'],
            robots=evaluation['robots'],passed=evaluation['passed'],failure=evaluation['failure'],
            body_clearance_min=evaluation.get('dynamic_clearance_min'),
            no_truth_input=evaluation.get('no_truth_input_passed'),robot_statistics={})
        stats_path=path/'experiment_statistics.json'
        statistics=json.loads(stats_path.read_text()) if stats_path.exists() else {}
        for key,s in statistics.items():
            rows=list(csv.DictReader((path/('tb3_'+key+'.csv')).open()))
            report['robot_statistics'][key]=dict(
                nominal_position=s['position_error_summary'],heading=s['heading_error_summary'],
                previous_plan_execution=s['plan_execution_summary'],statuses=s['statuses'],
                timing_p95_ms=evaluation['robot_statistics'][key]['timing_p95_ms'],
                timing_max_ms=s['max_computation_ms'],overruns=s['overruns'],
                # Command TV includes safety stops; removing them would hide failures.
                command_v_tv=float(np.abs(np.diff([float(r['v']) for r in rows])).sum()),
                command_omega_tv=float(np.abs(np.diff([float(r['omega']) for r in rows])).sum()),
                max_estimated_acceleration=max(float(r.get('max_obstacle_acceleration',0)) for r in rows))
        source=path/'sources/sha256.json'
        report['source_sha256']=json.loads(source.read_text()) if source.exists() else {}
        reports.append(report)
    (base/'gazebo_comparison.json').write_text(json.dumps(dict(runs=reports,
        note='Single runs, not statistical superiority. Nominal/heading/previous-plan errors use metrics schema 2. Clearance uses 20 Hz actual Gazebo positions and a 0.13 m circular robot proxy. Keep failures. Different closed-loop viewpoints mean forecast scores are not common-input ablations.'),indent=2))
    cases=['accelerating_crossing','decelerating_crossing','stop_go_crossing','crossing']
    fig,axes=plt.subplots(2,4,figsize=(17,7))
    for col,case in enumerate(cases):
        for report in [r for r in reports if r['scenario']==case and r['robots']==1 and r['failure'] is None]:
            path=base/report['directory'];meta=json.loads((path/'run_metadata.json').read_text())
            truth=list(csv.DictReader((path/'gazebo_truth.csv').open()))
            truth=[r for r in truth if 0<=float(r['time'])<=meta['total_time']+.05]
            color={'cv':'tab:blue','ca':'tab:orange','frozen':'gray'}[report['mode']]
            t=np.array([float(r['time']) for r in truth])
            robot=np.array([[float(r['tb3_0_x']),float(r['tb3_0_y'])] for r in truth])
            actor=np.array([[float(r['cbf_dynamic_0_x']),float(r['cbf_dynamic_0_y'])] for r in truth])
            label=report['mode'].upper()+(' (failed acceptance)' if not report['passed'] else '')
            axes[0,col].plot(*robot.T,color=color,lw=2,label=label)
            axes[0,col].plot(*actor.T,':',color=color,lw=1,alpha=.5)
            clearance=np.linalg.norm(robot-actor,axis=1)-.13-.18
            axes[1,col].plot(t,clearance,color=color,lw=1.5,label=label)
        axes[0,col].set_title(case.replace('_',' '));axes[0,col].set_aspect('equal',adjustable='datalim')
        axes[0,col].set_xlabel('x (m)');axes[0,col].set_ylabel('y (m)')
        axes[1,col].axhline(0,color='red',ls='--',lw=1)
        axes[1,col].set_xlabel('Time (s)');axes[1,col].set_ylabel('Body clearance (m)')
        for ax in axes[:,col]:ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(base/('CV_CA_Gazebo_comparison.'+ext),dpi=300)
    plt.close(fig)
    file=base/'common_input_replay_final/summary.json'
    if file.exists():
        data=json.loads(file.read_text());fig,axes=plt.subplots(1,2,figsize=(12,4))
        selected=[('crossing','constant_or_stopped'),('accelerating_crossing','accelerating'),
            ('decelerating_crossing','accelerating'),('stop_go_crossing','accelerating'),('stationary','constant_or_stopped')]
        for ax,noise in zip(axes,(.005,.015)):
            for offset,mode in zip((-.23,0.,.23),('frozen','cv','ca')):
                values=[next(r['modes'][mode]['displacement_rmse'] for r in data if
                    (r['case'],r['phase'],r['noise'],r['horizon_s'])==(case,phase,noise,1.)) for case,phase in selected]
                ax.bar(np.arange(len(selected))+offset,values,width=.23,label=mode.upper())
            ax.set_xticks(np.arange(len(selected)));ax.set_xticklabels(['constant','accelerate','decelerate','stop-go','stationary'],rotation=20)
            ax.set_title('Common input; center noise = %.3f m'%noise);ax.set_ylabel('1 s displacement RMSE (m)')
            ax.legend();ax.grid(axis='y',alpha=.3)
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(base/('CV_CA_common_input_comparison.'+ext),dpi=300)
        plt.close(fig)
    for r in reports:
        s=r['robot_statistics'].get('0',{})
        print(r['directory'],r['passed'],r['body_clearance_min'],s.get('nominal_position',{}).get('rmse'),s.get('statuses'))


if __name__=='__main__':main()
