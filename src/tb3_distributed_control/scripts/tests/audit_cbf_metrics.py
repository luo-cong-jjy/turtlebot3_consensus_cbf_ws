#!/usr/bin/env python3
"""Independent per-row audit, including prior plan reconstructed from saved trace."""
import argparse,csv,json
from pathlib import Path
import numpy as np


def audit(directory):
    p=Path(directory);meta=json.loads((p/'run_metadata.json').read_text());assert meta['metrics_schema_version']==2
    n=len(meta.get('robot_names',['tb3_'+str(i) for i in range(5)]));radius=meta['formation_radius']
    results={}
    for file in sorted(p.glob('tb3_*.csv')):
        i=int(file.stem.split('_')[1]);rows=list(csv.DictReader(file.open()))
        plans={}
        trace=p/(file.stem+'_solver_trace.jsonl')
        if trace.exists():
            for line in trace.open():
                record=json.loads(line);plan=record.get('accepted_auxiliary_plan')
                if plan is not None:plans[int(record['cycle_id'])]=plan
        plan_errors=[];nominal_errors=[];reference_residuals=[];heading_residuals=[];qualities={}
        causal=0
        for row in rows:
            r={k:float(v) for k,v in row.items() if k in ('x','y','theta','ros_time','odom_age','consensus_error_x','consensus_error_y',
                'reference_x','reference_y','heading_error','evaluation_ros_time','evaluation_time','reference_quality',
                'leader_reference_x','leader_reference_y','leader_reference_heading','plan_reference_x','plan_reference_y',
                'plan_reference_valid','plan_source_cycle','cycle_id','plan_execution_error_x','plan_execution_error_y',
                'observer_aligned_x','observer_aligned_y','observer_error_x','observer_error_y')}
            actual=np.array([r['x'],r['y']]);nominal=np.array([r['reference_x'],r['reference_y']])
            error=np.array([r['consensus_error_x'],r['consensus_error_y']])
            np.testing.assert_allclose(actual-nominal,error,atol=1e-12,equal_nan=True)
            assert abs(r['evaluation_ros_time']-(r['ros_time']-r['odom_age']))<1e-8
            angle=np.arctan2(np.sin(r['leader_reference_heading']-r['theta']),np.cos(r['leader_reference_heading']-r['theta']))
            np.testing.assert_allclose(angle,r['heading_error'],atol=1e-12,equal_nan=True)
            if r['plan_reference_valid']:
                source=int(r['plan_source_cycle']);assert source<int(r['cycle_id']);assert source in plans
                plan=plans[source];positions=np.array(plan['positions']);times=plan['origin']+plan['dt']*np.arange(len(positions))
                reference=np.array([np.interp(r['evaluation_ros_time'],times,positions[:,j]) for j in (0,1)])
                np.testing.assert_allclose(reference,[r['plan_reference_x'],r['plan_reference_y']],atol=1e-10)
                pe=np.array([r['plan_execution_error_x'],r['plan_execution_error_y']])
                np.testing.assert_allclose(reference-actual,pe,atol=1e-10)
                plan_errors.append(float(np.linalg.norm(pe)));nominal_errors.append(float(np.linalg.norm(error)));causal+=1
            else:assert np.isnan(r['plan_execution_error_x']) and np.isnan(r['plan_execution_error_y'])
            quality=int(r['reference_quality']);qualities[str(quality)]=qualities.get(str(quality),0)+1
            t=r['evaluation_time']
            if quality in (1,3) and 0<=t<=float(meta.get('total_time',1e9)):
                if meta['trajectory_type']=='line':leader=np.array([.4+.15*t,0.]);velocity=np.array([.15,0.])
                elif meta['trajectory_type']=='eight_slow_slow':
                    leader=np.array([2.5*np.sin(t/15),.3+2.5*np.sin(t/30)])
                    velocity=np.array([2.5/15*np.cos(t/15),2.5/30*np.cos(t/30)])
                else:continue
                target=leader+radius*np.array([np.cos(i*2*np.pi/n),np.sin(i*2*np.pi/n)])
                reference_residuals.append(float(np.linalg.norm(nominal-target)))
                expected_heading=np.arctan2(velocity[1],velocity[0])
                heading_residuals.append(float(abs(np.arctan2(np.sin(r['leader_reference_heading']-expected_heading),np.cos(r['leader_reference_heading']-expected_heading)))))
        assert causal>0
        assert max(reference_residuals,default=0)<1e-6
        assert max(heading_residuals,default=0)<1e-6
        assert max(plan_errors)>1e-5,'Plan error suspiciously identically zero'
        assert max(abs(np.asarray(plan_errors)-np.asarray(nominal_errors)))>1e-3,'Plan metric aliases formation error'
        results[str(i)]=dict(rows=len(rows),prior_plan_samples_audited=causal,reference_quality_counts=qualities,
            max_nominal_analytic_residual=max(reference_residuals,default=None),
            max_heading_analytic_residual=max(heading_residuals,default=None),
            mean_plan_execution_error=float(np.mean(plan_errors)),max_plan_execution_error=float(max(plan_errors)),
            max_nominal_error_during_plan_samples=float(max(nominal_errors)))
    result=dict(passed=True,robots=results,checks=['pose/reference same timestamp','target-minus-actual heading sign',
        'prior accepted plan trace reconstruction','no fabricated plan zero','no nominal/plan alias',
        'analytic prescribed leader cross-check'])
    (p/'metrics_audit.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    print(json.dumps(audit(args.directory),indent=2))
