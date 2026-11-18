#!/usr/bin/env python3
"""Check actual RViz markers against time-matched frozen solver inputs."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PACKAGE/'纯仿真代码'))
from mpc_dcbf_cached_solver import obstacle_batch,progress_obstacles


def audit(directory):
    path=Path(directory);inputs={};errors=[];checked=0;hard_count=0;soft_count=0;maximum_error=0.
    for trace in sorted(path.glob('tb3_*_solver_trace.jsonl')):
        observer=trace.name.split('_')[1]
        times={int(r['cycle_id']):float(r['ros_time']) for r in csv.DictReader((path/('tb3_'+observer+'.csv')).open())}
        inputs[observer]=[(times[r['cycle_id']],r['manager']['input_snapshot'])
            for r in map(json.loads,trace.open()) if r.get('manager',{}).get('input_snapshot')]
    paired={};duplicates=0
    for row in map(json.loads,(path/'prediction_regions.jsonl').open()):
        observer='0' if row['observer']=='selected' else row['observer']
        samples=inputs[observer]
        if row['markers']:
            key=round(row['markers'][0]['stamp'],8)
            pair=paired.setdefault(key,{})
            pair[row['observer']]=row['markers']
        for marker in row['markers']:
            if marker['action']!=0:continue
            hard=marker['ns'].startswith('prediction_hard_') and marker['type']==3
            soft=marker['ns'].startswith('prediction_soft_') and marker['type']==5
            if not(hard or soft):continue
            stamp=marker['stamp'];input_time,snapshot=min(samples,key=lambda item:abs(item[0]-stamp))
            if abs(input_time-stamp)>2e-8:
                errors.append(dict(stamp=stamp,error='No same-stamp solver input'));continue
            cfg=snapshot['config'];n=cfg['horizon'];stage=int(marker['ns'].rsplit('_',1)[1])
            obs=next((o for o in snapshot['obstacles'] if not o.get('peer',False)
                      and o.get('name')=='tracked_'+str(marker['id'])),None)
            if obs is None:errors.append(dict(stamp=stamp,error='No same-ID solver obstacle'));continue
            if hard:
                batch=obstacle_batch([obs],n+1,cfg['gamma']);hard_count+=1
            else:
                assert stage>n
                batch=obstacle_batch(progress_obstacles([obs],np.arange(stage+1)*cfg['dt'],cfg),stage+1,cfg['gamma']);soft_count+=1
            center=batch.centers[0,stage];a,b=batch.axes[0,stage];scale=1+cfg['margin']/b
            axes=np.array([a*scale,b+cfg['margin']]);c,s=batch.cosine[0,stage],batch.sine[0,stage]
            if hard:
                actual=np.array(marker['position'][:2]);sizes=np.array(marker['scale'][:2])/2.
                quat=marker['orientation'];angle=2*np.arctan2(quat[2],quat[3]);expected=np.arctan2(s,c)
                residual=max(abs(actual-center).max(),abs(sizes-axes).max(),
                             abs(np.arctan2(np.sin(angle-expected),np.cos(angle-expected))))
            else:
                points=np.array(marker['points'])[:,:2]-center
                local=points@np.array([[c,-s],[s,c]])
                residual=float(abs(np.sum((local/axes)**2,axis=1)-1.).max())
            maximum_error=max(maximum_error,float(residual));checked+=1
            if residual>1e-8 or marker['frame']!='world' or not 0<marker['color'][3]<1 or marker['lifetime']<=0:
                errors.append(dict(stamp=stamp,namespace=marker['ns'],residual=float(residual),error='Region differs from input or invalid display metadata'))
    for stamp,items in paired.items():
        if 'selected' in items and '0' in items:
            duplicates+=1
            if items['selected']!=items['0']:errors.append(dict(stamp=stamp,error='Selected and local display differ'))
    return dict(passed=checked>0 and hard_count>0 and not errors,checked_regions=checked,
        hard_regions=hard_count,soft_regions=soft_count,maximum_numeric_error=maximum_error,
        identical_local_selected_frames=duplicates,error_count=len(errors),first_error=errors[0] if errors else None,
        note='Display-to-input geometry audit only; does not certify future physical safety.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');args=p.parse_args()
    output=Path(args.directory)/'prediction_region_audit.json'
    if output.exists():raise FileExistsError(output)
    result=audit(args.directory);output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
