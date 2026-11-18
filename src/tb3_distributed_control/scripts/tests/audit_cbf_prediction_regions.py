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
    shape_count=0;direction_count=0;velocity_labels=0;center_markers=0
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
            shape=marker['ns'].startswith('prediction_shape_') and marker['type']==4
            direction=marker['ns']=='prediction_motion_direction' and marker['type']==0
            velocity_label=marker['ns']=='prediction_velocity' and marker['type']==9
            centers_marker=marker['ns']=='prediction_centers' and marker['type']==7
            if not(hard or soft or shape or direction or velocity_label or centers_marker):continue
            stamp=marker['stamp'];input_time,snapshot=min(samples,key=lambda item:abs(item[0]-stamp))
            if abs(input_time-stamp)>2e-8:
                errors.append(dict(stamp=stamp,error='No same-stamp solver input'));continue
            cfg=snapshot['config'];n=cfg['horizon']
            stage=n if direction or velocity_label or centers_marker else int(marker['ns'].rsplit('_',1)[1])
            obs=next((o for o in snapshot['obstacles'] if not o.get('peer',False)
                      and o.get('name')=='tracked_'+str(marker['id'])),None)
            if obs is None:errors.append(dict(stamp=stamp,error='No same-ID solver obstacle'));continue
            if stage<=n:
                batch=obstacle_batch([obs],n+1,cfg['gamma'])
            else:
                batch=obstacle_batch(progress_obstacles([obs],np.arange(stage+1)*cfg['dt'],cfg),stage+1,cfg['gamma'])
            center=batch.centers[0,stage];a,b=batch.axes[0,stage];scale=1+cfg['margin']/b
            axes=np.array([a*scale,b+cfg['margin']]);c,s=batch.cosine[0,stage],batch.sine[0,stage]
            if hard:
                hard_count+=1
                actual=np.array(marker['position'][:2]);sizes=np.array(marker['scale'][:2])/2.
                quat=marker['orientation'];angle=2*np.arctan2(quat[2],quat[3]);expected=np.arctan2(s,c)
                residual=max(abs(actual-center).max(),abs(sizes-axes).max(),
                             abs(np.arctan2(np.sin(angle-expected),np.cos(angle-expected))))
            elif centers_marker:
                center_markers+=1
                preview=cfg.get('spatial_activation_preview_seconds',1.) if cfg.get('spatial_activation_threshold',-1.)>=0 else n*cfg['dt']
                extra=max(n,int(round(preview/cfg['dt'])));stages=[0,n//2,n]
                extended=batch
                if extra>n:
                    stages += [n+(extra-n)//2,extra]
                    extended=obstacle_batch(progress_obstacles([obs],np.arange(extra+1)*cfg['dt'],cfg),extra+1,cfg['gamma'])
                expected=np.array([(batch if j<=n else extended).centers[0,j] for j in sorted(set(stages))])
                residual=float(abs(np.asarray(marker['points'])[:,:2]-expected).max())
            elif direction or velocity_label:
                start=batch.centers[0,0];v=(batch.centers[0,n]-start)/(n*cfg['dt']);speed=float(np.linalg.norm(v))
                if direction:
                    direction_count+=1
                    expected=np.array([start,start+.32*v/max(speed,1e-12)])
                    residual=float(abs(np.asarray(marker['points'])[:,:2]-expected).max())
                    if speed<=.015:errors.append(dict(stamp=stamp,error='Static estimate has direction arrow'))
                else:
                    velocity_labels+=1
                    residual=0. if marker.get('text')=='#%d v=%.3f m/s (dir)'%(marker['id'],speed) else 1.
            else:
                if soft:soft_count+=1
                if shape:axes=np.array([a,b]);shape_count+=1
                points=np.array(marker['points'])[:,:2]-center
                local=points@np.array([[c,-s],[s,c]])
                residual=float(abs(np.sum((local/axes)**2,axis=1)-1.).max())
            maximum_error=max(maximum_error,float(residual));checked+=1
            if residual>1e-8 or marker['frame']!='world' or not 0<marker['color'][3]<=1 or marker['lifetime']<=0:
                errors.append(dict(stamp=stamp,namespace=marker['ns'],residual=float(residual),error='Region differs from input or invalid display metadata'))
    for stamp,items in paired.items():
        if 'selected' in items and '0' in items:
            duplicates+=1
            if items['selected']!=items['0']:errors.append(dict(stamp=stamp,error='Selected and local display differ'))
    return dict(passed=checked>0 and hard_count>0 and not errors,checked_regions=checked,
        hard_regions=hard_count,soft_regions=soft_count,maximum_numeric_error=maximum_error,
        prediction_envelopes=shape_count,direction_icons=direction_count,velocity_labels=velocity_labels,
        center_markers=center_markers,
        identical_local_selected_frames=duplicates,error_count=len(errors),first_error=errors[0] if errors else None,
        note='Display-to-input geometry audit only; does not certify future physical safety.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');args=p.parse_args()
    output=Path(args.directory)/'prediction_region_audit.json'
    if output.exists():raise FileExistsError(output)
    result=audit(args.directory);output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
