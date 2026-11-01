#!/usr/bin/env python3
"""Convert legacy CBF CSV metrics to schema 2 in a NEW directory only.

Original measurements are never overwritten. Missing internal auxiliary plans
and time-aligned observer references remain NaN; they cannot be reconstructed
from nominal-reference aliases. This adapter is for this CBF leader protocol,
not arbitrary first-paper pickle/CSV formats.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCRIPTS));sys.path.insert(0,str(SCRIPTS.parent/'纯仿真代码'))
from cbf_mpc_core import timeline_from_leader_history,metric_definitions,wrap_angle,METRICS_SCHEMA_VERSION
from cbf_experiment_results import export_results


def convert(source,output,render=True,evaluation_end=None):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Choose a new output directory: '+str(output))
    meta=json.loads((source/'run_metadata.json').read_text())
    if evaluation_end is not None:
        if not np.isfinite(evaluation_end) or evaluation_end<=0:raise ValueError('Invalid evaluation end')
        meta['total_time']=float(evaluation_end)
    if int(meta.get('metrics_schema_version',1))>=2:raise ValueError('Already schema 2; do not re-convert')
    if meta.get('method')!='MPC-D-CBF':raise ValueError('This adapter requires a legacy MPC-D-CBF experiment')
    with (source/'leader.csv').open() as stream:leader=list(csv.DictReader(stream))
    if len(leader)<2:raise ValueError('Insufficient leader history to decode reference timing')
    timeline=timeline_from_leader_history(leader,float(meta.get('control_dt',.1)))
    names=meta.get('robot_names',['tb3_'+str(i) for i in range(5)])
    files=[source/'run_metadata.json',source/'leader.csv'];converted={}
    required={'x','y','theta','time','ros_time','odom_age','v','omega','observer_x','observer_y','solve_ms','detected_ellipses','status'}
    for i,name in enumerate(names):
        file=source/(name+'.csv')
        if not file.exists():continue
        with file.open() as stream:rows=list(csv.DictReader(stream))
        files.append(file)
        if not rows:continue
        if not required.issubset(rows[0]):raise ValueError('Missing measured pose/timing in '+str(file))
        offset=float(meta.get('formation_radius',.5))*np.array([np.cos(i*2*np.pi/len(names)),np.sin(i*2*np.pi/len(names))])
        for row in rows:
            stamp=float(row['ros_time'])-float(row['odom_age']);reference=timeline.evaluate(stamp)
            nominal=reference['position']+offset;actual=np.array([float(row['x']),float(row['y'])])
            for field in ('consensus_error_x','consensus_error_y','heading_error'):
                if field in row:row['legacy_'+field]=row[field]
            row.update(metrics_schema_version=METRICS_SCHEMA_VERSION,evaluation_ros_time=stamp,
                evaluation_time=reference['logical_time'] if np.isfinite(reference['logical_time']) else float(row['time'])-float(row['odom_age']),
                reference_quality=reference['quality'],leader_reference_x=reference['position'][0],
                leader_reference_y=reference['position'][1],leader_reference_heading=reference['heading'],
                nominal_reference_x=nominal[0],nominal_reference_y=nominal[1],reference_x=nominal[0],reference_y=nominal[1],
                consensus_error_x=actual[0]-nominal[0],consensus_error_y=actual[1]-nominal[1],
                heading_error=wrap_angle(reference['heading']-float(row['theta'])),plan_reference_valid=0,
                plan_source_cycle=-1,plan_reference_age=np.nan)
            for field in ('plan_reference_x','plan_reference_y','plan_execution_error_x','plan_execution_error_y',
                          'observer_aligned_x','observer_aligned_y','observer_error_x','observer_error_y'):
                row[field]=np.nan
        converted[i]=rows
    hashes={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    output.mkdir(parents=True)
    meta.update(timestamp=meta['timestamp']+'-metrics-v2',output_dir=str(output),metrics_schema_version=METRICS_SCHEMA_VERSION,
        metric_definitions=metric_definitions(),converted_from=str(source),source_sha256=hashes,
        unavailable_metrics=['previous accepted auxiliary plan / execution','time-aligned observer reference'])
    (output/'run_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    shutil.copyfile(source/'leader.csv',output/'leader.csv')
    for i,rows in converted.items():
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (output/('tb3_%d.csv'%i)).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    for name in ('geometry_estimates.csv','geometry_statistics.json','dynamic_truth.csv','dynamic_scenario.json'):
        if (source/name).exists():shutil.copyfile(source/name,output/name)
    if render:export_results(output)
    assert all(hashlib.sha256((source/f).read_bytes()).hexdigest()==sha for f,sha in hashes.items())
    return meta


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source');parser.add_argument('--output',required=True)
    parser.add_argument('--evaluation-end',type=float,help='Known experiment duration if missing in legacy metadata')
    args=parser.parse_args();convert(args.source,args.output,evaluation_end=args.evaluation_end);print('New metrics-only export:',args.output)
