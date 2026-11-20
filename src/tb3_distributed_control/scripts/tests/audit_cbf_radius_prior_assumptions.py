#!/usr/bin/env python3
"""Offline check of necessary centre-disk conditions, not an online labeler.

Radial matching (80% within .04m) can include mixed clusters; a violation is
failure of the disk condition, not proof of a particular noise source.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def audit(root):
    root=Path(root);reports={};excluded=[]
    for case in sorted(root.glob('V1_*')):
        status=json.loads((case/'evaluation.json').read_text())
        if status.get('failure') is not None:
            excluded.append(dict(case=case.name,reason=status['failure']));continue
        meta=json.loads((case/'run_metadata.json').read_text())
        reach=meta['known_radius']+meta['known_radius_tolerance']+meta['known_point_error']
        spec=json.loads((case/'dynamic_scenario.json').read_text())['specifications'][0]
        rows=list(csv.DictReader((case/'gazebo_truth.csv').open()))
        times=np.array([float(r['time']) for r in rows]);origin=float(np.median([float(r['ros_time'])-float(r['time']) for r in rows]))
        xy=np.array([[float(r[spec['name']+'_x']),float(r[spec['name']+'_y'])] for r in rows])
        matched=violations=active=active_bad=ambiguous=0;maximum=0.;details=[]
        for source in sorted(case.glob('tb3_*_local_clusters.jsonl')):
            for frame in map(json.loads,source.open()):
                t=frame['stamp']-origin
                if not 4<=t<=20:continue
                centre=np.array([np.interp(t,times,xy[:,j]) for j in (0,1)])
                selections=[]
                for measurement in frame['measurements']:
                    distance=np.linalg.norm(np.asarray(measurement['points'])-centre,axis=1)
                    if np.mean(abs(distance-spec['radius'])<=.04)>=.8:selections.append((measurement,distance))
                if len(selections)!=1:
                    ambiguous+=int(len(selections)>1);continue
                measurement,distance=selections[0];guard=measurement.get('known_radius_guard')
                is_active=guard is not None and any('known_radius_guard' in tr and all(tr['known_radius_guard'].get(k)==v for k,v in guard.items()) for tr in frame['tracks'])
                effective=meta['known_radius']+meta['known_radius_tolerance']+(guard['point_error_bound'] if guard is not None else meta['known_point_error'])
                excess=float(distance.max()-effective);bad=excess>1e-9
                matched+=1;violations+=int(bad);active+=int(is_active);active_bad+=int(is_active and bad);maximum=max(maximum,excess)
                if is_active and bad:details.append(dict(observer=frame['observer'],time=t,maximum_disk_excess=excess))
        reports[case.name]=dict(matched_clusters=matched,necessary_disk_condition_violations=violations,
            active_matched_guards=active,active_guard_condition_violations=active_bad,
            maximum_disk_excess=maximum,ambiguous_frames=ambiguous,active_failure_details=details)
    return dict(interpretation=__doc__,window=[4.,20.],reports=reports,excluded_incomplete_attempts=excluded,
        matched_clusters=sum(r['matched_clusters'] for r in reports.values()),
        necessary_disk_condition_violations=sum(r['necessary_disk_condition_violations'] for r in reports.values()),
        active_matched_guards=sum(r['active_matched_guards'] for r in reports.values()),
        active_guard_condition_violations=sum(r['active_guard_condition_violations'] for r in reports.values()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root');parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    result=audit(args.root);out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ('reports','interpretation')},indent=2))
