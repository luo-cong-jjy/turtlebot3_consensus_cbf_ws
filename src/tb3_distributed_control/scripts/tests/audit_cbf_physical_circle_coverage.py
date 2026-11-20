#!/usr/bin/env python3
"""Offline counterexamples: scan-time target circle vs recorded model envelope.

Needs independently labelled observer replay files. A positive boundary excess
is a sampled counterexample; passing samples alone does not prove containment.
No truth or radius is supplied to a controller by this utility.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def audit(root):
    root=Path(root);failures=[];excluded=[];evaluated=[];observers=0;frames_checked=0;angle=np.arange(4096)*2*np.pi/4096
    unit=np.column_stack([np.cos(angle),np.sin(angle)])
    for case in sorted(root.glob('V1_*')):
        evaluation=case/'evaluation.json'
        status=json.loads(evaluation.read_text()) if evaluation.exists() else {}
        if status.get('failure') is not None:
            excluded.append(dict(case=case.name,reason=status['failure']))
            continue
        evaluated.append(case.name)
        specs=json.loads((case/'dynamic_scenario.json').read_text())['specifications']
        if len(specs)!=1:raise ValueError('Exactly one circular evaluation actor required')
        radius=specs[0]['radius'];meta=json.loads((case/'run_metadata.json').read_text())
        body_radius=meta['active_protocol']['body_radius']
        first=json.loads(next((case/'tb3_0_solver_trace.jsonl').open()))
        margin=first['manager']['input_snapshot']['config']['margin']
        truth=list(csv.DictReader((case/'gazebo_truth.csv').open()))
        origin=float(np.median([float(r['ros_time'])-float(r['time']) for r in truth]))
        for source in sorted(case.glob('observer_replay_tb3_*.json')):
            replay=json.loads(source.read_text());observers+=1;frames_checked+=len(replay['records'])
            if replay.get('counterfactual_noise') is not None or replay.get('counterfactual_known_radius_refine') is not None or replay.get('counterfactual_known_radius') is not None or replay.get('offline_quality_rejection_interval') is not None:
                raise ValueError('Modified counterfactual geometry must not be audited against original logged axes')
            if not replay['configured_replay_passed']:raise ValueError('Recorded forecast replay must agree first')
            bad=[r for r in replay['records'] if not r['circle']['occupied_circle_coverage']['0.0']]
            if not bad:continue
            frames=list(map(json.loads,(case/(replay['observer']+'_local_clusters.jsonl')).open()))
            stamps=np.array([f['stamp'] for f in frames])-origin
            for row in bad:
                k=int(np.argmin(abs(stamps-row['time'])))
                if abs(stamps[k]-row['time'])>1e-8:raise ValueError('Missing exact scan-time forecast')
                track=next(t for t in frames[k]['tracks'] if t['id']==row['circle']['track_id'])
                c,s=np.cos(track['theta']),np.sin(track['theta']);rotation=np.array([[c,-s],[s,c]])
                center=np.array(track['centers'][0]);axes=np.array(track['axes'][0])
                boundary=radius*unit+row['truth_position'];local=(boundary-center)@rotation
                maximum=float(np.linalg.norm(local/axes,axis=1).max())
                expanded_axes=axes*(1+margin/axes[1])
                expanded=((radius+body_radius)*unit+row['truth_position']-center)@rotation
                proxy=float(np.linalg.norm(expanded/expanded_axes,axis=1).max())
                failures.append(dict(case=case.name,observer=replay['observer'],time=row['time'],
                    track_id=track['id'],observation_kind=track['observation_kind'],
                    sampled_boundary_max_norm=maximum,expanded_collision_boundary_max_norm=proxy,
                    outside_sample_found=maximum>1,expanded_outside_sample_found=proxy>1,
                    center_error=row['circle']['centre_error']))
    return dict(diagnostic=__doc__,applicable=frames_checked>0,evaluated_attempts=evaluated,labelled_observer_replays=observers,labelled_frames_checked=frames_checked,excluded_incomplete_attempts=excluded,candidate_uncovered_frames=len(failures),
        confirmed_outside_samples=sum(f['outside_sample_found'] for f in failures),
        confirmed_expanded_outside_samples=sum(f['expanded_outside_sample_found'] for f in failures),failures=failures)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root');parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    result=audit(args.root);out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!='failures'},indent=2))
