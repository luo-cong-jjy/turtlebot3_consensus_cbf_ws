#!/usr/bin/env python3
"""Recorded centre-set construction and its scan-time forecast containment.

Conditional measured-geometry audit, not ground-truth safety certification.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from test_cbf_inline_geometry import inline


def audit(directory):
    path=Path(directory);meta=json.loads((path/'run_metadata.json').read_text())
    radius=meta.get('known_radius',0.);errors=[];measurements=0;guards=0;inconsistent=0
    for source in sorted(path.glob('tb3_*_local_clusters.jsonl')):
        for frame in map(json.loads,source.open()):
            for measured in frame['measurements']:
                if radius<=0:continue
                expected=inline.known_radius_center_guard(measured['points'],radius,
                    meta['known_radius_tolerance'],meta['known_point_error'],
                    refine=meta.get('known_radius_refine',False) and not measured.get('circle_quality',{}).get('accepted',False))
                measurements+=1
                if measured.get('known_radius_guard')!=expected:
                    errors.append(dict(stamp=frame['stamp'],error='Centre-set construction differs from raw own returns'))
            for track in frame['tracks']:
                guard=track.get('known_radius_guard')
                if guard is None:continue
                guards+=1;inconsistent+=int(not guard['consistent'])
                disk=guard.get('selected_guard_disk',guard)
                required=disk['radius']+np.linalg.norm(np.asarray(disk['center'])-track['centers'][0])
                if track['observation_kind']!='mvee' or min(track['axes'][0])+1e-10<required:
                    errors.append(dict(stamp=frame['stamp'],error='Recorded forecast does not enclose centre-set object disk'))
    return dict(passed=not errors and measurements>0,measurements_checked=measurements,
        active_forecast_guards=guards,inconsistent_active_guards=inconsistent,
        error_count=len(errors),first_error=errors[0] if errors else None,interpretation=__doc__)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory');parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    result=audit(args.directory);out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
