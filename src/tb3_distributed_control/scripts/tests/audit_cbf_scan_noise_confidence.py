#!/usr/bin/env python3
"""Verify recorded raw counts and per-scan range-noise events, not task safety."""
import argparse,json
from pathlib import Path
from test_cbf_inline_geometry import inline


def audit(directory):
    path=Path(directory);meta=json.loads((path/'run_metadata.json').read_text());frames=measurements=0;counts=[];errors=[];bounds=[]
    for source in sorted(path.glob('tb3_*_local_clusters.jsonl')):
        for frame in map(json.loads,source.open()):
            n=frame.get('raw_ray_count')
            expected=inline.scan_gaussian_error_bound(meta['known_noise_sigma'],meta['known_noise_alpha'],n,
                meta['known_range_error'],meta['known_transform_error'],meta['known_point_error'])
            frames+=1;counts.append(n);bounds.append(expected['effective_point_error'])
            if expected['union_tail_bound']>meta['known_noise_alpha']:errors.append('Union tail exceeds declared alpha')
            for measurement in frame['measurements']:
                measurements+=1;guard=measurement.get('known_radius_guard',{})
                if guard.get('scan_noise_confidence')!=expected or guard.get('point_error_bound')!=expected['effective_point_error']:
                    errors.append('Confidence payload disagrees with actual frame ray count')
                if measurement['point_count']>n:errors.append('Cluster count exceeds raw per-scan count')
    return dict(passed=frames>0 and measurements>0 and not errors,frames=frames,measurements=measurements,
        raw_ray_count_min=min(counts) if counts else None,raw_ray_count_max=max(counts) if counts else None,
        effective_error_min=min(bounds) if bounds else None,effective_error_max=max(bounds) if bounds else None,
        error_count=len(errors),first_error=errors[0] if errors else None,
        scope='One robot / one scan / raw range-noise event. Not trajectory, association, pose or collision confidence.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    r=audit(a.directory);out.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2));raise SystemExit(0 if r['passed'] else 1)
