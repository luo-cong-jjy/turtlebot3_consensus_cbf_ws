#!/usr/bin/env python3
"""Recompute nominal body completion; no probability or physical guarantee."""
import argparse,json
from pathlib import Path
import numpy as np
from test_cbf_inline_geometry import inline


def audit(directory):
    p=Path(directory);meta=json.loads((p/'run_metadata.json').read_text());measured=active=0;errors=[]
    radius=meta.get('nominal_body_radius',0.)
    for file in sorted(p.glob('tb3_*_local_clusters.jsonl')):
        for frame in map(json.loads,file.open()):
            for m in frame['measurements']:
                if radius<=0:continue
                base=m.get('base_enclosure',m);ellipse=inline.Ellipse(*base['center'],base['a'],base['b'],base['theta'],base['point_count'])
                expected=inline.nominal_body_completion(m['points'],ellipse,radius,meta['nominal_body_padding']);measured+=1
                if m.get('nominal_body_guard')!=expected:errors.append(dict(stamp=frame['stamp'],error='Nominal completion differs from own points'))
            for track in frame['tracks']:
                g=track.get('nominal_body_guard')
                if g is None:continue
                active+=1;needed=g['radius']+np.linalg.norm(np.asarray(g['center'])-track['centers'][0])
                if g['probability_claim'] or min(track['axes'][0])+1e-10<needed:errors.append(dict(stamp=frame['stamp'],error='Invalid nominal forecast enclosure or probability claim'))
    return dict(applicable=radius>0,passed=not errors and (radius<=0 or measured>0),measurements=measured,active_guards=active,error_count=len(errors),first_error=errors[0] if errors else None,
        note='Nominal geometric consistency only; padding is not a Gaussian hard bound or confidence level.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    r=audit(a.directory);out.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2));raise SystemExit(0 if r['passed'] else 1)
