#!/usr/bin/env python3
"""Check saved circular measurements enclose the old MVEE and real returns."""
import argparse,json
from pathlib import Path
import numpy as np


def audit(directory):
    p=Path(directory);circle=0;mvee=0;holds=0;errors=[];max_excess=-1e9;retained=0;prior_excess=-1e9
    for frame in map(json.loads,(p/'obstacle_predictions.jsonl').open()):
        for item in frame['measurements']:
            points=np.asarray(item['points']);center=np.asarray(item['center']);c,s=np.cos(item['theta']),np.sin(item['theta'])
            local=(points-center)@np.array([[c,-s],[s,c]])
            normalized=np.linalg.norm(local/[item['a'],item['b']],axis=1)
            if normalized.max()>1+1e-8:errors.append(dict(stamp=frame['stamp'],error='Measured points outside selected geometry'))
            if item.get('observation_kind','mvee')!='circle':mvee+=1;continue
            circle+=1;holds+=int(not item['position_observed']);old=item['base_enclosure']
            angles=np.arange(4096)*(2*np.pi/4096);c,s=np.cos(old['theta']),np.sin(old['theta'])
            boundary=np.column_stack((old['a']*np.cos(angles),old['b']*np.sin(angles)))@np.array([[c,s],[-s,c]])+old['center']
            excess=float(np.linalg.norm(boundary-center,axis=1).max()-item['a']);max_excess=max(max_excess,excess)
            if abs(item['a']-item['b'])>1e-10 or excess>1e-9:
                errors.append(dict(stamp=frame['stamp'],error='Old complete padded ellipse not enclosed',excess=excess))
            if 'retained_prior' in item:
                retained+=1;prior=item['retained_prior']
                extra=float(np.linalg.norm(np.asarray(prior['center'])-center)+prior['radius']-item['a'])
                prior_excess=max(prior_excess,extra)
                if extra>1e-9 or item['position_observed']:
                    errors.append(dict(stamp=frame['stamp'],error='Prior circle lost or copied position treated as measurement',excess=extra))
    return dict(passed=not errors and circle+mvee>0,circle_measurements=circle,mvee_measurements=mvee,
        geometry_only_circle_holds=holds,maximum_old_enclosure_excess=max_excess if circle else None,
        retained_prior_checks=retained,maximum_retained_prior_excess=prior_excess if retained else None,
        error_count=len(errors),first_error=errors[0] if errors else None,
        note='Checks measured geometry inclusion, not unknown physical-object or future safety certification.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory');a=parser.parse_args()
    p=Path(a.directory)/'measurement_geometry_audit.json'
    if p.exists():raise FileExistsError(p)
    result=audit(a.directory);p.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)
