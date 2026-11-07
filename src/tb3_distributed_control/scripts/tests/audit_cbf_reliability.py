#!/usr/bin/env python3
"""Reconstruct online expert weights/scores from fitted measurements alone."""
import argparse
import json
from pathlib import Path
import numpy as np
from test_cbf_dynamic import inline


def audit(directory):
    path=Path(directory);meta=json.loads((path/'run_metadata.json').read_text())
    if meta['prediction_mode']!='adaptive':return dict(applicable=False)
    def new_tracker():return inline.make_dynamic_tracker(mode='adaptive',dt=float(meta['control_dt']),
        ca_jerk_density=meta['ca_jerk_density'],ca_acceleration_variance=meta['ca_acceleration_variance'],
        singer_tau=meta['singer_tau'],reaction_time=meta['dynamic_reaction_time'],
        extra_margin=meta['dynamic_extra_margin'],acceleration_buffer=meta['acceleration_buffer'],
        stage_velocity_guidance=meta['stage_velocity_guidance'],transport_geometry=meta.get('transport_geometry',False),**meta['reliability_parameters'])
    frames=[json.loads(s) for s in (path/'obstacle_predictions.jsonl').open()]
    checked=0;max_weight=0.;max_score=0.;max_center=0.;first_mismatch=None;trackers={}
    for index,frame in enumerate(frames):
        observer=frame['observer']
        if observer not in trackers:trackers[observer]=new_tracker()
        tracker=trackers[observer]
        stamp=float(frame['stamp'])
        fits=[inline.Ellipse(*m['center'],m['a'],m['b'],m['theta'],m['point_count']) for m in frame['measurements']]
        tracker.update(fits,stamp)
        expected={o['id']:o for o in tracker.obstacles(tracker.snapshot(stamp),stamp)}
        actual={o['id']:o for o in frame['tracks']}
        try:
            assert set(expected)==set(actual),'Track IDs differ (possible missing log prefix/frame)'
            for key,o in actual.items():
                e=expected[key]
                if o['score_count']:
                    assert o['score_origin']<=stamp-tracker.score_horizon+1e-9
                np.testing.assert_allclose(o['model_weights'],e['model_weights'],rtol=0,atol=1e-10)
                assert o['score_count']==e['score_count'] and o['selected_model']==e['selected_model']
                score=np.array([np.nan if v is None else v for v in o['model_scores']])
                es=np.array([np.nan if v is None else v for v in e['model_scores']])
                np.testing.assert_allclose(score,es,rtol=0,atol=1e-10,equal_nan=True)
                np.testing.assert_allclose(o['centers'],e['prediction_centers'],rtol=0,atol=1e-9)
                np.testing.assert_allclose(o['axes'],e['prediction_axes'],rtol=0,atol=1e-9)
                covariance=np.asarray(o['position_covariances'])
                np.testing.assert_allclose(covariance,e['prediction_position_covariances'],rtol=0,atol=1e-10)
                assert np.all(np.linalg.eigvalsh(covariance)>=-1e-12)
                max_weight=max(max_weight,float(np.max(np.abs(o['model_weights']-e['model_weights']))))
                if np.isfinite(score).any():max_score=max(max_score,float(np.nanmax(np.abs(score-es))))
                max_center=max(max_center,float(np.max(np.abs(o['centers']-e['prediction_centers']))))
                checked+=1
        except (AssertionError,ValueError) as exc:
            first_mismatch=dict(frame=index,observer=observer,stamp=stamp,error=str(exc));break
    result=dict(applicable=True,passed=first_mismatch is None and checked>0,frames=len(frames),
        forecast_records_reconstructed=checked,observers=len(trackers),max_weight_difference=max_weight,
        max_score_difference=max_score,max_center_difference=max_center,first_mismatch=first_mismatch,
        note='Only logged fitted ellipses and stamps enter reconstruction. No Gazebo truth or scenario phase. A missing prefix/frame can invalidate exact reconstruction; do not silently repair it.')
    (path/'reliability_audit.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    result=audit(args.directory);print(json.dumps(result,indent=2))
    if result.get('passed') is False:raise SystemExit(1)
