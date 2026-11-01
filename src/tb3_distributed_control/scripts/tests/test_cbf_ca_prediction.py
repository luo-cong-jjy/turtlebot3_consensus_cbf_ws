#!/usr/bin/env python3
"""Motion-model checks on the production tracker, including old CV fixtures."""
import math
from pathlib import Path
import unittest
import numpy as np
import yaml
from test_cbf_dynamic import inline,ellipse
from test_cbf_inline_geometry import definitions,PACKAGE
from mpc_dcbf_cached_solver import continuous_guidance
from cbf_mpc_core import default_config

scenario=definitions(PACKAGE/'scripts/cbf_dynamic_scenario.py','_ca_scenario_test')


class CAPredictionChecks(unittest.TestCase):
    def test_original_cv_snapshot_unchanged(self):
        data=np.load(Path(__file__).parent/'fixtures/cv_tracker_reference.npz')
        tracker=inline.DynamicEllipseTracker()
        for stamp,cx,cy,a,b,theta in data['measurements']:
            tracker.update([inline.Ellipse(cx,cy,a,b,theta,20)],stamp)
        obs=tracker.obstacles(tracker.snapshot(float(data['measurements'][-1,0])),float(data['now']))[0]
        for key,actual in [('state',tracker.tracks[1]['x']),('covariance',tracker.tracks[1]['P']),
                ('centers',obs['prediction_centers']),('axes',obs['prediction_axes']),('execution_margin',obs['execution_margin'])]:
            np.testing.assert_allclose(actual,data[key],rtol=0,atol=1e-12,err_msg=key)

    def test_exact_ca_transition_and_covariance_composition(self):
        tracker=inline.DynamicEllipseTracker(mode='ca')
        state=np.zeros(9);state[[0,1,5,6,7,8]]=[1,2,.1,-.2,.03,-.04]
        p=np.eye(9)*.02
        x,full=tracker.propagate(state,p,.7)
        np.testing.assert_allclose(x[:2],[1+.1*.7+.5*.03*.7**2,2-.2*.7-.5*.04*.7**2])
        np.testing.assert_allclose(x[5:7],[.121,-.228])
        halfway,hp=tracker.propagate(state,p,.3);split,sp=tracker.propagate(halfway,hp,.4)
        np.testing.assert_allclose(split,x,atol=1e-14);np.testing.assert_allclose(sp,full,atol=1e-14)
        _,q=tracker.propagate(np.zeros(9),np.zeros((9,9)),.7)
        self.assertGreaterEqual(np.linalg.eigvalsh(q).min(),-1e-14)
        self.assertAlmostEqual(q[7,7],.007)
        self.assertAlmostEqual(q[0,7],.01*.7**3/6)

    def test_ca_estimates_acceleration_and_forecasts_without_changing_snapshot(self):
        tracker=inline.DynamicEllipseTracker(mode='ca')
        for k in range(81):
            t=k*.1;tracker.update([ellipse(1+.02*t+.5*.025*t*t,2-.06*t)],t)
        track=tracker.snapshot(8.)[0];before=track['P'].copy()
        obs=tracker.forecast(track,8.17)
        np.testing.assert_allclose(obs['acceleration'],[.025,0.],atol=.003)
        np.testing.assert_allclose(obs['prediction_centers'][-1],[1+.02*9.17+.5*.025*9.17**2,2-.06*9.17],atol=.01)
        self.assertEqual(obs['prediction_velocities'].shape,(11,2))
        np.testing.assert_array_equal(before,track['P'])
        self.assertTrue(np.all(np.diff(obs['prediction_execution_margins'])>0))
        self.assertTrue(np.all(np.linalg.eigvalsh(obs['prediction_position_covariances'])>=-1e-12))

    def test_ca_missed_scans_expire_and_clock_reset(self):
        tracker=inline.DynamicEllipseTracker(mode='ca',timeout=.6)
        for k in range(30):tracker.update([ellipse(.02*k,0.)],.1*k)
        tracker.update([],3.1);self.assertEqual(len(tracker.obstacles(tracker.snapshot(3.1),3.3)),1)
        self.assertEqual(tracker.obstacles(tracker.snapshot(3.1),3.6),[])
        tracker.update([ellipse(1.,0.)],.1);self.assertEqual(list(tracker.tracks),[1])

    def test_ca_rejects_invalid_covariance_parameters(self):
        for kwargs in (dict(ca_jerk_density=0.),dict(ca_jerk_density=float('nan')),
                dict(ca_acceleration_variance=-1),dict(ca_acceleration_variance=float('inf'))):
            with self.assertRaises(ValueError):inline.DynamicEllipseTracker(mode='ca',**kwargs)
        with self.assertRaises(ValueError):inline.DynamicEllipseTracker(mode='ca').propagate(np.zeros(9),np.eye(9),-.1)

    def test_ca_guidance_accepts_varying_segment_velocity(self):
        cfg=default_config();goals=np.zeros((11,3));goals[:,0]=.3+.015*np.arange(11)
        t=np.arange(11)*.1;centers=np.column_stack((-.08*t-.5*.025*t*t,np.zeros(11)))
        obs=dict(name='accelerating',center=np.zeros(2),a=.2,b=.15,theta=0.,dynamic=True,
            guidance=False,prediction_model='ca',prediction_centers=centers)
        guide,level=continuous_guidance(np.array([-.6,0.,0.]),goals,[obs],cfg,{})
        self.assertEqual(guide.shape,(10,2));self.assertGreater(level,0.)
        self.assertTrue(np.all(np.isfinite(guide)));self.assertLessEqual(np.linalg.norm(guide,axis=1).max(),.22+1e-12)

    def test_prescribed_motion_continuous_at_each_acceleration_switch(self):
        specs=yaml.safe_load((PACKAGE/'config/cbf_dynamic_scenarios.yaml').read_text())
        for name in ('accelerating_crossing','decelerating_crossing','stop_go_crossing'):
            spec=specs[name][0];stamp=0.
            for seg in spec['acceleration_segments']:
                stamp+=seg['duration']
                before=np.array(scenario.trajectory(spec,stamp-1e-7));after=np.array(scenario.trajectory(spec,stamp+1e-7))
                np.testing.assert_allclose(before,after,atol=1e-7)
            for t in np.arange(0.,32.,.05):
                _,_,vx,vy=scenario.trajectory(spec,t)
                self.assertLessEqual(math.hypot(vx,vy),.22+1e-12)
        np.testing.assert_allclose(scenario.trajectory(specs['crossing'][0],12.),[2.1,.12,0.,.16])
        np.testing.assert_allclose(scenario.trajectory(specs['stop_go_crossing'][0],10.),[3.,-.42,0.,0.],atol=1e-12)


if __name__=='__main__':unittest.main()
