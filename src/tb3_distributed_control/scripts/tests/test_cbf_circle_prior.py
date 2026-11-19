"""Maintain the inferred shape under bad arcs, without false position updates."""
import unittest
import numpy as np
from test_cbf_circle_observer import inline,observation


class CirclePriorChecks(unittest.TestCase):
    def seeded(self):
        model=inline.DynamicEllipseTracker(geometry_observer='circle',circle_prior_retention=True)
        for k in range(25):
            stamp=1+k*.1;e,_=observation([1+.01*k,.4]);model.update([e],stamp)
        return model,stamp,model.snapshot(stamp)[0]

    def failed(self,model,stamp,center):
        e,_=observation(center);e.circle_candidate=dict(accepted=False,reason='controlled_quality_loss')
        model.update([e],stamp);return e,model.snapshot(stamp)[0]

    def test_long_quality_loss_propagates_motion_and_covariance(self):
        model,stamp,track=self.seeded()
        for k in range(1,16):
            expected_x,expected_p=model.propagate(track['x'],track['P'],.1)
            e,track=self.failed(model,stamp+k*.1,[1.24+.01*k,.4])
            self.assertEqual(track['observation_kind'],'circle');self.assertFalse(track['position_observed'])
            self.assertTrue(track['prior_shape_retained']);self.assertEqual(track.get('observer_resets',0),0)
            np.testing.assert_allclose(track['x'][:2],expected_x[:2],atol=1e-12)
            np.testing.assert_allclose(track['P'][:2,:2],expected_p[:2,:2],atol=1e-12)
            o=model.forecast(track,stamp+k*.1)
            self.assertEqual(o['circle_quality_degraded'],k>=4)

    def test_offset_prior_circle_is_contained_when_current_cluster_is_tiny(self):
        model,stamp,track=self.seeded();key=track['id']
        model.tracks[key]['circle_prior_offset']=np.array([.04,-.03]);model.tracks[key]['circle_prior_radius']=.3
        center=track['x'][:2]+track['x'][5:7]*.1
        e=inline.Ellipse(*center,.03,.025,.5,3);e.circle_candidate=dict(accepted=False,reason='short_arc')
        model.update([e],stamp+.1)
        self.assertGreaterEqual(e.a,.35-1e-12)
        prior=e.retained_prior
        self.assertLessEqual(np.linalg.norm(np.array(prior['center'])-[e.cx,e.cy])+prior['radius'],e.a+1e-12)

    def test_scan_disappearance_still_expires_track(self):
        model,stamp,track=self.seeded();model.update([],stamp+model.timeout+.01)
        self.assertEqual(model.snapshot(stamp+model.timeout+.01),[])
        e=inline.Ellipse(2.,.4,.1,.05,0.,3);e.circle_candidate=dict(accepted=False,reason='new_object')
        model.update([e],stamp+model.timeout+.11)
        fresh=model.snapshot(stamp+model.timeout+.11)[0]
        self.assertNotEqual(fresh['id'],track['id']);self.assertEqual(fresh['observation_kind'],'mvee')
        self.assertNotIn('circle_prior_radius',fresh)

    def test_genuine_arc_reacquisition_updates_position_again(self):
        model,stamp,track=self.seeded()
        for k in range(1,9):e,track=self.failed(model,stamp+k*.1,[1.24+.01*k,.4])
        predicted_cov=track['P'][:2,:2].copy()
        e,_=observation([1.33,.4]);model.update([e],stamp+.9);after=model.snapshot(stamp+.9)[0]
        self.assertTrue(after['position_observed']);self.assertFalse(after['prior_shape_retained'])
        self.assertLess(float(np.trace(after['P'][:2,:2])),float(np.trace(predicted_cov)))
        self.assertEqual(after.get('observer_resets',0),0)

    def test_retention_flag_requires_circle_model(self):
        with self.assertRaises(ValueError):inline.DynamicEllipseTracker(circle_prior_retention=True)
