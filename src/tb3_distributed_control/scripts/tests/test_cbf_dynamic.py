#!/usr/bin/env python3
"""Predictive association/KF regression on the actual in-controller definitions."""
import unittest
import numpy as np
from test_cbf_inline_geometry import inline
from mpc_dcbf_static_obstacles import reference_covariance_matrices,reference_axis_inflation
from mpc_dcbf_cached_solver import barrier_values,continuous_guidance,_stationary_guidance
from cbf_mpc_core import default_config

inline.reference_covariance_matrices=reference_covariance_matrices
inline.reference_axis_inflation=reference_axis_inflation


def ellipse(x,y,theta=0.):return inline.Ellipse(x,y,.2,.12,theta,20)


class DynamicChecks(unittest.TestCase):
    def test_dynamic_relative_guidance_generates_bypass_and_respects_world_speed(self):
        cfg=default_config();state=np.array([-.6,0.,0.]);goals=np.zeros((11,3))
        goals[:,0]=.3+.015*np.arange(11)
        centers=np.column_stack((-.008*np.arange(11),np.zeros(11)))
        obs=dict(name='moving',center=np.zeros(2),a=.2,b=.15,theta=0.,dynamic=True,
            guidance=False,prediction_centers=centers)
        guide,level=continuous_guidance(state,goals,[obs],cfg,{})
        self.assertGreater(level,0.)
        self.assertGreater(np.max(np.abs(guide[:,1])),1e-6)
        self.assertLessEqual(float(np.linalg.norm(guide,axis=1).max()),.22+1e-12)

    def test_static_guidance_is_numerically_unchanged(self):
        cfg=default_config();state=np.array([-.6,0.,0.]);goals=np.zeros((11,3));goals[:,0]=.3+.015*np.arange(11)
        obs=dict(name='static',center=np.zeros(2),a=.2,b=.15,theta=0.)
        expected=_stationary_guidance(state,goals,[obs],cfg,{})
        actual=continuous_guidance(state,goals,[obs],cfg,{})
        np.testing.assert_array_equal(actual[0],expected[0]);self.assertEqual(actual[1],expected[1])

    def test_no_obstacle_window_keeps_empty_barrier_matrix_shape(self):
        self.assertEqual(barrier_values(np.zeros((11,3)),[],.12).shape,(0,11))

    def test_execution_buffer_is_exactly_an_added_barrier_margin(self):
        original=dict(center=np.array([1.,2.]),a=.4,b=.12,theta=.7)
        extra=.08;scaled=dict(original,a=original['a']*(1+extra/original['b']),b=original['b']+extra)
        states=np.random.default_rng(7).normal(size=(11,3))
        np.testing.assert_allclose(barrier_values(states,[scaled],.12),
            barrier_values(states,[original],.12)-extra,atol=1e-10)

    def test_constant_velocity_prediction_and_control_timestamp_alignment(self):
        tracker=inline.DynamicEllipseTracker()
        for k in range(61):tracker.update([ellipse(1.+.12*k*.1,2.-.08*k*.1)],k*.1)
        self.assertEqual(list(tracker.tracks),[1])
        snapshot=tracker.snapshot(6.)
        before=tracker.tracks[1]['P'].copy()
        obs=tracker.obstacles(snapshot,6.25)[0]
        np.testing.assert_allclose(obs['prediction_centers'][0],[1.+.12*6.25,2.-.08*6.25],atol=.012)
        np.testing.assert_allclose(obs['prediction_centers'][-1],[1.+.12*7.25,2.-.08*7.25],atol=.025)
        self.assertTrue(obs['dynamic'])
        self.assertTrue(np.all(np.diff(obs['prediction_axes'],axis=0)>=-1e-12))
        np.testing.assert_array_equal(before,tracker.tracks[1]['P'])

    def test_predicted_matching_preserves_identity_through_crossing_and_reordering(self):
        tracker=inline.DynamicEllipseTracker(association_distance=.35)
        for k in range(41):
            t=k*.1;objects=[ellipse(-.4+.2*t,0.),ellipse(.4-.2*t,.08)]
            if k%2:objects.reverse()
            tracker.update(objects,t)
        self.assertEqual(len(tracker.tracks),2)
        self.assertGreater(tracker.tracks[1]['x'][5],.1)
        self.assertLess(tracker.tracks[2]['x'][5],-.1)

    def test_missed_observations_keep_prediction_then_expire_and_reset(self):
        tracker=inline.DynamicEllipseTracker(timeout=.6)
        for k in range(21):tracker.update([ellipse(k*.01,0.)],k*.1)
        p=tracker.tracks[1]['P'].copy();tracker.update([ellipse(100.,0.)],2.)
        np.testing.assert_array_equal(p,tracker.tracks[1]['P'])
        tracker.update([],2.2)
        self.assertEqual(len(tracker.snapshot(2.4)),1)
        self.assertEqual(tracker.obstacles(tracker.snapshot(2.4),2.7),[])
        tracker.update([],2.7);self.assertEqual(tracker.tracks,{})
        tracker.update([ellipse(3.,0.)],.1);self.assertEqual(list(tracker.tracks),[1])

    def test_frozen_ablation_changes_centers_without_reading_future_truth(self):
        tracker=inline.DynamicEllipseTracker(mode='frozen')
        for k in range(30):tracker.update([ellipse(.02*k,1.)],k*.1)
        obs=tracker.obstacles(tracker.snapshot(2.9),2.9)[0]
        np.testing.assert_allclose(obs['prediction_centers'],np.tile(obs['center'],(11,1)))
        self.assertGreater(np.linalg.norm(obs['velocity']),.1)

    def test_major_axis_wrap_does_not_invent_large_rotation(self):
        tracker=inline.DynamicEllipseTracker()
        for k in range(20):tracker.update([ellipse(1.,0.,np.pi/2-.01 if k%2 else -np.pi/2+.01)],k*.1)
        angles=[t['x'][4] for t in tracker.tracks.values()]
        self.assertEqual(len(angles),1)
        self.assertLess(abs(np.cos(angles[0])),.03)


if __name__=='__main__':unittest.main()
