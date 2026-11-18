"""Partial-arc centre reconstruction, enclosure and honest missing updates."""
import unittest
import numpy as np
from test_cbf_inline_geometry import inline
from mpc_dcbf_static_obstacles import reference_covariance_matrices,reference_axis_inflation
inline.reference_covariance_matrices=reference_covariance_matrices
inline.reference_axis_inflation=reference_axis_inflation


def observation(center, radius=.2, sensor=(0.,0.),noise=0.,rng=None):
    center=np.asarray(center);sensor=np.asarray(sensor)
    facing=np.arctan2(*(sensor-center)[::-1]);angle=facing+np.linspace(-1.1,1.1,25)
    points=center+radius*np.column_stack((np.cos(angle),np.sin(angle)))
    if noise:points+=rng.normal(0,noise,points.shape)
    fit=inline.enclosing_ellipse(points);fit.circle_candidate=inline.circle_observation(points,fit,sensor)
    return fit,points


class CircleObserverChecks(unittest.TestCase):
    def test_unknown_radius_and_old_complete_enclosure_are_preserved(self):
        fit,points=observation([1.,.2],.27);candidate=fit.circle_candidate
        self.assertTrue(candidate['accepted']);self.assertAlmostEqual(candidate['radius'],.27,places=6)
        angles=np.linspace(-np.pi,np.pi,8192);c,s=np.cos(fit.theta),np.sin(fit.theta)
        boundary=np.column_stack((fit.a*np.cos(angles),fit.b*np.sin(angles)))@np.array([[c,s],[-s,c]])+[fit.cx,fit.cy]
        self.assertLessEqual(float(np.linalg.norm(boundary-candidate['center'],axis=1).max()),candidate['enclosure_radius'])
        self.assertLessEqual(float(np.linalg.norm(points-candidate['center'],axis=1).max()),candidate['enclosure_radius'])

    def test_quality_gate_rejects_lines_short_and_obviously_nonround_shapes(self):
        for points in [np.column_stack((np.linspace(1,1.4,20),np.zeros(20))),
                       np.array([[1.,0.],[1.1,.1],[1.2,.12]])]:
            e=inline.enclosing_ellipse(points);self.assertFalse(inline.circle_observation(points,e,[0,0])['accepted'])
        t=np.linspace(-np.pi,np.pi,80);points=np.column_stack((1+.45*np.cos(t),.12*np.sin(t)))
        e=inline.enclosing_ellipse(points);self.assertFalse(inline.circle_observation(points,e,[0,0])['accepted'])

    def test_view_changes_do_not_become_object_velocity(self):
        model=inline.DynamicEllipseTracker(geometry_observer='circle');rng=np.random.default_rng(52)
        velocity=np.array([.08,-.04])
        for j in range(70):
            stamp=1+j*.1;center=np.array([1.,.7])+velocity*(stamp-1)
            sensor=np.array([.1+.12*(stamp-1),-.3+.15*np.sin(stamp/3)])
            e,_=observation(center,sensor=sensor,noise=.002,rng=rng)
            self.assertTrue(e.circle_candidate['accepted']);model.update([e],stamp)
        track=model.snapshot(stamp)[0]
        np.testing.assert_allclose(track['x'][5:7],velocity,atol=.015)
        np.testing.assert_allclose(track['x'][:2],center,atol=.012)

    def test_short_gap_does_not_measure_its_own_prediction(self):
        model=inline.DynamicEllipseTracker(geometry_observer='circle')
        for j in range(20):
            stamp=1+j*.1;e,_=observation([1+.01*j,.4]);model.update([e],stamp)
        track=model.snapshot(stamp)[0];expected_x,expected_p=model.propagate(track['x'],track['P'],.1)
        bad,_=observation([1.2,.4]);bad.circle_candidate=dict(accepted=False,reason='test_gap')
        model.update([bad],stamp+.1);after=model.snapshot(stamp+.1)[0]
        self.assertFalse(after['position_observed']);self.assertEqual(after['observation_kind'],'circle')
        np.testing.assert_allclose(after['x'][:2],expected_x[:2],atol=1e-12)
        np.testing.assert_allclose(after['P'][:2,:2],expected_p[:2,:2],atol=1e-12)
        for k in (2,3,4):
            bad,_=observation([1.2+.01*k,.4]);bad.circle_candidate=dict(accepted=False,reason='test_gap')
            model.update([bad],stamp+k*.1)
        final=model.snapshot(stamp+.4)[0]
        self.assertEqual(final['observation_kind'],'mvee');self.assertTrue(final['position_observed'])
        self.assertGreaterEqual(final['observer_resets'],1)
        self.assertTrue(np.isfinite(final['P']).all())

    def test_unvalidated_prediction_mode_is_rejected(self):
        with self.assertRaises(ValueError):inline.DynamicEllipseTracker(mode='ca',geometry_observer='circle')
