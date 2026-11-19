"""Disk-intersection containment, numerical degeneracy and forecast bracketing."""
import unittest
import numpy as np
from test_cbf_inline_geometry import inline
from mpc_dcbf_static_obstacles import reference_covariance_matrices,reference_axis_inflation
inline.reference_covariance_matrices=reference_covariance_matrices
inline.reference_axis_inflation=reference_axis_inflation


class TightCenterGuardTests(unittest.TestCase):
    def test_two_disk_lens_has_analytic_coordinate_bounds(self):
        a=.055;d=.135;g=inline.known_radius_center_guard([[-a,0],[a,0]],.12,refine=True)
        half=np.array([d-a,np.sqrt(d*d-a*a)])
        np.testing.assert_allclose(g['center_box_lower'],-half,atol=1e-8)
        np.testing.assert_allclose(g['center_box_upper'],half,atol=1e-8)
        self.assertLess(g['radius'],.26)

    def test_every_retained_disk_contains_full_true_body_for_bounded_arcs(self):
        rng=np.random.RandomState(52)
        for n in (1,3,5,12,30):
            for unused in range(20):
                center=rng.uniform(-3,3,2);radius=rng.uniform(.115,.125)
                angles=np.linspace(0,rng.uniform(.1,np.pi),n)+rng.uniform(-3,3)
                points=center+radius*np.column_stack([np.cos(angles),np.sin(angles)])
                error=rng.normal(size=(n,2));error*=rng.uniform(0,.01,(n,1))/np.linalg.norm(error,axis=1)[:,None]
                g=inline.known_radius_center_guard(points+error,.12,refine=True)
                for disk in g.get('enclosing_disks',[g]):
                    self.assertLessEqual(np.linalg.norm(center-disk['center'])+radius,disk['radius']+1e-10)

    def test_degenerate_empty_and_large_sets_retain_a_finite_guard(self):
        for points in ([[0,0]]*5,[[0,0],[.27,0]],[[0,0],[1,0]],np.column_stack([np.linspace(0,.1,65),np.zeros(65)])):
            g=inline.known_radius_center_guard(points,.12,refine=True)
            self.assertTrue(np.isfinite(g['radius']));self.assertGreaterEqual(g['radius'],.125)
        self.assertEqual(inline.known_radius_center_guard(np.column_stack([np.linspace(0,.1,65),np.zeros(65)]),.12,refine=True)['refinement_status'],'point_limit_coarse_retained')

    def test_forecast_is_between_original_and_coarse_without_changing_motion(self):
        points=np.array([[0.,-.075],[-.02,0.],[0.,.075]])
        trackers=[inline.DynamicEllipseTracker(geometry_observer='circle',transport_geometry=True,
            known_radius=r,known_radius_refine=refine) for r,refine in [(0.,False),(.12,False),(.12,True)]]
        for k in range(5):
            shift=np.array([.003*k,0])
            for tracker in trackers:
                e=inline.Ellipse(.01+shift[0],0,.08,.025,.4,3);e.circle_candidate=dict(accepted=False)
                if tracker.known_radius:e.known_radius_guard=inline.known_radius_center_guard(points+shift,.12,refine=tracker.known_radius_refine)
                tracker.update([e],1+k*.1)
        old,coarse,tight=[t.obstacles(t.snapshot(1.4),1.4)[0] for t in trackers]
        self.assertTrue((tight['prediction_axes']+1e-12>=old['prediction_axes']).all())
        self.assertTrue((tight['prediction_axes']<=coarse['prediction_axes']+1e-12).all())
        self.assertTrue((tight['prediction_axes']<coarse['prediction_axes']-1e-6).any())
        np.testing.assert_array_equal(tight['prediction_centers'],coarse['prediction_centers'])
        for field in ('x','P'):np.testing.assert_array_equal(trackers[1].tracks[1][field],trackers[2].tracks[1][field])


if __name__=='__main__':unittest.main()
