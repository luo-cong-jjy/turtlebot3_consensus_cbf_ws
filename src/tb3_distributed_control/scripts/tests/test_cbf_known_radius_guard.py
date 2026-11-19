"""Bounded-point containment and unchanged motion/filter semantics."""
import unittest
import numpy as np
from test_cbf_inline_geometry import inline
from mpc_dcbf_static_obstacles import reference_covariance_matrices,reference_axis_inflation
inline.reference_covariance_matrices=reference_covariance_matrices
inline.reference_axis_inflation=reference_axis_inflation


class KnownRadiusGuardTests(unittest.TestCase):
    def test_full_body_containment_for_bounded_sparse_arcs(self):
        rng=np.random.RandomState(17)
        for count in (1,3,5,20):
            for unused in range(30):
                center=rng.uniform(-2,2,2);radius=rng.uniform(.115,.125)
                angles=np.linspace(-.3,.8,count)+rng.uniform(-np.pi,np.pi)
                points=center+radius*np.column_stack([np.cos(angles),np.sin(angles)])
                noise=rng.normal(size=(count,2));noise*=rng.uniform(0,.01,(count,1))/np.linalg.norm(noise,axis=1)[:,None]
                g=inline.known_radius_center_guard(points+noise,.12)
                self.assertTrue(g['consistent'])
                self.assertLessEqual(np.linalg.norm(center-g['center'])+radius,g['radius']+1e-12)

    def test_bad_configuration_and_nonfinite_returns_rejected(self):
        for radius,tolerance,error in [(0,.005,.01),(.12,-1,.01),(.12,.005,float('nan'))]:
            with self.assertRaises(ValueError):inline.known_radius_center_guard([[0,0]],radius,tolerance,error)
        with self.assertRaises(ValueError):inline.known_radius_center_guard([[float('nan'),0]],.12)
        with self.assertRaises(ValueError):inline.DynamicEllipseTracker(known_radius=.12)

    def test_inconsistent_cluster_is_flagged_and_union_not_unsafe_fit(self):
        g=inline.known_radius_center_guard([[0,0],[1,0]],.12)
        self.assertFalse(g['consistent']);self.assertEqual(g['method'],'inconsistent_union_bound')
        self.assertGreaterEqual(g['radius'],.76)

    def test_guard_changes_geometry_only_and_never_shrinks_old_axes(self):
        points=np.array([[0,-.04],[.003,0],[0,.04]])
        guard=inline.known_radius_center_guard(points,.12)
        trackers=[inline.DynamicEllipseTracker(geometry_observer='circle',transport_geometry=True,
            known_radius=r) for r in (0.,.12)]
        for k in range(4):
            for tracker in trackers:
                e=inline.Ellipse(.01+k*.003,0,.04,.025,0,3);e.circle_candidate=dict(accepted=False)
                if tracker.known_radius:e.known_radius_guard=dict(guard,center=(np.array(guard['center'])+[k*.003,0]).tolist())
                tracker.update([e],1+k*.1)
        old,new=[t.obstacles(t.snapshot(1.3),1.3)[0] for t in trackers]
        np.testing.assert_array_equal(old['prediction_centers'],new['prediction_centers'])
        np.testing.assert_array_equal(old['prediction_velocities'],new['prediction_velocities'])
        np.testing.assert_array_equal(old['prediction_position_covariances'],new['prediction_position_covariances'])
        self.assertTrue((new['prediction_axes']>=old['prediction_axes']).all())
        self.assertIn('known_radius_guard',new);self.assertNotIn('known_radius_guard',old)
        self.assertEqual(old['position_observed'],new['position_observed'])
        for key in ('x','P'):np.testing.assert_array_equal(trackers[0].tracks[1][key],trackers[1].tracks[1][key])


if __name__=='__main__':unittest.main()
