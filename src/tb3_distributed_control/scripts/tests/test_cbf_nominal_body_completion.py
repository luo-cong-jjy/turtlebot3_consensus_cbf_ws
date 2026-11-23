"""Known-body nominal completion without whole-scan probability claims."""
import unittest
import numpy as np
from test_cbf_inline_geometry import inline
from mpc_dcbf_static_obstacles import reference_covariance_matrices,reference_axis_inflation
inline.reference_covariance_matrices=reference_covariance_matrices
inline.reference_axis_inflation=reference_axis_inflation


class NominalBodyTests(unittest.TestCase):
    def test_sparse_same_body_arcs_are_completed_and_measured_ellipse_preserved(self):
        center=np.array([1.,.2]);radius=.12
        for count in (3,5,15):
            angles=np.linspace(2.7,3.6,count);p=center+radius*np.column_stack([np.cos(angles),np.sin(angles)])
            e=inline.enclosing_ellipse(p,.025,.03,.01,100)
            g=inline.nominal_body_completion(p,e,radius,.03)
            self.assertGreaterEqual(g['radius']+1e-12,np.linalg.norm(center-g['center'])+radius)
            self.assertFalse(g['probability_claim'])
            phi=np.arange(1024)*2*np.pi/1024;c,s=np.cos(e.theta),np.sin(e.theta)
            boundary=np.column_stack([e.a*np.cos(phi),e.b*np.sin(phi)])@np.array([[c,s],[-s,c]])+[e.cx,e.cy]
            self.assertLessEqual(np.linalg.norm(boundary-g['center'],axis=1).max(),g['radius']+1e-9)

    def test_geometry_only_and_exclusive_with_experimental_confidence_guard(self):
        p=np.array([[0.,-.04],[.003,0.],[0.,.04]])
        trackers=[inline.DynamicEllipseTracker(geometry_observer='circle',transport_geometry=True,
            nominal_body_radius=r) for r in (0.,.12)]
        for k in range(4):
            for tracker in trackers:
                e=inline.Ellipse(.01+k*.003,0,.04,.025,0,3);e.circle_candidate=dict(accepted=False)
                if tracker.nominal_body_radius:e.nominal_body_guard=inline.nominal_body_completion(p+[k*.003,0],e,.12)
                tracker.update([e],1+k*.1)
        old,new=[t.obstacles(t.snapshot(1.3),1.3)[0] for t in trackers]
        np.testing.assert_array_equal(old['prediction_centers'],new['prediction_centers'])
        self.assertTrue((new['prediction_axes']>=old['prediction_axes']).all())
        for key in ('x','P'):np.testing.assert_array_equal(trackers[0].tracks[1][key],trackers[1].tracks[1][key])
        self.assertIn('nominal_body_guard',new);self.assertNotIn('known_radius_guard',new)
        with self.assertRaises(ValueError):inline.DynamicEllipseTracker(geometry_observer='circle',transport_geometry=True,known_radius=.12,nominal_body_radius=.12)




class FixedRadiusObservationTests(unittest.TestCase):
    def test_three_point_known_radius_recovers_nominal_centre(self):
        center=np.array([1.,.1]);a=np.linspace(2.7,3.6,3);points=center+.12*np.column_stack([np.cos(a),np.sin(a)])
        e=inline.enclosing_ellipse(points,.025,.03,.01,100);r=inline.fixed_radius_observation(points,e,[0,0],.12)
        self.assertTrue(r['accepted']);np.testing.assert_allclose(r['center'],center,atol=1e-6)
        self.assertFalse(r['probability_claim']);self.assertGreaterEqual(r['enclosure_radius'],.15)

    def test_degenerate_or_inconsistent_points_are_not_a_forced_circle(self):
        for points in ([[1,0]]*3,[[1,0],[1,.3],[1,.6]]):
            e=inline.Ellipse(1,0,.1,.03,0,3);r=inline.fixed_radius_observation(points,e,[0,0],.12)
            self.assertFalse(r['accepted'])


if __name__=='__main__':unittest.main()
