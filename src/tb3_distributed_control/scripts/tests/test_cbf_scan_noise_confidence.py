"""Per-scan Gaussian union bound and unchanged zero-noise legacy mode."""
import math
import unittest
import numpy as np
from test_cbf_inline_geometry import inline


class ScanNoiseConfidenceTests(unittest.TestCase):
    def test_quantile_meets_joint_raw_scan_tail_without_independence_assumption(self):
        for n in (1,360,1000):
            for alpha in (.1,.01,.001):
                bound=inline.scan_gaussian_error_bound(.01,alpha,n,.015,0.,.01)
                self.assertLessEqual(n*math.erfc(bound['z']/math.sqrt(2.)),alpha)
                self.assertAlmostEqual(bound['effective_point_error'],.01*bound['z']+.015)
        bound=inline.scan_gaussian_error_bound(.01,.01,360,.015)
        self.assertGreater(bound['effective_point_error'],.056)
        self.assertLess(bound['effective_point_error'],.058)

    def test_actual_count_and_valid_probability_required(self):
        for n in (None,0,-1,2.5,True):
            with self.assertRaises(ValueError):inline.scan_gaussian_error_bound(.01,.01,n)
        for alpha in (0,1,float('nan')):
            with self.assertRaises(ValueError):inline.scan_gaussian_error_bound(.01,alpha,360)

    def test_noise_off_retains_exact_guard_and_missing_count_is_not_fabricated(self):
        points=np.array([[0,-.04],[.003,0],[0,.04]])
        self.assertEqual(inline.known_radius_center_guard(points,.12,refine=True),
            inline.known_radius_center_guard(points,.12,refine=True,noise_sigma=0.))
        with self.assertRaises(ValueError):inline.known_radius_center_guard(points,.12,noise_sigma=.01)

    def test_voxel_averages_of_bounded_raw_returns_keep_full_body_containment(self):
        rng=np.random.RandomState(81);center=np.array([.5,-.2]);radius=.12
        bound=inline.scan_gaussian_error_bound(.01,.01,360,.015)['effective_point_error']
        for unused in range(40):
            angles=rng.uniform(-1,1,30);unit=np.column_stack([np.cos(angles),np.sin(angles)])
            error=rng.uniform(-bound,bound,30)
            raw=center+radius*unit+error[:,None]*unit
            points=raw.reshape(10,3,2).mean(1)
            g=inline.known_radius_center_guard(points,.12,refine=True,noise_sigma=.01,raw_ray_count=360,range_error=.015)
            for disk in g.get('enclosing_disks',[g]):
                self.assertLessEqual(np.linalg.norm(center-disk['center'])+radius,disk['radius']+1e-10)


if __name__=='__main__':unittest.main()
