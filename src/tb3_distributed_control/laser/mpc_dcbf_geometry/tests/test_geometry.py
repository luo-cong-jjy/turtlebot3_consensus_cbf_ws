#!/usr/bin/env python3
"""Geometry regressions independent of a ROS master or Gazebo."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpc_dcbf_geometry_core import (EllipseTracker, dbscan, enclosing_ellipse,
                                    scan_points, transform_points, voxel_downsample)
from geometry_rviz import render_config


class GeometryTests(unittest.TestCase):
    def assert_enclosed(self, points, ellipse):
        c, s = np.cos(ellipse.theta), np.sin(ellipse.theta)
        local = (points - [ellipse.cx, ellipse.cy]) @ np.array([[c, -s], [s, c]])
        radius_squared = np.sum((local / [ellipse.a, ellipse.b])**2, axis=1)
        self.assertLessEqual(float(radius_squared.max()), 1.0 + 1e-8)

    def test_scan_invalid_and_max_range_returns(self):
        p = scan_points([np.nan, np.inf, 0, 0.1, 1, 3.5], 0, np.pi/2, 0.12, 3.5)
        self.assertEqual(p.shape, (1, 2))
        np.testing.assert_allclose(p[0], [1, 0], atol=1e-12)

    def test_transform_includes_laser_offset_and_orientation(self):
        p = transform_points([[1, 0]], [2, 3, 0.172], [0, 0, np.sqrt(0.5), np.sqrt(0.5)])
        np.testing.assert_allclose(p, [[2, 4, 0.172]], atol=1e-12)

    def test_voxel_centroids(self):
        result = voxel_downsample([[0.01, 0.02], [0.03, 0.04], [1, 1]], 0.1)
        np.testing.assert_allclose(result, [[0.02, 0.03], [1, 1]])

    def test_dbscan_two_objects_and_noise(self):
        p = np.array([[0, 0], [.03, 0], [.06, 0], [1, 1], [1.03, 1], [1.06, 1], [5, 5]])
        labels = dbscan(p, .08, 3)
        np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1, -1])

    def test_cluster_crosses_scan_zero_angle(self):
        angles = np.deg2rad([358, 359, 0, 1, 2])
        p = np.column_stack((np.cos(angles), np.sin(angles)))
        np.testing.assert_array_equal(dbscan(p, .04, 3), np.zeros(5))

    def test_empty_input(self):
        self.assertEqual(dbscan(np.empty((0, 2))).size, 0)
        self.assertEqual(transform_points(np.empty((0, 2)), [0, 0, 0], [0, 0, 0, 1]).shape, (0, 3))

    def test_rotated_ellipse_axes_center_and_enclosure(self):
        t = np.linspace(0, 2*np.pi, 80, endpoint=False)
        angle = .6
        c, s = np.cos(angle), np.sin(angle)
        p = np.column_stack((2*np.cos(t), .5*np.sin(t))) @ np.array([[c, s], [-s, c]]) + [4, -3]
        ellipse = enclosing_ellipse(p, padding=0)
        self.assertAlmostEqual(ellipse.cx, 4, places=6)
        self.assertAlmostEqual(ellipse.cy, -3, places=6)
        self.assertAlmostEqual(ellipse.a, 2, places=5)
        self.assertAlmostEqual(ellipse.b, .5, places=5)
        self.assertAlmostEqual(ellipse.theta, angle, places=5)
        self.assert_enclosed(p, ellipse)

    def test_rectangle_minimum_ellipse(self):
        p = np.array([[-1, -.5], [-1, .5], [1, -.5], [1, .5]])
        ellipse = enclosing_ellipse(p, padding=0)
        self.assertAlmostEqual(ellipse.a, np.sqrt(2), places=5)
        self.assertAlmostEqual(ellipse.b, np.sqrt(.5), places=5)
        self.assert_enclosed(p, ellipse)

    def test_lines_and_duplicate_points(self):
        for p in [np.array([[1, 1]] * 4), np.array([[0, 0], [1, 1], [2, 2]]),
                  np.array([[0, 0], [0, 2]])]:
            ellipse = enclosing_ellipse(p)
            self.assertTrue(np.isfinite([ellipse.cx, ellipse.cy, ellipse.a, ellipse.b, ellipse.theta]).all())
            self.assertGreaterEqual(ellipse.a, ellipse.b)
            self.assertGreater(ellipse.b, 0)
            self.assert_enclosed(p, ellipse)

    def test_irregular_points_all_enclosed_even_at_iteration_cap(self):
        rng = np.random.RandomState(7)
        p = rng.normal(size=(100, 2)) * [2, .1] + [1e5, -2e5]
        self.assert_enclosed(p, enclosing_ellipse(p, max_iterations=2))

    def test_hungarian_matching_survives_order_change(self):
        tracker = EllipseTracker(max_distance=.5)
        first = [enclosing_ellipse([[0, 0]]), enclosing_ellipse([[2, 0]])]
        tracker.update(first, 1.)
        second = [enclosing_ellipse([[2.03, 0]]), enclosing_ellipse([[.02, 0]])]
        tracker.update(second, 1.1)
        self.assertEqual([e.track_id for e in second], [2, 1])

    def test_tracks_expire_and_clock_rewind_resets_history(self):
        tracker = EllipseTracker(timeout=.5)
        a = tracker.update([enclosing_ellipse([[0, 0]])], 1.)[0]
        tracker.update([], 2.)
        self.assertEqual(len(tracker.tracks), 0)
        b = tracker.update([enclosing_ellipse([[0, 0]])], 2.1)[0]
        self.assertNotEqual(a.track_id, b.track_id)
        c = tracker.update([enclosing_ellipse([[0, 0]])], .1)[0]
        self.assertEqual(c.track_id, 1)
        self.assertEqual(len(tracker.tracks), 1)

    def test_static_shape_variance_drops_after_warmup(self):
        tracker = EllipseTracker()
        for t in [1., 1.1, 1.2]:
            ellipse = tracker.update([enclosing_ellipse([[0, 0], [1, 0]])], t)[0]
        self.assertAlmostEqual(ellipse.variance, 0.)

    def test_custom_namespace_rviz_topics_model_and_frame(self):
        template = Path(__file__).resolve().parents[1] / 'geometry.rviz'
        config = render_config(template, 'robot_test', 'map')
        manager = config['Visualization Manager']
        self.assertEqual(manager['Global Options']['Fixed Frame'], 'map')
        model = next(d for d in manager['Displays'] if d['Class'] == 'rviz/RobotModel')
        self.assertEqual(model['TF Prefix'], 'robot_test')
        self.assertEqual(model['Robot Description'], '/robot_test/robot_description')
        scan = next(d for d in manager['Displays'] if d['Class'] == 'rviz/LaserScan')
        self.assertEqual(scan['Topic'], '/robot_test/scan')


if __name__ == '__main__':
    unittest.main()
