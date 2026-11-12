"""Reference change must not manufacture a lateral route or alter metrics."""
from pathlib import Path
import sys
import unittest
import numpy as np
PACKAGE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PACKAGE / 'scripts'), str(PACKAGE / '纯仿真代码')]
from cbf_mpc_core import nearest_line_reference, metric_definitions


class NearestReferenceChecks(unittest.TestCase):
    def test_projection_preserves_cross_track_error_and_reference_spacing(self):
        angle = .7
        tangent = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-tangent[1], tangent[0]])
        goals = np.column_stack((np.array([2., -1.]) + np.arange(26)[:, None] * .015 * tangent,
                                 np.full(26, angle)))
        state = np.r_[goals[0, :2] - .8 * tangent + .45 * normal, -.3]
        frozen = goals.copy()
        local = nearest_line_reference(state, goals, .1)
        self.assertAlmostEqual(np.dot(local[0, :2] - state[:2], tangent), 0.)
        self.assertAlmostEqual(np.dot(local[0, :2] - state[:2], normal), -.45)
        np.testing.assert_allclose(np.diff(local[:, :2], axis=0), np.diff(goals[:, :2], axis=0), atol=1e-14)
        np.testing.assert_array_equal(goals, frozen)
        self.assertAlmostEqual(local[-1, 2], angle)

    def test_time_translation_does_not_shift_the_nearest_path_reference(self):
        goals = np.zeros((11, 3)); goals[:, 0] = 2. + np.arange(11) * .015
        later = goals.copy(); later[:, 0] += .3
        state = [1., .4, .9]
        np.testing.assert_allclose(nearest_line_reference(state, goals, .1),
                                   nearest_line_reference(state, later, .1), atol=1e-14)
        self.assertIn('leader reference at the same stamp', metric_definitions()['position_error'])

    def test_invalid_or_curved_references_are_not_silently_projected(self):
        goals = np.zeros((11, 3)); goals[:, 0] = np.arange(11) * .015
        curve = goals.copy(); curve[:, 1] = np.arange(11) ** 2 * .001
        for state, nodes, dt in [([0, 0, 0], curve, .1), ([0, 0, 0], goals, 0),
                                  ([np.nan, 0, 0], goals, .1), ([0, 0, 0], goals[:1], .1)]:
            with self.assertRaises(ValueError):
                nearest_line_reference(state, nodes, dt)

    def test_stationary_reference_is_retained_without_invented_motion(self):
        hold = np.tile([.2, .3, .7], (11, 1))
        np.testing.assert_array_equal(nearest_line_reference([1, 2, 0], hold, .1), hold)
