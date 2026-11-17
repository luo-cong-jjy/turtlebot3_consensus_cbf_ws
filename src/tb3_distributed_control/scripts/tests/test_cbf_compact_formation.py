"""Formation feasibility and compact references keep the original safety set."""
from pathlib import Path
import sys
import unittest
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码'),str(Path(__file__).resolve().parent)]
from audit_cbf_pentagon import audit
from test_cbf_spatial_pass import inputs
from mpc_dcbf_cached_solver import continuous_guidance,spatial_lateral_extras,CachedMPCProblem,barrier_values


class CompactFormationChecks(unittest.TestCase):
    def test_radius04_circle_peer_model_and_urdf_bound_are_compatible(self):
        result=audit(.4)
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['adjacent_center_distance'],.4702282018339785)
        self.assertAlmostEqual(result['circular_peer_center_boundary'],.32)
        self.assertLess(result['body_bound']['collision_outer_radius'],.13)

    def test_too_small_formation_is_rejected_not_silently_clipped(self):
        self.assertFalse(audit(.2)['passed'])

    def test_three_compact_levels_change_reference_not_barrier_or_goal(self):
        cfg,state,goals,obs,memory=inputs();old_goals=goals.copy()
        probe=np.column_stack((np.linspace(0,2,20),np.zeros(20),np.zeros(20)))
        before=barrier_values(probe,obs,cfg['margin'])
        cfg['spatial_lateral_extras']=[.03,.06,.10]
        continuous_guidance(state,goals,obs,cfg,memory)
        self.assertEqual(memory['spatial_summary']['lateral_extras'],[.03,.06,.10])
        self.assertEqual(memory['spatial_summary']['candidates'],19)
        np.testing.assert_array_equal(goals,old_goals)
        np.testing.assert_array_equal(before,barrier_values(probe,obs,cfg['margin']))
        self.assertTrue(CachedMPCProblem(cfg,1,False).validate(memory['progress_plan'],state,obs,[-.22,.22],memory['progress_previous']))

    def test_legacy_levels_and_invalid_parameters(self):
        np.testing.assert_array_equal(spatial_lateral_extras({}),[.12,.22,.32])
        for value in ([.1,.2],[.1,.2,.3,.4],[-.01,.1,.2],[.2,.1,.3],[.01,float('nan'),.2]):
            with self.assertRaises(ValueError):spatial_lateral_extras(dict(spatial_lateral_extras=value))
