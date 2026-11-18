"""Early return must be screened, continuous at entry and optional."""
import copy
import unittest
import numpy as np
from test_cbf_spatial_pass import inputs
from mpc_dcbf_cached_solver import (continuous_guidance,CachedMPCProblem,
    MPCSolverManager,warm_spatial_rollouts)


class QualifiedReturnChecks(unittest.TestCase):
    def setup_case(self):
        cfg,state,goals,obs,memory=inputs()
        state[:]=[.6,.6,0.];goals[:,0]=.6+.015*np.arange(11)
        obs[0].update(center=np.array([.55,0.]),a=.2,b=.2)
        obs[0]['prediction_centers']=np.column_stack((.55-.006*np.arange(11),np.zeros(11)))
        obs[0]['prediction_axes']=np.full((11,2),.2)
        obs[0]['guidance_motion']['models'][0].update(position=[.55,0.],velocity=[-.06,0.])
        memory.update(spatial_time=4.,spatial_selected=dict(name='actor',start=0.,y0=0.,dy0=0.,
            target=.6,duration=3.,returning=False,side=1),
            spatial_return_departure=dict(name='actor',count=1))
        return cfg,state,goals,obs,memory

    def test_separating_robot_can_return_before_full_longitudinal_pass(self):
        cfg,state,goals,obs,memory=self.setup_case()
        legacy=copy.deepcopy(memory);continuous_guidance(state,goals,obs,cfg,legacy)
        self.assertFalse(legacy['spatial_selected']['returning'])
        cfg['spatial_predictive_return']=True
        continuous_guidance(state,goals,obs,cfg,memory)
        self.assertTrue(memory['spatial_summary']['early_return_started'])
        self.assertTrue(memory['spatial_summary']['selected_return_qualified'])
        self.assertLessEqual(memory['spatial_selected']['duration'],6.)
        self.assertTrue(CachedMPCProblem(cfg,1,False).validate(memory['progress_plan'],state,obs,
            [-.22,.22],memory['progress_previous']))

    def test_approaching_target_does_not_enable_early_return(self):
        cfg,state,goals,obs,memory=self.setup_case();cfg['spatial_predictive_return']=True
        state[0]=-.1;goals[:,0]=-.1+.015*np.arange(11)
        continuous_guidance(state,goals,obs,cfg,memory)
        self.assertFalse(memory['spatial_summary']['return_gate']['eligible'])
        self.assertEqual(memory['spatial_summary']['return_candidates'],0)

    def test_future_obstacle_blocks_shortcut_even_when_a_prefix_is_feasible(self):
        cfg,state,goals,obs,memory=self.setup_case()
        cfg.update(spatial_predictive_return=True,spatial_prefix_retention=True)
        obs.append(dict(name='return_wall',center=np.array([1.2,.1]),a=1.1,b=.25,theta=0.))
        continuous_guidance(state,goals,obs,cfg,memory)
        self.assertEqual(memory['spatial_summary']['return_full_qualified'],0)
        self.assertFalse(memory['spatial_selected']['returning'])

    def test_extra_candidate_native_graph_matches_numpy_and_preserves_entry_velocity(self):
        cfg,state,goals,obs,memory=self.setup_case();cfg['spatial_predictive_return']=True
        state[2]=.3
        a_memory=copy.deepcopy(memory);a=continuous_guidance(state,goals,obs,cfg,a_memory)
        warm_spatial_rollouts(cfg);cfg['spatial_compiled_rollout']=True
        b_memory=copy.deepcopy(memory);b=continuous_guidance(state,goals,obs,cfg,b_memory)
        np.testing.assert_allclose(a[0],b[0],atol=2e-12,rtol=0)
        self.assertEqual(a[1],b[1]);self.assertIn(b_memory['spatial_summary']['candidates'],(23,24))
        self.assertAlmostEqual(b_memory['spatial_selected']['y0'],.6)
        self.assertAlmostEqual(b_memory['spatial_selected']['dy0'],.15*np.sin(.3))

    def test_bad_return_times_rejected_at_startup(self):
        cfg=self.setup_case()[0]
        for times in ([0,3,6],[4,3,6],[3,6],[3,float('nan'),6]):
            cfg['spatial_return_durations']=times
            with self.assertRaises(ValueError):MPCSolverManager(cfg)
