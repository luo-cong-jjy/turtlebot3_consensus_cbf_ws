"""0.22 is an auxiliary-guidance activation h, not clearance or safety removal."""
from pathlib import Path
import sys
import unittest
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码'),str(Path(__file__).resolve().parent)]
from cbf_mpc_core import default_config
from test_cbf_spatial_pass import inputs
from mpc_dcbf_cached_solver import (spatial_activation_measurement,continuous_guidance,
    CachedMPCProblem,MPCSolverManager,barrier_values,rollout)


class Trigger022Checks(unittest.TestCase):
    def setup_gate(self,center=.58):
        cfg=default_config();cfg.update(spatial_pass=True,execution_backup=True,
            spatial_activation_threshold=.22,spatial_activation_preview_seconds=1.)
        state=np.zeros(3);goals=np.zeros((11,3));goals[:,0]=.015*np.arange(11)
        obs=dict(name='actor',center=np.array([center,0.]),a=.16,b=.16,theta=0.,
            prediction_centers=np.column_stack((center-.006*np.arange(11),np.zeros(11))))
        memory=dict(progress_previous=np.array([.15,0.]),progress_bounds=[-.22,.22])
        return cfg,state,goals,[obs],memory

    def test_predicted_h_can_trigger_before_current_h_crosses022(self):
        cfg,state,goals,obs,memory=self.setup_gate()
        r=spatial_activation_measurement(state,goals,obs,cfg,memory)
        self.assertGreater(r['current_min_h'],.22);self.assertLess(r['predicted_min_h'],.22)
        self.assertTrue(r['triggered']);self.assertEqual(r['predicted_steps'],10)

    def test_current_boundary_inclusive_and_far_reference_inactive(self):
        cfg,state,goals,obs,memory=self.setup_gate(.50);cfg['spatial_activation_preview_seconds']=0.
        self.assertTrue(spatial_activation_measurement(state,goals,obs,cfg,memory)['triggered'])
        obs[0]['center']=np.array([5.,0.]);obs[0]['prediction_centers']=np.tile([5.,0.],(11,1))
        result=continuous_guidance(state,goals,obs,cfg,memory)
        self.assertEqual(result[1],0.);self.assertEqual(memory['spatial_summary']['mode'],'activation_inactive')
        self.assertNotIn('progress_plan',memory)

    def test_peer_does_not_trigger_external_guide(self):
        cfg,state,goals,obs,memory=self.setup_gate(.20);obs[0]['peer']=True
        result=spatial_activation_measurement(state,goals,obs,cfg,memory)
        self.assertFalse(result['triggered']);self.assertIsNone(result['current_min_h'])

    def test_inactive_guide_keeps_original_hard_dcbf_rate(self):
        cfg,state,goals,obs,memory=self.setup_gate(.55)
        state[2]=np.pi;goals[:,0]=-.015*np.arange(11);goals[:,2]=np.pi
        obs[0]['a']=obs[0]['b']=.2;obs[0]['prediction_centers']=np.tile([.55,0.],(11,1))
        result=continuous_guidance(state,goals,obs,cfg,memory);self.assertEqual(result[1],0.)
        u=np.tile([-.22,0.],(10,1));states=rollout(state,u,.1)
        self.assertGreater(barrier_values(states,obs,.12).min(),0.)
        self.assertFalse(CachedMPCProblem(cfg,1,False).validate(dict(states=states,controls=u,
            plan_controls=np.zeros((10,2))),state,obs,[-.22,.22],u[0]))

    def test_disabled_mode_matches_original_reference_and_latch_survives_distance_recovery(self):
        cfg,state,goals,obs,memory=inputs()
        from mpc_dcbf_cached_solver import _spatial_pass_guidance_active
        import copy
        a=continuous_guidance(state,goals,obs,cfg,copy.deepcopy(memory))
        b=_spatial_pass_guidance_active(state,goals,obs,cfg,copy.deepcopy(memory))
        np.testing.assert_array_equal(a[0],b[0])
        cfg.update(spatial_activation_threshold=.22,spatial_activation_preview_seconds=1.)
        memory.update(spatial_activation_active=True,spatial_selected=dict(name='actor',start=0.,
            y0=0.,dy0=0.,target=.5,duration=6.,returning=False,side=1))
        continuous_guidance(state,goals,obs,cfg,memory)
        self.assertTrue(memory['spatial_summary']['activation']['active'])
        self.assertTrue(memory['spatial_summary']['activation']['latched'])

    def test_window_cannot_exceed_existing_soft_preview(self):
        cfg,state,goals,obs,memory=self.setup_gate();cfg['spatial_activation_preview_seconds']=9.
        with self.assertRaises(ValueError):MPCSolverManager(cfg)

    def test_extended_trigger_is_soft_and_keeps_hard_window_unchanged(self):
        cfg,state,goals,obs,memory=self.setup_gate()
        cfg['spatial_activation_preview_seconds']=2.
        result=spatial_activation_measurement(state,goals,obs,cfg,memory)
        self.assertEqual(result['predicted_steps'],20)
        self.assertEqual(result['hard_steps'],10);self.assertEqual(cfg['horizon'],10)
        self.assertEqual(result['prediction_scope'],'soft_mean_extension_held_hard_axes')
        self.assertLess(result['predicted_min_h'],result['current_min_h'])

    def test_shift_timing_validation_at_startup(self):
        cfg=default_config()
        for values in ([0,3,4],[3,2,4],[2,3],[2,float('nan'),4]):
            cfg['spatial_shift_durations']=values
            with self.assertRaises(ValueError):MPCSolverManager(cfg)

    def test_fast_timing_keeps_native_and_numpy_reference_equal(self):
        import copy
        from mpc_dcbf_cached_solver import warm_spatial_rollouts
        cfg,state,goals,obs,memory=inputs();cfg['spatial_shift_durations']=[2.,3.,4.]
        a=continuous_guidance(state,goals,obs,cfg,copy.deepcopy(memory))
        warm_spatial_rollouts(cfg);cfg['spatial_compiled_rollout']=True
        b=continuous_guidance(state,goals,obs,cfg,copy.deepcopy(memory))
        np.testing.assert_allclose(a[0],b[0],atol=1e-12,rtol=0)
        self.assertEqual(a[1],b[1])
