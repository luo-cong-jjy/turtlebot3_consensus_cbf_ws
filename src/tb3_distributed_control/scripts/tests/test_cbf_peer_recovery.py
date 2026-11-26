"""Moving lane retention must not relax hard CBFs or replace timed goals."""
from pathlib import Path
import sys,unittest,copy
import numpy as np
P=Path(__file__).resolve().parents[2];sys.path[:0]=[str(P/'scripts'),str(P/'纯仿真代码')]
from cbf_mpc_core import default_config
from mpc_dcbf_cached_solver import (peer_recovery_guidance,CachedMPCProblem,rollout,
    warm_spatial_rollouts,delay_compensated_request,MPCSolverManager)


def inputs(peer_x=0.):
    cfg=default_config();cfg.update(spatial_pass=True,execution_backup=True,
        spatial_activation_threshold=.22,spatial_activation_preview_seconds=2.,peer_recovery_corridor=True)
    state=np.array([0.,.45,0.]);goals=np.zeros((11,3));goals[:,0]=.015*np.arange(11)
    centers=np.column_stack((peer_x+.015*np.arange(11),np.zeros(11)))
    obs=[dict(name='peer_2',peer=True,guidance=False,center=centers[0].copy(),a=.2,b=.2,theta=0.,
        prediction_centers=centers,recovery_stamp=10.)]
    memory=dict(progress_previous=np.array([.15,0.]),progress_bounds=[-.22,.22],spatial_summary=dict(mode='return'))
    return cfg,state,goals,obs,memory


class PeerRecoveryTests(unittest.TestCase):
    def test_blocked_return_keeps_moving_lane_with_original_valid_prefix(self):
        cfg,state,goals,obs,memory=inputs();before=goals.copy();shape=obs[0]['prediction_centers'].copy()
        obs[0]['prediction_centers'][:,0]=.01*np.arange(11);shape=obs[0]['prediction_centers'].copy()
        result=peer_recovery_guidance(state,goals,obs,cfg,memory,(np.zeros((10,2)),0.))
        self.assertEqual(result[1],1.);self.assertEqual(memory['peer_recovery_summary']['mode'],'hold_lane')
        self.assertTrue(memory['peer_recovery_summary']['return_deferred'])
        self.assertGreater(memory['progress_plan']['controls'][:,0].mean(),.09)
        self.assertTrue(CachedMPCProblem(cfg,1,False).validate(memory['progress_plan'],state,obs,[-.22,.22],memory['progress_previous']))
        np.testing.assert_array_equal(goals,before);np.testing.assert_array_equal(obs[0]['prediction_centers'],shape)

    def test_longitudinal_coordination_can_open_a_return_without_stopping(self):
        cfg,state,goals,obs,memory=inputs()
        peer_recovery_guidance(state,goals,obs,cfg,memory,(np.zeros((10,2)),0.))
        s=memory['peer_recovery_summary']
        self.assertEqual(s['mode'],'return');self.assertAlmostEqual(s['reference_speed_cap'],.10)
        self.assertGreaterEqual(s['minimum_speed'][s['selected']],.09)
        self.assertTrue(CachedMPCProblem(cfg,1,False).validate(memory['progress_plan'],state,obs,[-.22,.22],memory['progress_previous']))

    def test_clear_lane_starts_return_and_retains_absolute_phase(self):
        cfg,state,goals,obs,memory=inputs(-1.)
        peer_recovery_guidance(state,goals,obs,cfg,memory,(np.zeros((10,2)),0.))
        self.assertEqual(memory['peer_recovery_summary']['mode'],'return');start=memory['peer_recovery_selected']['start']
        u=memory['progress_plan']['controls'][0].copy();state=rollout(state,u[None],.1)[1]
        goals[:,0]+=.015;obs[0]['recovery_stamp']+=.1;obs[0]['prediction_centers'][:,0]+=.015
        memory['progress_previous']=u
        peer_recovery_guidance(state,goals,obs,cfg,memory,(np.zeros((10,2)),0.))
        self.assertEqual(memory['peer_recovery_selected']['start'],start)
        self.assertAlmostEqual(memory['peer_recovery_summary']['phase_age'],.1)

    def test_unsafe_now_never_claims_recovery_candidate_valid(self):
        cfg,state,goals,obs,memory=inputs();obs[0]['prediction_centers'][:,1]=.45
        base=(np.ones((10,2)),.2)
        self.assertIs(peer_recovery_guidance(state,goals,obs,cfg,memory,base),base)
        self.assertEqual(memory['peer_recovery_summary']['mode'],'no_feasible_corridor')
        self.assertNotIn('progress_plan',memory)

    def test_active_external_obstacle_does_not_get_overridden(self):
        cfg,state,goals,obs,memory=inputs();memory['spatial_summary']['mode']='side_shift_hold'
        obs.append(dict(name='ob',center=np.array([.3,.45]),a=.1,b=.1,theta=0.))
        base=(np.ones((10,2)),.3)
        self.assertIs(peer_recovery_guidance(state,goals,obs,cfg,memory,base),base)
        self.assertEqual(memory['peer_recovery_summary']['mode'],'external_avoidance_active')

    def test_compiled_and_numpy_paths_agree_and_clock_is_shifted_not_faked(self):
        cfg,state,goals,obs,memory=inputs(-1.);native=dict(cfg,spatial_compiled_rollout=True);warm_spatial_rollouts(native)
        other=copy.deepcopy(memory)
        a=peer_recovery_guidance(state,goals,obs,cfg,memory,(np.zeros((10,2)),0.))
        b=peer_recovery_guidance(state,goals,obs,native,other,(np.zeros((10,2)),0.))
        np.testing.assert_allclose(a[0],b[0],atol=1e-12)
        task=dict(state=state,goals=goals,obstacles=obs,stamp=10.)
        future=delay_compensated_request(task,np.array([.15,0.]),cfg)
        self.assertAlmostEqual(future['obstacles'][0]['recovery_stamp'],10.1)
        self.assertEqual(obs[0]['recovery_stamp'],10.)

    def test_incompatible_setup_is_explicitly_rejected(self):
        cfg,*_=inputs()
        for values in (dict(spatial_pass=False),dict(spatial_activation_threshold=-1.)):
            with self.assertRaises(ValueError):MPCSolverManager(dict(cfg,**values))

    def test_backward_facing_reference_uses_exact_signed_unicycle_symmetry(self):
        cfg,state,goals,obs,memory=inputs(-1.)
        state[2]=-np.pi+.1;memory['progress_previous']=np.array([-.15,0.])
        base=(np.zeros((10,2)),0.);old=copy.deepcopy(memory)
        self.assertIs(peer_recovery_guidance(state,goals,obs,cfg,old,base),base)
        cfg['peer_recovery_signed_chart']=True
        peer_recovery_guidance(state,goals,obs,cfg,memory,base)
        p=memory['progress_plan'];self.assertTrue(memory['peer_recovery_summary']['reverse_chart'])
        self.assertTrue((p['controls'][:,0]<0).all())
        np.testing.assert_allclose(p['states'],rollout(state,p['controls'],.1),atol=1e-12)
        self.assertTrue(CachedMPCProblem(cfg,1,False).validate(p,state,obs,[-.22,.22],memory['progress_previous']))
        native=dict(cfg,spatial_compiled_rollout=True);warm_spatial_rollouts(native);other=copy.deepcopy(memory)
        peer_recovery_guidance(state,goals,obs,native,other,base)
        np.testing.assert_allclose(other['progress_plan']['states'],p['states'],atol=1e-12)


if __name__=='__main__':unittest.main()
