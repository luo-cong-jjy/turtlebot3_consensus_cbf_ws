"""Native reference evaluation preserves recurrence; carry is freshly checked."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock,patch
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
from cbf_mpc_core import default_config
import mpc_dcbf_cached_solver as core


class ReferenceRuntimeChecks(unittest.TestCase):
    def test_completion_reserve_moves_nlp_deadline_earlier_without_changing_validation(self):
        cfg=default_config();cfg.update(spatial_pass=True,execution_backup=True,realtime=True,
            fallback_reserve=.035,nlp_wall_time=.065,manager_wall_time=.085)
        manager=core.MPCSolverManager(cfg);problem=core.CachedMPCProblem(cfg,0,False)
        problem.solve=Mock(side_effect=RuntimeError('forced NLP failure'))
        manager.problems[(0,False)]=problem
        u=np.tile([.15,0.],(10,1));goals=np.zeros((11,3));goals[:,0]=.015*np.arange(11)
        begin=100.
        with patch.object(core.time,'perf_counter',return_value=begin),\
                patch.object(core,'continuous_guidance',return_value=(np.zeros((10,2)),0.)):
            result=manager.solve(np.zeros(3),goals,[],[-.22,.22],dict(controls=u,plan_controls=u),
                'robot',applied_previous=u[0],deadline=begin+.097)
        requested=problem.solve.call_args.kwargs['deadline']
        self.assertAlmostEqual(requested,begin+.050,places=10)
        self.assertTrue(problem.validate(result,np.zeros(3),[],[-.22,.22],u[0]))

    def test_native_graph_matches_numpy_for_sizes_return_phases_and_delta_bounds(self):
        rng=np.random.default_rng(117)
        for count,delta in [(19,False),(20,True),(21,False)]:
            cfg=default_config();cfg['delta_limit']=delta;steps=80;dt=.1;times=np.arange(steps+1)*dt
            forward=np.array([np.cos(.3),np.sin(.3)]);normal=np.array([-forward[1],forward[0]])
            targets=np.array([.4,0.])+times[:,None]*.15*forward
            # A passed target exercises the capture/hold/return state, including
            # descriptors which are already in the return phase.
            center=np.tile([-1.,0.],(steps+1,1));support=np.full(steps+1,.35)
            start=-rng.uniform(0.,10.,count);duration=rng.choice([4.5,6.,7.5],count)
            y0=rng.normal(0.,.1,count);dy0=rng.normal(0.,.02,count);target=rng.choice([-.5,.5],count)
            returning=rng.random(count)>.5
            args=(np.array([.4,.1,.3]),np.array([.15,.2]),targets,center,support,forward,normal,.15,0.,
                  start,duration,y0,dy0,target,returning,[-.22,.22],cfg,steps)
            a=core._spatial_rollout_numpy(*args);b=core._spatial_rollout_compiled(*args)
            for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=2e-12,rtol=0)

    def test_online_call_cannot_construct_a_cold_graph(self):
        with self.assertRaisesRegex(core.PreviewBudgetExceeded,'not warmed'):
            core._spatial_rollout_function(7,9,.1,allow_build=False)

    def test_carried_prefix_is_reanchored_and_rejected_after_an_unsafe_model_update(self):
        cfg=default_config();state=np.array([.1,.05,0.]);u=np.tile([.15,0.],(10,1))
        problem=core.CachedMPCProblem(cfg,1,False)
        far=dict(name='actor',center=np.array([2.,0.]),a=.2,b=.2,theta=0.)
        good=core.checked_spatial_guidance_carry(problem,state,[far],[-.22,.22],u[0],u,cfg)
        self.assertIsNotNone(good);np.testing.assert_array_equal(good['states'][0],state)
        near=dict(far,center=np.array([.2,.05]))
        self.assertIsNone(core.checked_spatial_guidance_carry(problem,state,[near],[-.22,.22],u[0],u,cfg))
        self.assertIsNone(core.checked_spatial_guidance_carry(problem,state,[far],[-.1,.1],u[0],u,cfg))

    def test_preview_timeout_uses_only_a_currently_validated_carry_when_enabled(self):
        for enabled in (False,True):
            cfg=default_config();cfg.update(spatial_pass=True,execution_backup=True,realtime=True,
                spatial_timeout_carry=enabled,nlp_wall_time=.1,manager_wall_time=.3)
            manager=core.MPCSolverManager(cfg);problem=core.CachedMPCProblem(cfg,0,False)
            problem.solve=Mock(side_effect=RuntimeError('forced solve failure'))
            manager.problems[(0,False)]=problem
            u=np.tile([.15,0.],(10,1));last=dict(controls=u,plan_controls=u.copy(),validated_steps=10)
            goals=np.zeros((11,3));goals[:,0]=.015*np.arange(11)
            with patch.object(core,'continuous_guidance',side_effect=core.PreviewBudgetExceeded('forced preview timeout')):
                result=manager.solve(np.zeros(3),goals,[],[-.22,.22],last,'robot',applied_previous=u[0])
            record=manager.last_records['robot']
            self.assertTrue(record['preview_budget_hit'])
            self.assertEqual(record['guide_level'],float(enabled))
            self.assertEqual(record.get('spatial_timeout_carry',False),enabled)
            self.assertTrue(problem.validate(result,np.zeros(3),[],[-.22,.22],u[0]))
