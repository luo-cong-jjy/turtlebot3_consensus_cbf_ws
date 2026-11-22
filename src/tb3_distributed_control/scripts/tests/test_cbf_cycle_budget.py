"""Numerical validation equivalence, packed indexing and budget admission."""
from pathlib import Path
import sys,time
import unittest
from unittest.mock import patch,Mock
import numpy as np
P=Path(__file__).resolve().parents[2];sys.path[:0]=[str(P/'scripts'),str(P/'纯仿真代码')]
from cbf_mpc_core import default_config
import mpc_dcbf_cached_solver as core


class CycleBudgetTests(unittest.TestCase):
    def config(self,**values):
        c=default_config();c.update(cycle_budget_guard=True,realtime=True,execution_backup=True,
            fallback_reserve=.035,manager_wall_time=.085,nlp_wall_time=.065);c.update(values);return c

    def test_packed_extraction_matches_matrix_values_in_both_solution_and_debug(self):
        cfg=self.config();problem=core.CachedMPCProblem(cfg,1,False)
        legacy=core.CachedMPCProblem(dict(cfg,cycle_budget_guard=False),1,False)
        self.assertEqual((problem.opti.nx,problem.opti.ng,problem.opti.np),(legacy.opti.nx,legacy.opti.ng,legacy.opti.np))
        goals=np.zeros((11,3));goals[:,0]=.015*np.arange(11);u=np.tile([.15,.01],(10,1));q=np.tile([.15,0.],(10,1))
        obs=[dict(name='far',center=np.array([4.,2.]),a=.2,b=.15,theta=.2)]
        for deadline in (time.perf_counter()+1.,time.perf_counter()-1.):
            result=problem.solve(np.zeros(3),goals,obs,[-.22,.22],u[0],u,q,deadline=deadline)
            value=problem.opti.debug.value;v=np.asarray(value(problem.solution_pack)).reshape(-1);n=10
            for unpacked,expression in [(v[:2*n].reshape((n,2),order='F'),problem.u),
                    (v[2*n:4*n].reshape((n,2),order='F'),problem.q),(v[4*n:-1].reshape((n+1,3),order='F'),problem.x)]:
                np.testing.assert_array_equal(unpacked,np.asarray(value(expression)))
            self.assertAlmostEqual(float(v[-1]),float(value(problem.opti.f)))
            self.assertTrue(problem.validate(result,np.zeros(3),obs,[-.22,.22],u[0]))

    def test_capture_is_output_only_and_cannot_bypass_unsafe_candidate(self):
        cfg=self.config();p=core.CachedMPCProblem(cfg,1,False);state=np.zeros(3);u=np.tile([.15,0.],(10,1))
        for center in ([4.,0.],[.2,0.]):
            obs=[dict(name='ob',center=np.asarray(center),a=.15,b=.15,theta=0.)]
            for corruption in ('none','speed','dynamics','plan','nan'):
                candidate=dict(controls=u.copy(),states=core.rollout(state,u,.1),plan_controls=u.copy())
                if corruption=='speed':
                    candidate['controls'][0,0]=.5
                    candidate['states']=core.rollout(state,candidate['controls'],.1)
                if corruption=='dynamics':candidate['states'][1,0]+=.01
                if corruption=='plan':candidate['plan_controls'][0,0]=.5
                if corruption=='nan':candidate['states'][0,0]=np.nan
                expected=p.validate(candidate,state,obs,[-.22,.22],u[0]);capture={'values':np.full((1,11),100.)}
                self.assertEqual(expected,p.validate(candidate,state,obs,[-.22,.22],u[0],_diagnostics=capture))
                if corruption!='none':self.assertFalse(expected)

    def test_work_deadline_keeps_finish_reserve_and_skips_nlp_after_input_delay(self):
        cfg=self.config();m=core.MPCSolverManager(cfg);p=core.CachedMPCProblem(cfg,0,False)
        p.solve=Mock(side_effect=RuntimeError('test'));m.problems[(0,False)]=p
        goals=np.zeros((11,3));goals[:,0]=.015*np.arange(11);u=np.tile([.15,0.],(10,1))
        with patch.object(core.time,'perf_counter',return_value=100.),patch.object(core,'continuous_guidance',return_value=(u,0.)):
            result=m.solve(np.zeros(3),goals,[],[-.22,.22],dict(controls=u,plan_controls=u),'r',deadline=100.020,applied_previous=u[0])
        p.solve.assert_not_called();self.assertTrue(p.validate(result,np.zeros(3),[],[-.22,.22],u[0]))
        b=m.last_records['r']['cycle_budget'];self.assertAlmostEqual(b['work_deadline'],100.012)

    def test_postprocess_spike_ages_out_even_when_nlp_is_skipped(self):
        cfg=self.config();m=core.MPCSolverManager(cfg);p=core.CachedMPCProblem(cfg,0,False)
        p.solve=Mock(side_effect=RuntimeError('test'));m.problems[(0,False)]=p
        goals=np.zeros((11,3));u=np.tile([.15,0.],(10,1));m.robot_memory['r']=dict(positions=core.deque(maxlen=11),cooldown=0,preferred=1,hold=0,postprocess_history=core.deque([.2],maxlen=16))
        with patch.object(core.time,'perf_counter',return_value=100.),patch.object(core,'continuous_guidance',return_value=(u,0.)):
            for unused in range(15):m.solve(np.zeros(3),goals,[],[-.22,.22],dict(controls=u,plan_controls=u),'r',deadline=100.020,applied_previous=u[0])
            self.assertGreater(m.last_records['r']['cycle_budget']['postprocess_reserve_ms'],200.)
            m.solve(np.zeros(3),goals,[],[-.22,.22],dict(controls=u,plan_controls=u),'r',deadline=100.020,applied_previous=u[0])
        self.assertAlmostEqual(m.last_records['r']['cycle_budget']['postprocess_reserve_ms'],35.)



class FeasibleCostAdmissionTests(unittest.TestCase):
    def config(self,**values):
        c=default_config();c.update(cycle_budget_guard=True,budget_objective_gap=.045,realtime=True,
            execution_backup=True,fallback_reserve=.035,manager_wall_time=.085,nlp_wall_time=.065);c.update(values);return c

    def test_low_cost_full_prefix_finishes_without_native_nlp(self):
        cfg=self.config();m=core.MPCSolverManager(cfg);p=core.CachedMPCProblem(cfg,0,False)
        p.solve=Mock(side_effect=RuntimeError('must not run'));m.problems[(0,False)]=p
        goals=np.zeros((11,3));goals[:,0]=.015*np.arange(11);u=np.tile([.15,0.],(10,1))
        with patch.object(core,'continuous_guidance',return_value=(np.zeros((10,2)),0.)):
            result=m.solve(np.zeros(3),goals,[],[-.22,.22],dict(controls=u,plan_controls=u),'r',deadline=time.perf_counter()+.5,applied_previous=u[0])
        p.solve.assert_not_called();self.assertFalse(result['success'])
        self.assertTrue(p.validate(result,np.zeros(3),[],[-.22,.22],u[0]));self.assertEqual(result['validated_steps'],10)
        c=m.last_records['r']['cycle_budget']['objective_certificate'];self.assertLessEqual(c['gap_upper_bound'],.045)
        self.assertGreater(c['gap_upper_bound'],0.);self.assertEqual(c['lower_bound'],0.)

    def test_high_cost_and_negative_weights_do_not_claim_a_certificate(self):
        for changes in ({},{'q_plan':[-1.,20.]}):
            cfg=self.config(**changes);m=core.MPCSolverManager(cfg);p=core.CachedMPCProblem(cfg,0,False)
            p.solve=Mock(side_effect=RuntimeError('forced'));m.problems[(0,False)]=p
            goals=np.zeros((11,3));goals[:,0]=.015*np.arange(11)+(0. if changes else .1);u=np.tile([.15,0.],(10,1))
            with patch.object(core.time,'perf_counter',return_value=100.),patch.object(core,'continuous_guidance',return_value=(np.zeros((10,2)),0.)):
                m.solve(np.zeros(3),goals,[],[-.22,.22],dict(controls=u,plan_controls=u),'r',deadline=100.097,applied_previous=u[0])
            p.solve.assert_called_once();self.assertIsNone(m.last_records['r']['cycle_budget']['objective_certificate'])

    def test_one_step_only_and_active_guidance_cannot_qualify(self):
        cfg=self.config();candidate=dict(validated_steps=1,cost=0.)
        self.assertIsNone(core.feasible_objective_certificate(candidate,cfg,(np.zeros((10,2)),0.)))
        candidate['validated_steps']=10
        self.assertIsNone(core.feasible_objective_certificate(candidate,cfg,(np.zeros((10,2)),1.)))
        self.assertIsNone(core.feasible_objective_certificate(candidate,dict(cfg,prediction_risk=True),(np.zeros((10,2)),0.)))


if __name__=='__main__':unittest.main()
