#!/usr/bin/env python3
"""Deadline, failure accounting, bounded fallback and dual-identity regression."""
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np

PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
from cbf_mpc_core import default_config,peer_obstacle
from mpc_dcbf_cached_solver import CachedMPCProblem,MPCSolverManager,barrier_values,rollout,first_step_mask,PreviewBudgetExceeded


def config(**extra):
    value=default_config();value.update(realtime=True,continuous_guidance=False,horizon=3,
        nlp_wall_time=.5,manager_wall_time=1.)
    value.update(extra);return value


class RealtimeChecks(unittest.TestCase):
    def test_predictive_candidate_requires_explicit_opt_in(self):
        for enabled in (False,True):
            cfg=config(execution_backup=True,progress_guidance=True,predictive_bypass=enabled)
            manager=MPCSolverManager(cfg)
            with patch('mpc_dcbf_cached_solver.continuous_guidance',return_value=(np.zeros((3,2)),0.)) as guide:
                manager.solve(np.zeros(3),np.zeros((4,3)),[],(-.22,.22),None,'r')
            actual=guide.call_args.args[3]
            for name in ('predictive_obstruction','active_side_lock','relative_avoidance_flow'):
                self.assertEqual(actual[name],enabled)

    def test_infeasible_escape_improves_residual_without_claiming_cbf_feasibility(self):
        cfg=config(execution_backup=True,infeasible_escape=True);manager=MPCSolverManager(cfg)
        problem=CachedMPCProblem(cfg,1,False);problem.solve=Mock(side_effect=RuntimeError('forced'))
        manager.problems[(1,False)]=problem
        obs=dict(center=np.array([.25,0.]),a=.15,b=.15,theta=0.,name='closing',
            prediction_centers=np.column_stack((.25-.05*np.arange(4),np.zeros(4))))
        state=np.zeros(3);result=manager.solve(state,np.zeros((4,3)),[obs],(-.22,.22),None,'r')
        self.assertEqual(result['status'],'best_effort_escape')
        self.assertEqual(result['validated_steps'],0)
        self.assertFalse(result['success'])
        self.assertLess(result['controls'][0,0],0.)
        self.assertLess(result['min_slack'],0.)
        self.assertGreater(result['min_slack'],result['escape_zero_min_slack']+2e-6)
        self.assertFalse(problem.validate(result,state,[obs],(-.22,.22),np.zeros(2),one_step=True))
        self.assertTrue(manager.last_records['r']['escape_prepared'])
        self.assertFalse(manager.last_records['r']['backup_prepared'])

    def test_escape_protects_other_barriers_that_zero_keeps_safe(self):
        cfg=config(execution_backup=True,infeasible_escape=True);manager=MPCSolverManager(cfg)
        problem=CachedMPCProblem(cfg,2,False);problem.solve=Mock(side_effect=RuntimeError('forced'))
        manager.problems[(2,False)]=problem
        front=dict(center=np.array([.25,0.]),a=.15,b=.15,theta=0.,name='closing',
            prediction_centers=np.column_stack((.25-.05*np.arange(4),np.zeros(4))))
        behind=dict(center=np.array([-.275,0.]),a=.15,b=.15,theta=0.,name='behind')
        result=manager.solve(np.zeros(3),np.zeros((4,3)),[front,behind],(-.22,.22),None,'r')
        self.assertEqual(result['validated_steps'],0)
        self.assertGreaterEqual(result['escape_next_h'][1],-2e-6)
        self.assertTrue(np.all(np.abs(result['controls'])<=np.array([.22,2.84])+2e-6))

    def test_escape_does_not_return_an_action_that_cannot_improve_over_zero(self):
        cfg=config(execution_backup=True,infeasible_escape=True);manager=MPCSolverManager(cfg)
        problem=CachedMPCProblem(cfg,4,False);problem.solve=Mock(side_effect=RuntimeError('forced'))
        manager.problems[(4,False)]=problem;obstacles=[]
        for i,direction in enumerate([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]):
            direction=np.array(direction);centers=(.25-.05*np.arange(4))[:,None]*direction
            obstacles.append(dict(name=str(i),center=centers[0],prediction_centers=centers,a=.15,b=.15,theta=0.))
        with self.assertRaises(RuntimeError):manager.solve(np.zeros(3),np.zeros((4,3)),obstacles,(-.22,.22),None,'r')
        self.assertFalse(manager.last_records['r']['escape_prepared'])

    def test_preview_budget_exhaustion_withdraws_partial_guidance_and_keeps_checked_backup(self):
        cfg=config(execution_backup=True,progress_guidance=True);manager=MPCSolverManager(cfg)
        problem=CachedMPCProblem(cfg,0,False)
        problem.solve=Mock(side_effect=RuntimeError('forced NLP failure'))
        manager.problems[(0,False)]=problem
        def interrupted(state,goals,obs,config,memory):
            memory['progress_plan']={'stale_partial':True}
            raise PreviewBudgetExceeded()
        with patch('mpc_dcbf_cached_solver.continuous_guidance',side_effect=interrupted):
            result=manager.solve(np.zeros(3),np.zeros((4,3)),[],(-.22,.22),None,'r')
        self.assertTrue(manager.last_records['r']['preview_budget_hit'])
        self.assertTrue(manager.last_records['r']['backup_prepared'])
        self.assertNotIn('progress_plan',manager.robot_memory['r'])
        self.assertEqual(problem.solve.call_args.args[7][1],0.)
        self.assertTrue(problem.validate(result,np.zeros(3),[],(-.22,.22),np.zeros(2)))

    def test_prepared_backup_exists_before_nlp_and_survives_a_failed_attempt(self):
        cfg=config(execution_backup=True);manager=MPCSolverManager(cfg)
        problem=CachedMPCProblem(cfg,1,False)
        state=np.zeros(3);goals=np.tile([.1,0.,0.],(4,1))
        obstacle=dict(name='approaching',center=np.array([.35,0.]),a=.15,b=.15,theta=0.,
            prediction_centers=np.column_stack((.35-.04*np.arange(4),np.zeros(4))))
        zero=dict(controls=np.zeros((1,2)),states=np.zeros((2,3)),plan_controls=np.zeros((1,2)))
        self.assertFalse(problem.validate(zero,state,[obstacle],(-.22,.22),np.zeros(2),one_step=True))
        calls=[];validate=problem.validate
        def checked(*args,**kwargs):
            result=validate(*args,**kwargs)
            if result:calls.append('validated')
            return result
        problem.validate=checked
        def failure(*args,**kwargs):
            self.assertIn('validated',calls)
            calls.append('solve');raise RuntimeError('forced native failure')
        problem.solve=Mock(side_effect=failure)
        manager.problems[(1,False)]=problem
        result=manager.solve(state,goals,[obstacle],(-.22,.22),None,'r')
        self.assertEqual(problem.solve.call_count,1)
        self.assertEqual(result['validated_steps'],1)
        self.assertLess(result['controls'][0,0],0.)
        self.assertFalse(result['success'])
        record=manager.last_records['r']
        self.assertTrue(record['backup_prepared'])
        self.assertEqual(record['accepted_source'],'prepared_control')
        self.assertTrue(validate(dict(controls=result['controls'][:1],states=result['states'][:2],
            plan_controls=np.zeros((1,2))),state,[obstacle],(-.22,.22),np.zeros(2),one_step=True))

    def test_full_backup_is_revalidated_after_obstacle_changes_and_has_exact_objective(self):
        cfg=config(execution_backup=True);manager=MPCSolverManager(cfg)
        problem=CachedMPCProblem(cfg,0,False);state=np.zeros(3)
        goals=np.tile([.04,0.,0.],(4,1));u=np.tile([.1,.2],(3,1));q=np.tile([.08,.01],(3,1))
        guidance=(np.tile([.1,0.],(3,1)),1.,np.array([.02,.04,.06]))
        result=problem.solve(state,goals,[],(-.22,.22),np.zeros(2),u,q,guidance)
        candidate=manager._full_backup(problem,state,goals,[],(-.22,.22),np.zeros(2),
            result['controls'],result['plan_controls'],guidance,'shifted')
        self.assertIsNotNone(candidate)
        self.assertAlmostEqual(candidate['cost'],result['cost'],places=7)
        blocked=dict(center=np.zeros(2),a=.15,b=.15,theta=0.)
        self.assertIsNone(manager._full_backup(problem,state,goals,[blocked],(-.22,.22),np.zeros(2),
            result['controls'],result['plan_controls'],guidance,'shifted'))

    def test_feasible_preview_initializes_nlp_and_is_returned_on_failure(self):
        cfg=config(execution_backup=True,progress_guidance=True);manager=MPCSolverManager(cfg)
        problem=CachedMPCProblem(cfg,0,False);state=np.zeros(3)
        u=np.tile([-.1,.2],(3,1));states=rollout(state,u,.1)
        q=np.diff(states[:,:2],axis=0)/.1
        def preview(state,goals,obs,cfg,memory):
            memory['progress_plan']=dict(controls=u,plan_controls=q,states=states,mode='cw')
            return q,1.,states[1:,2]
        def failure(*args,**kwargs):
            np.testing.assert_allclose(args[5],u)
            np.testing.assert_allclose(args[6],q)
            self.assertIsNone(kwargs['dual_guess'])
            raise RuntimeError('forced native failure')
        problem.solve=Mock(side_effect=failure);manager.problems[(0,False)]=problem
        with patch('mpc_dcbf_cached_solver.continuous_guidance',side_effect=preview):
            result=manager.solve(state,states,[],(-.22,.22),None,'r')
        np.testing.assert_allclose(result['controls'],u)
        self.assertEqual(result['validated_steps'],3)
        self.assertFalse(result['success'])
        self.assertEqual(manager.last_records['r']['status'],'validated_full_backup')

    def test_execution_backup_expired_budget_does_not_validate_or_solve(self):
        manager=MPCSolverManager(config(execution_backup=True))
        problem=SimpleNamespace(opti=SimpleNamespace(ng=0),solve=Mock(),validate=Mock())
        manager.problems[(0,False)]=problem
        with self.assertRaises(RuntimeError):
            manager.solve(np.zeros(3),np.zeros((4,3)),[],(-.22,.22),None,'r',deadline=time.perf_counter()-1.)
        problem.solve.assert_not_called();problem.validate.assert_not_called()

    def test_preview_body_heading_is_passed_separately_from_reverse_velocity(self):
        problem=CachedMPCProblem(config(execution_backup=True),0,False)
        state=np.zeros(3);u=np.tile([-.1,0.],(3,1));goals=rollout(state,u,.1)
        guidance=(np.tile([-.1,0.],(3,1)),1.,np.zeros(3))
        problem.solve(state,goals,[],(-.22,.22),np.zeros(2),u,guidance[0],guidance)
        np.testing.assert_allclose(problem.opti.value(problem.guide_heading),0.)

    def test_iteration_deadline_stops_cleanly_and_candidate_is_revalidated(self):
        cfg=config();problem=CachedMPCProblem(cfg,0,False)
        state=np.zeros(3);goals=np.tile(state,(4,1))
        result=problem.solve(state,goals,[],(-.22,.22),np.zeros(2),np.zeros((3,2)),
            np.zeros((3,2)),deadline=time.perf_counter()-1.)
        self.assertTrue(problem.last_attempt['deadline_stop'])
        self.assertFalse(result['converged'])
        self.assertTrue(problem.validate(result,state,[],(-.22,.22),np.zeros(2)))

    def test_failure_is_logged_before_exception_and_only_one_nlp_is_attempted(self):
        cfg=config();manager=MPCSolverManager(cfg)
        problem=SimpleNamespace(opti=SimpleNamespace(ng=0),
            last_attempt=dict(status='Maximum_Iterations_Exceeded',iterations=120,wall_ms=1.,feasible=False))
        problem.solve=Mock(side_effect=RuntimeError('forced solver failure'))
        problem.validate=Mock(return_value=False)
        manager.problems[(0,False)]=problem
        with self.assertRaises(RuntimeError):
            manager.solve(np.zeros(3),np.zeros((4,3)),[],(-.22,.22),None,'r',cycle_token=9)
        self.assertEqual(problem.solve.call_count,1)
        record=manager.last_records['r']
        self.assertEqual(record['cycle_token'],9)
        self.assertEqual(record['attempts'],1)
        self.assertEqual(record['attempt_trace'][0]['iterations'],120)
        self.assertFalse(record['accepted_candidate'])
        self.assertEqual(manager.summary()['robots']['r']['failed_cycles'],1)
        self.assertEqual(len(manager.robot_memory['r']['pending_turns']),2)
        # The next period consumes a pending turn, never performs all turns now.
        with self.assertRaises(RuntimeError):
            manager.solve(np.zeros(3),np.zeros((4,3)),[],(-.22,.22),None,'r',cycle_token=10)
        self.assertEqual(problem.solve.call_count,2)
        self.assertEqual(manager.last_records['r']['seed'],'turn_left')

    def test_expired_budget_never_starts_an_nlp_or_returns_an_unchecked_control(self):
        manager=MPCSolverManager(config())
        problem=SimpleNamespace(opti=SimpleNamespace(ng=0),solve=Mock(),validate=Mock())
        manager.problems[(0,False)]=problem
        with self.assertRaises(RuntimeError):
            manager.solve(np.zeros(3),np.zeros((4,3)),[],(-.22,.22),None,'r',deadline=time.perf_counter()-1.)
        problem.solve.assert_not_called()
        problem.validate.assert_not_called()
        self.assertEqual(manager.last_records['r']['attempts'],0)

    def test_shifted_duals_require_matching_obstacle_identity_and_actual_previous_command(self):
        cfg=config();manager=MPCSolverManager(cfg)
        state=np.zeros(3);goals=np.tile(state,(4,1));goals[:,0]=np.arange(4)*.01
        obstacles=[dict(center=np.array([5.,5.]),a=.3,b=.2,theta=0.,name='first')]
        first=manager.solve(state,goals,obstacles,(-.22,.22),None,'r')
        self.assertTrue(first['converged'])
        second=manager.solve(state,goals,obstacles,(-.22,.22),first,'r',applied_previous=np.array([-.1,.2]))
        self.assertTrue(manager.last_records['r']['dual_warm'])
        problem=manager.problems[(1,False)]
        np.testing.assert_allclose(problem.opti.value(problem.previous),[-.1,.2])
        obstacles[0]['name']='different_track'
        manager.solve(state,goals,obstacles,(-.22,.22),second,'r')
        self.assertFalse(manager.last_records['r']['dual_warm'])

    def test_dual_time_shift_maps_bounds_dynamics_and_barriers(self):
        problem=CachedMPCProblem(config(horizon=2),1,False)
        self.assertEqual(problem.dual_shift[5],6)
        np.testing.assert_array_equal(problem.dual_shift[13:18],np.arange(18,23))
        np.testing.assert_array_equal(problem.dual_shift[23:25],[24,24])
        # Delta-bound variants must also have a complete verified layout.
        for first in (False,True):
            p=CachedMPCProblem(config(delta_limit=True),2,first)
            self.assertEqual(len(p.dual_shift),p.opti.ng)

    def test_short_prediction_validation_checks_correct_moving_peer_nodes(self):
        peer=peer_obstacle(1,[1.,0.,0.],[-.2,0.],.1,3,.2)
        h=barrier_values(np.zeros((2,3)),[peer],.12)[0]
        np.testing.assert_allclose(h,[.68,.66],atol=1e-11)
        problem=CachedMPCProblem(config(),1,False)
        u=np.array([[.1,0.]])
        candidate=dict(controls=u,states=rollout(np.zeros(3),u,.1),plan_controls=np.zeros((1,2)))
        self.assertTrue(problem.validate(candidate,np.zeros(3),[peer],(-.22,.22),np.zeros(2),one_step=True))

    def test_vectorized_screen_matches_numeric_validator_for_bounds_rotation_and_recovery(self):
        rng=np.random.default_rng(72)
        commands=rng.uniform([-.3,-3.2],[.3,3.2],(250,2))
        peer=peer_obstacle(1,[.7,.2,0.],[-.1,.2],.1,3,.2)
        ellipse=dict(center=np.array([.35,0.]),a=.25,b=.15,theta=.4,
                     prediction_angles=np.array([.4,.6,.8,1.]),gamma=.03)
        for delta in (False,True):
            cfg=config(delta_limit=delta);problem=CachedMPCProblem(cfg,2,bool(delta))
            state=np.array([0.,0.,.2]);previous=np.array([.05,.1])
            mask,_,_=first_step_mask(state,commands,[peer,ellipse],(-.22,.22),previous,cfg)
            expected=[]
            for command in commands:
                short=dict(controls=command[None,:],states=rollout(state,command[None,:],.1),plan_controls=np.zeros((1,2)))
                expected.append(problem.validate(short,state,[peer,ellipse],(-.22,.22),previous,one_step=True))
            np.testing.assert_array_equal(mask,expected)

    def test_native_failure_has_a_checked_one_step_fallback_not_a_false_full_horizon_claim(self):
        cfg=config();manager=MPCSolverManager(cfg)
        problem=CachedMPCProblem(cfg,1,False)
        problem.last_attempt=dict(status='forced_failure',iterations=1,feasible=False)
        problem.solve=Mock(side_effect=RuntimeError('forced failure'))
        manager.problems[(1,False)]=problem
        obstacle=peer_obstacle(1,[1.,0.,0.],[-.2,0.],.1,3,.2)
        result=manager.solve(np.zeros(3),np.tile([.1,0.,0.],(4,1)),[obstacle],(-.22,.22),None,'r')
        self.assertEqual(result['validated_steps'],1)
        self.assertFalse(result['success'])
        self.assertTrue(problem.validate(result,np.zeros(3),[obstacle],(-.22,.22),np.zeros(2),one_step=True))


if __name__=='__main__':unittest.main()
