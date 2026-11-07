"""Wang eq14/ellipse comparison; physical envelope and original mode preserved."""
from pathlib import Path
import sys
import unittest
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
from cbf_mpc_core import default_config
from mpc_dcbf_cached_solver import (CachedMPCProblem,MPCSolverManager,barrier_values,
    first_step_mask,rollout,adapt_negative_recovery)


def config(**kwargs):
    cfg=default_config();cfg.update(execution_backup=True,suppress_guidance=True,risk_penalty='wang_inverse',
        prediction_risk=True,risk_weight=.3);cfg.update(kwargs)
    return cfg


def circle(x=.8,y=0.):
    return dict(name='circle',center=np.array([x,y]),a=.15,b=.15,theta=0.,
        prediction_centers=np.tile([x,y],(11,1)))


class WangCoreChecks(unittest.TestCase):
    def test_eq14_closed_form_and_per_node_discount(self):
        cfg=config();problem=CachedMPCProblem(cfg,1,False)
        expected=.3*np.exp(-.1*np.arange(1,11)).sum()/(np.sqrt(.8**2+1e-12)+.02)
        self.assertAlmostEqual(problem.prediction_risk_cost(np.zeros((11,3)),[circle()]),expected,places=12)

    def test_original_cutoff_is_inclusive_and_discontinuous(self):
        problem=CachedMPCProblem(config(),1,False);states=np.zeros((11,3))
        self.assertGreater(problem.prediction_risk_cost(states,[circle(1.)]),0.)
        self.assertEqual(problem.prediction_risk_cost(states,[circle(1.+1e-6)]),0.)

    def test_geometric_witness_can_be_safe_but_violate_dcbf_rate(self):
        cfg=config();geometric=dict(cfg,constraint_mode='ellipse');obs=[circle(.3)]
        u=np.zeros((10,2));u[0]=[.22,0.];state=np.zeros(3)
        candidate=dict(states=rollout(state,u,.1),controls=u,plan_controls=np.zeros_like(u))
        old=CachedMPCProblem(cfg,1,False);new=CachedMPCProblem(geometric,1,False)
        self.assertGreater(barrier_values(candidate['states'],obs,.12).min(),0.)
        self.assertFalse(old.validate(candidate,state,obs,(-.22,.22),np.zeros(2)))
        self.assertTrue(new.validate(candidate,state,obs,(-.22,.22),np.zeros(2)))
        self.assertFalse(first_step_mask(state,u[:1],obs,(-.22,.22),np.zeros(2),cfg)[0][0])
        self.assertTrue(first_step_mask(state,u[:1],obs,(-.22,.22),np.zeros(2),geometric)[0][0])

    def test_initial_safe_set_assumption_is_not_relaxed_for_ellipse(self):
        cfg=config(constraint_mode='ellipse');problem=CachedMPCProblem(cfg,1,False)
        obs=[circle(.1)];state=np.zeros(3);u=np.tile([-.22,0.],(10,1))
        candidate=dict(states=rollout(state,u,.1),controls=u,plan_controls=np.zeros_like(u))
        self.assertFalse(problem.validate(candidate,state,obs,(-.22,.22),u[0]))
        self.assertFalse(first_step_mask(state,u[:1],obs,(-.22,.22),u[0],cfg)[0][0])
        with self.assertRaisesRegex(RuntimeError,'initial state'):
            problem.solve(state,np.zeros((11,3)),obs,(-.22,.22),u[0],u,u)
        self.assertEqual(problem.last_attempt['status'],'ellipse_initial_precondition_failed')
        _,info=adapt_negative_recovery(state,obs,(-.22,.22),u[0],cfg)
        self.assertFalse(info['adapted'])

    def test_ellipse_objective_and_backup_cost_match_solver(self):
        cfg=config(constraint_mode='ellipse');problem=CachedMPCProblem(cfg,1,False)
        goals=np.zeros((11,3));goals[:,0]=.015*np.arange(11);u=np.tile([.15,0.],(10,1));obs=[circle(.65,-.25)]
        answer=problem.solve(np.zeros(3),goals,obs,(-.22,.22),u[0],u,u)
        self.assertTrue(problem.validate(answer,np.zeros(3),obs,(-.22,.22),u[0]))
        value=problem.candidate_cost(answer,goals,u[0],(np.zeros((10,2)),0.),obs)
        self.assertAlmostEqual(value,answer['cost'],places=6)

    def test_expanded_ellipse_has_exactly_the_same_forbidden_region(self):
        rng=np.random.default_rng(91);angle=.7;rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        obs=dict(center=np.array([.8,-.3]),a=.4,b=.2,theta=angle)
        states=np.column_stack((rng.normal(size=(100,2)),np.zeros(100)))
        h=np.array([barrier_values(s[None,:],[obs],.12)[0,0] for s in states])
        local=(states[:,:2]-obs['center'])@rotation
        normalized=(local[:,0]/(.4*(1+.12/.2)))**2+(local[:,1]/(.2+.12))**2-1
        np.testing.assert_array_equal(h>=0,normalized>=0)

    def test_invalid_modes_and_wang_tail_are_rejected(self):
        for key,value in [('constraint_mode','unknown'),('risk_penalty','unknown'),('wang_iota',0.),('wang_mu',-1.),('risk_tail_seconds',1.)]:
            with self.assertRaises(ValueError):CachedMPCProblem(config(**{key:value}),1,False)

    def test_manager_records_precondition_failure_without_invalid_backup(self):
        cfg=config(constraint_mode='ellipse',realtime=True,nlp_wall_time=.1,manager_wall_time=.3)
        manager=MPCSolverManager(cfg);state=np.zeros(3)
        with self.assertRaises(RuntimeError):manager.solve(state,np.zeros((11,3)),[circle(.1)],(-.22,.22),None,'robot')
        record=manager.last_records['robot']
        self.assertFalse(record['accepted_candidate'])
        self.assertEqual(record['attempt_trace'][0]['status'],'ellipse_initial_precondition_failed')

