"""Risk-cost scale, causal forecasts, unchanged constraints and backup scoring."""
from pathlib import Path
import sys
import math
import unittest
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
from cbf_mpc_core import default_config
from mpc_dcbf_cached_solver import CachedMPCProblem,continuous_guidance,rollout,MPCSolverManager


def setup(risk=True,**extra):
    cfg=default_config();cfg.update(prediction_risk=risk,suppress_guidance=True,execution_backup=True,**extra)
    n=cfg['horizon'];goals=np.zeros((n+1,3));goals[:,0]=np.arange(n+1)*.015
    centers=np.column_stack((np.full(n+1,.65),-.25+.02*np.arange(n+1)))
    obs=dict(name='crossing',center=centers[0].copy(),prediction_centers=centers,
        prediction_axes=np.tile([.15,.15],(n+1,1)),a=.15,b=.15,theta=0.,dynamic=True)
    return cfg,goals,[obs]


class PredictionRiskChecks(unittest.TestCase):
    def test_circle_closed_form_discount_and_mask(self):
        cfg,goals,obs=setup();problem=CachedMPCProblem(cfg,1,False)
        states=goals.copy();h=np.linalg.norm(states[1:,:2]-obs[0]['prediction_centers'][1:],axis=1)-.15-.12
        expected=sum(3.*math.exp(-.5*.1*j)*max(1-x/.35,0.)**4 for j,x in enumerate(h,1))
        self.assertAlmostEqual(problem.prediction_risk_cost(states,obs),expected,places=10)
        self.assertEqual(problem.prediction_risk_cost(states,[dict(obs[0],peer=True)]),0.)

    def test_same_current_center_but_different_future_changes_risk(self):
        cfg,goals,obs=setup();problem=CachedMPCProblem(cfg,1,False)
        away=obs[0]['prediction_centers'].copy();away[:,1]=-.25-.02*np.arange(11)
        alternative=[dict(obs[0],prediction_centers=away)]
        np.testing.assert_array_equal(obs[0]['prediction_centers'][0],away[0])
        self.assertGreater(problem.prediction_risk_cost(goals,obs),problem.prediction_risk_cost(goals,alternative))
        self.assertEqual(problem.prediction_risk_cost(goals,[dict(obs[0],center=np.array([99.,99.]))]),
            problem.prediction_risk_cost(goals,obs)) # No stale center substituted for stage forecasts.

    def test_cost_cutoff_has_zero_first_and_second_derivatives(self):
        import casadi as ca
        h=ca.MX.sym('h');f=ca.fmax(1-h/.35,0.)**4
        derivative=ca.Function('cutoff',[h],[f,ca.jacobian(f,h),ca.hessian(f,h)[0]])
        values=np.array([float(v) for v in derivative(.35)])
        np.testing.assert_array_equal(values,np.zeros(3))
        self.assertTrue(all(np.isfinite(float(v)) for v in derivative(-.1)))

    def test_suppress_guide_clears_stale_prefix_but_does_not_emit_control(self):
        cfg,goals,obs=setup();memory=dict(progress_plan={'old':True},guide_index=0,progress_decision={'old':True})
        vectors,level=continuous_guidance(np.zeros(3),goals,obs,dict(cfg,progress_guidance=True),memory)
        self.assertEqual(level,0.);np.testing.assert_array_equal(vectors,np.zeros((10,2)))
        self.assertNotIn('progress_plan',memory);self.assertNotIn('progress_decision',memory)
        self.assertEqual(memory['progress_summary']['mode'],'disabled')

    def test_objective_matches_checked_backup_using_same_geometry(self):
        cfg,goals,obs=setup();problem=CachedMPCProblem(cfg,1,False)
        u=np.tile([.15,0.],(10,1));q=np.tile([.15,0.],(10,1));previous=np.array([.15,0.])
        answer=problem.solve(np.zeros(3),goals,obs,(-.22,.22),previous,u,q)
        score=problem.candidate_cost(answer,goals,previous,(np.zeros((10,2)),0.),obs)
        self.assertAlmostEqual(score,answer['cost'],places=6)
        self.assertGreater(answer['risk_cost'],0.)
        manager=MPCSolverManager(cfg)
        backup=manager._full_backup(problem,np.zeros(3),goals,obs,(-.22,.22),previous,
            answer['controls'],answer['plan_controls'],(np.zeros((10,2)),0.),'checked')
        self.assertIsNotNone(backup);self.assertAlmostEqual(backup['cost'],score,places=6)

    def test_zero_weight_preserves_solution_and_constraint_count(self):
        old,goals,obs=setup(False);new=dict(old,prediction_risk=True,risk_weight=0.)
        old_problem=CachedMPCProblem(old,1,False);new_problem=CachedMPCProblem(new,1,False)
        self.assertEqual(old_problem.opti.nx,new_problem.opti.nx)
        self.assertEqual(old_problem.opti.ng,new_problem.opti.ng)
        u=np.tile([.15,0.],(10,1));previous=np.array([.15,0.])
        a=old_problem.solve(np.zeros(3),goals,obs,(-.22,.22),previous,u,u)
        b=new_problem.solve(np.zeros(3),goals,obs,(-.22,.22),previous,u,u)
        np.testing.assert_allclose(a['controls'],b['controls'],atol=1e-7,rtol=0.)
        self.assertAlmostEqual(a['cost'],b['cost'],places=7)
        self.assertTrue(new_problem.validate(a,np.zeros(3),obs,(-.22,.22),previous))
        unsafe=dict(a,states=rollout(np.zeros(3),np.tile([.22,0.],(10,1)),.1),controls=np.tile([.22,0.],(10,1)))
        near=[dict(obs[0],prediction_centers=np.tile([.15,0.],(11,1)))]
        self.assertFalse(old_problem.validate(unsafe,np.zeros(3),near,(-.22,.22),previous))
        self.assertFalse(new_problem.validate(unsafe,np.zeros(3),near,(-.22,.22),previous))

    def test_invalid_penalty_parameters_are_rejected(self):
        cfg,_,_=setup()
        for key,value in [('risk_weight',-1.),('risk_clearance',0.),('risk_discount',float('nan'))]:
            with self.assertRaises(ValueError):CachedMPCProblem(dict(cfg,**{key:value}),1,False)

