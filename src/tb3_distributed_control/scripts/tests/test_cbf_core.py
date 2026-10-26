#!/usr/bin/env python3
"""Numerical regression for ROS timing, sensed envelopes and moving-peer CBFs."""
from pathlib import Path
import sys
import unittest
import numpy as np

PACKAGE=Path(__file__).resolve().parents[2]
ROOT=PACKAGE.parents[1]
sys.path.insert(0,str(PACKAGE/'scripts'))
sys.path.insert(0,str(PACKAGE/'纯仿真代码'))
from cbf_mpc_core import default_config,TimedReference,ObservedEnvelopes,peer_obstacle
from mpc_dcbf_cached_solver import CachedMPCProblem,barrier_values,rollout


class CoreChecks(unittest.TestCase):
    def test_reference_timestamps_and_curve(self):
        ref=TimedReference(2);v=np.array([.13,.07]);acc=np.array([-.01,.004])
        position=lambda t:np.array([.1,.3])+v*t+.5*acc*t*t
        for k in range(41):
            t=k*.1;raw=position(t+.2)
            ref.observe(t,raw,(raw-position(t+.1))/.1)
        age=.07;g=ref.goals(age,0.)
        expected=np.array([position(t+age+j*.1)+ref.offset() for j in range(11)])
        np.testing.assert_allclose(g[:,:2],expected,atol=1e-12)
        q=np.diff(g[:,:2],axis=0)/.1
        u=np.column_stack((np.linalg.norm(q,axis=1),((np.diff(g[:,2])+np.pi)%(2*np.pi)-np.pi)/.1))
        np.testing.assert_allclose(rollout(g[0],u,.1),g,atol=1e-12)

    def test_observation_is_used_at_new_track_initialization(self):
        model=ObservedEnvelopes();raw=dict(id=4,center=np.array([1.,0.]),a=.3,b=.2,theta=0.,variance=0.)
        first=model.update([raw],1.)[0]
        self.assertLess(model.tracks[4][1][0,0],.001)
        self.assertGreater(first['a'],raw['a'])
        self.assertTrue(np.all(np.diff(first['prediction_axes'][:,0])>0))
        saved=model.tracks[4][1].copy();model.update([raw],1.)
        np.testing.assert_array_equal(saved,model.tracks[4][1])

    def test_moving_peer_barrier_uses_each_prediction_node(self):
        peer=peer_obstacle(1,[1.,0.,0.],[-.1,0.],.1,10,.2)
        states=np.zeros((11,3));h=barrier_values(states,[peer],.12)[0]
        np.testing.assert_allclose(h,1.-.01*np.arange(11)-.2-.12,atol=1e-11)
        self.assertFalse(peer['guidance'])

    def test_rotating_ellipse_barrier_uses_each_prediction_angle(self):
        import casadi as ca
        obs=dict(center=np.zeros(2),a=.4,b=.1,theta=0.,
                 prediction_angles=np.array([0.,np.pi/2]))
        states=np.tile([.6,0.,0.],(2,1))
        np.testing.assert_allclose(barrier_values(states,[obs],.12)[0],[-.07,.38],atol=1e-11)
        cfg=default_config();cfg['horizon']=2
        problem=CachedMPCProblem(cfg,1,False);op=problem.opti
        indices=[i for i in range(op.g.numel()) if ca.depends_on(op.g[i],problem.angles)]
        self.assertEqual(len(indices),2)
        evaluate=ca.Function('check_rotating_cbf',[op.x,op.p],[op.g[indices]])
        op.set_initial(problem.x,np.tile(states[0],(3,1)))
        op.set_value(op.p,np.zeros(op.p.numel()))
        op.set_value(problem.centers,np.zeros((3,2)))
        op.set_value(problem.angles,[0.,np.pi/2,0.])
        op.set_value(problem.axes,np.tile([.4,.1],(3,1)))
        op.set_value(problem.barrier_gamma,.3)
        constraints=np.asarray(evaluate(op.debug.value(op.x,op.initial()),op.value(op.p))).ravel()
        np.testing.assert_allclose(constraints,[-.429,.336],atol=1e-11)

    def test_original_static_nlp_is_numerically_preserved(self):
        old=np.load(Path(__file__).parent/'fixtures/static_mpc_regression.npz')
        cfg=default_config();state=np.array([.4,0.,.2]);g=np.tile(state,(11,1));g[:,0]+=.01*np.arange(11)
        obstacles=[dict(center=np.array([1.7,0.]),a=.4,b=.4,theta=0.)]
        guess=np.tile([.1,0.],(10,1));plan=guess.copy()
        new=CachedMPCProblem(cfg,1,False).solve(state,g,obstacles,(-.22,.22),None,guess,plan)
        np.testing.assert_allclose(new['controls'],old['controls'],atol=1e-6,rtol=0)
        np.testing.assert_allclose(new['states'],old['states'],atol=1e-6,rtol=0)

    def test_outside_envelope_recovery_is_explicit_and_validated(self):
        cfg=default_config();state=np.array([.32,0.,-1.])
        controls=np.tile([.1,0.],(10,1))
        candidate=dict(states=rollout(state,controls,.1),controls=controls,
                       plan_controls=np.zeros((10,2)))
        obstacle=dict(center=np.zeros(2),a=.25,b=.25,theta=0.)
        problem=CachedMPCProblem(cfg,1,False)
        self.assertFalse(problem.validate(dict(candidate),state,[obstacle],(-.22,.22),None))
        obstacle['gamma']=.03
        self.assertTrue(problem.validate(dict(candidate),state,[obstacle],(-.22,.22),None))


if __name__=='__main__':unittest.main()
