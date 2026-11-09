"""Rear passing is a bounded candidate, not a negative obstacle-velocity rule."""
from pathlib import Path
import sys
import unittest
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
from cbf_mpc_core import default_config
from mpc_dcbf_cached_solver import (continuous_guidance,_legacy_continuous_guidance,
    CachedMPCProblem,MPCSolverManager,progress_centers,rollout,PreviewBudgetExceeded)


def inputs(y=-.6):
    cfg=default_config();cfg.update(rear_pass=True,execution_backup=True)
    state=np.array([1.3,0.,0.]);goals=np.zeros((11,3));goals[:,0]=1.3+.015*np.arange(11)
    o=dict(name='moving',center=np.array([2.1,y]),a=.18,b=.18,theta=0.,dynamic=True,guidance=False,
        guidance_motion=dict(stamp=10.,weights=[1.],models=[dict(mode='cv',position=[2.1,y],velocity=[0.,.16],acceleration=[0.,0.],tau=1.)]))
    o['prediction_centers']=progress_centers(o,np.arange(11)*.1,.1);o['prediction_axes']=np.tile([.18,.18],(11,1))
    memory=dict(progress_previous=np.array([.15,0.]),progress_bounds=[-.22,.22])
    return cfg,state,goals,[o],memory


class RearPassChecks(unittest.TestCase):
    def test_mirrored_crossing_changes_the_side_instead_of_always_turning_down(self):
        cfg,state,goals,obs,memory=inputs(-.8)
        continuous_guidance(state,goals,obs,cfg,memory)
        self.assertEqual(memory['rear_summary']['mode'],'rear')
        self.assertLess(memory['rear_selected']['controls'][:,1].min(),0.)
        reflected=dict(obs[0]);reflected['center']=obs[0]['center']*np.array([1.,-1.])
        reflected['prediction_centers']=obs[0]['prediction_centers']*np.array([1.,-1.])
        reflected['guidance_motion']=dict(stamp=10.,weights=[1.],models=[dict(mode='cv',position=[2.1,.8],velocity=[0.,-.16],acceleration=[0.,0.],tau=1.)])
        other=dict(progress_previous=np.array([.15,0.]),progress_bounds=[-.22,.22])
        continuous_guidance(state,goals,[reflected],cfg,other)
        self.assertEqual(other['rear_summary']['mode'],'rear')
        self.assertGreater(other['rear_selected']['controls'][:,1].max(),0.)
        self.assertLess(rollout(state,memory['rear_selected']['controls'],.1)[:,1].min(),-.1)
        self.assertGreater(rollout(state,other['rear_selected']['controls'],.1)[:,1].max(),.1)

    def test_kept_reference_advances_its_phase_instead_of_restarting_the_turn(self):
        cfg,state,goals,obs,memory=inputs(-.8)
        continuous_guidance(state,goals,obs,cfg,memory);old=memory['rear_selected']['controls'].copy()
        advanced=rollout(state,old[:1],.1)[1];goals=goals.copy();goals[:,0]+=.015
        o=dict(obs[0],center=obs[0]['center']+np.array([0.,.016]),prediction_centers=obs[0]['prediction_centers']+np.array([0.,.016]))
        o['guidance_motion']=dict(stamp=10.1,weights=[1.],models=[dict(mode='cv',position=o['center'].tolist(),velocity=[0.,.16],acceleration=[0.,0.],tau=1.)])
        memory['progress_previous']=old[0].copy();continuous_guidance(advanced,goals,[o],cfg,memory)
        self.assertTrue(memory['rear_summary']['kept'])
        np.testing.assert_allclose(memory['rear_selected']['controls'][0],old[1],atol=1e-12)

    def test_selected_prefix_is_bounded_and_original_validator_accepts(self):
        cfg,state,goals,obs,memory=inputs();before=goals.copy()
        vectors,level,headings=continuous_guidance(state,goals,obs,cfg,memory)
        self.assertEqual(memory['rear_summary']['mode'],'rear');self.assertEqual(level,1.)
        self.assertLessEqual(memory['rear_summary']['rear_offset'],-.32)
        plan=memory['progress_plan'];problem=CachedMPCProblem(cfg,1,False)
        self.assertTrue(problem.validate(plan,state,obs,[-.22,.22],memory['progress_previous']))
        np.testing.assert_allclose(plan['states'],rollout(state,plan['controls'],.1),atol=1e-12)
        np.testing.assert_array_equal(vectors,plan['plan_controls']);np.testing.assert_array_equal(headings,plan['states'][1:,2])
        np.testing.assert_array_equal(goals,before)
        self.assertTrue(np.all(np.abs(plan['controls'][:,0])<=.22));self.assertTrue(np.all(np.abs(plan['controls'][:,1])<=2.84))

    def test_receding_crossing_object_does_not_keep_extra_guide_active(self):
        cfg,state,goals,obs,memory=inputs(.5)
        answer=continuous_guidance(state,goals,obs,cfg,memory)
        self.assertEqual(answer[1],0.);self.assertIn(memory['rear_summary']['mode'],('released','direct_clear'))
        self.assertNotIn('progress_plan',memory)

    def test_static_obstacle_keeps_legacy_reference(self):
        cfg,state,goals,obs,_=inputs();o=dict(obs[0],dynamic=False,guidance=True)
        o.pop('guidance_motion');o.pop('prediction_centers')
        a=continuous_guidance(state,goals,[o],cfg,{})
        b=_legacy_continuous_guidance(state,goals,[o],dict(cfg,rear_pass=False),{})
        np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1],b[1])

    def test_budget_aborts_instead_of_returning_partial_arrays(self):
        cfg,state,goals,obs,memory=inputs()
        with self.assertRaises(PreviewBudgetExceeded):continuous_guidance(state,goals,obs,dict(cfg,_preview_deadline=0.),memory)
        self.assertNotIn('progress_plan',memory)

    def test_incompatible_candidate_flags_are_rejected(self):
        cfg,*_=inputs()
        for key,value in [('execution_backup',False),('suppress_guidance',True),('prediction_risk',True),('progress_guidance',True),('constraint_mode','ellipse')]:
            with self.assertRaises(ValueError):MPCSolverManager(dict(cfg,**{key:value}))

