"""Bounded spatial references retain timed goals and original DCBF prefix."""
from pathlib import Path
import sys
import unittest
import numpy as np
PACKAGE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PACKAGE/'scripts'), str(PACKAGE/'纯仿真代码')]
from cbf_mpc_core import default_config
from mpc_dcbf_cached_solver import (continuous_guidance, CachedMPCProblem,
    MPCSolverManager, rollout, PreviewBudgetExceeded)


def inputs():
    cfg = default_config(); cfg.update(spatial_pass=True, execution_backup=True)
    state = np.array([.4, 0., 0.]); goals = np.zeros((11, 3))
    goals[:, 0] = .4 + .015*np.arange(11)
    center = np.array([1.35, 0.]); r = .2175
    obs = dict(name='actor', center=center, a=r, b=r, theta=0., dynamic=True,
               guidance=False, prediction_model='cv', prediction_centers=center+np.arange(11)[:,None]*[.005,0.],
               prediction_axes=np.full((11,2),r), guidance_motion=dict(stamp=0.,weights=[1.],models=[
                   dict(mode='cv',position=center.tolist(),velocity=[.05,0.],acceleration=[0.,0.],tau=1.)]))
    memory = dict(progress_previous=np.array([.15,0.]),progress_bounds=[-.22,.22])
    return cfg,state,goals,[obs],memory


class SpatialPassChecks(unittest.TestCase):
    def test_geometry_contraction_is_rate_limited_and_retains_a_valid_prefix(self):
        cfg,state,goals,obs,memory=inputs()
        memory.update(spatial_time=0.,spatial_stamp=0.,spatial_selected=dict(
            name='actor',start=0.,y0=0.,dy0=0.,target=.8,duration=7.5,returning=False,side=1))
        continuous_guidance(state,goals,obs,cfg,memory)
        self.assertTrue(memory['spatial_summary']['geometry_contracted'])
        self.assertTrue(memory['spatial_summary']['kept'])
        self.assertAlmostEqual(memory['spatial_selected']['target'],.794)
        self.assertTrue(CachedMPCProblem(cfg,1,False).validate(memory['progress_plan'],state,obs,
            [-.22,.22],memory['progress_previous']))

    def test_unclassified_slow_mover_is_not_ignored_and_clock_epoch_does_not_jump(self):
        cfg,state,goals,obs,memory = inputs()
        obs[0]['dynamic']=False
        obs[0].pop('guidance_motion')
        obs[0]['prediction_centers']=np.tile(obs[0]['center'],(11,1))
        answer=continuous_guidance(state,goals,obs,cfg,memory)
        self.assertEqual(answer[1],1.)
        self.assertEqual(memory['spatial_summary']['mode'],'side_shift_hold')
        command=memory['progress_plan']['controls'][0].copy()
        state=rollout(state,command[None],.1)[1];goals[:,0]+=.015;memory['progress_previous']=command
        obs[0]['dynamic']=True
        obs[0]['guidance_motion']=dict(stamp=1000.,weights=[1.],models=[dict(
            mode='cv',position=obs[0]['center'].tolist(),velocity=[.05,0.],acceleration=[0.,0.],tau=1.)])
        continuous_guidance(state,goals,obs,cfg,memory)
        self.assertEqual(memory['spatial_summary']['mode'],'side_shift_hold')
        self.assertAlmostEqual(memory['spatial_summary']['phase_age'],.1)

    def test_asymmetric_geometry_selects_mirrored_sides(self):
        sides=[];vectors=[]
        for shift in (.2,-.2):
            cfg,state,goals,obs,memory=inputs()
            obs[0]['center'][1]=shift
            obs[0]['prediction_centers'][:,1]=shift
            obs[0]['guidance_motion']['models'][0]['position'][1]=shift
            answer=continuous_guidance(state,goals,obs,cfg,memory)
            self.assertEqual(answer[1],1.)
            sides.append(memory['spatial_summary']['side']);vectors.append(answer[0])
        self.assertEqual(sides,[-1,1])
        np.testing.assert_allclose(vectors[0][:,1],-vectors[1][:,1],atol=1e-12)

    def test_actual_pass_starts_return_and_completed_return_releases_reference(self):
        cfg,state,goals,obs,memory=inputs()
        state=np.array([3.,.45,0.]);goals[:,0]+=2.6
        memory.update(spatial_time=10.,spatial_stamp=0.,spatial_selected=dict(
            name='actor',start=4.,y0=0.,dy0=0.,target=.45,duration=6.,returning=False,side=1))
        continuous_guidance(state,goals,obs,cfg,memory)
        self.assertEqual(memory['spatial_summary']['mode'],'return')
        self.assertEqual(memory['spatial_selected']['target'],0.)
        memory['spatial_time']+=6.1;state[1]=0.
        continuous_guidance(state,goals,obs,cfg,memory)
        self.assertEqual(memory['spatial_summary']['mode'],'returned_to_timed_path')
        self.assertNotIn('progress_plan',memory)

    def test_slow_same_lane_target_causes_early_side_shift_with_valid_original_prefix(self):
        cfg,state,goals,obs,memory = inputs(); before = goals.copy()
        vectors,level,headings = continuous_guidance(state,goals,obs,cfg,memory)
        self.assertEqual(level,1.); self.assertEqual(memory['spatial_summary']['mode'],'side_shift_hold')
        self.assertGreaterEqual(memory['spatial_summary']['minimum_preview_speed'],.09)
        plan = memory['progress_plan']
        self.assertTrue(CachedMPCProblem(cfg,1,False).validate(plan,state,obs,[-.22,.22],memory['progress_previous']))
        np.testing.assert_array_equal(goals,before)
        np.testing.assert_allclose(plan['states'],rollout(state,plan['controls'],.1),atol=1e-12)
        np.testing.assert_array_equal(vectors,plan['plan_controls'])
        np.testing.assert_array_equal(headings,plan['states'][1:,2])

    def test_next_cycle_keeps_absolute_phase_and_side(self):
        cfg,state,goals,obs,memory = inputs()
        continuous_guidance(state,goals,obs,cfg,memory); selected = memory['spatial_selected'].copy()
        command = memory['progress_plan']['controls'][0].copy(); state = rollout(state,command[None],.1)[1]
        goals[:,0] += .015; memory['progress_previous'] = command
        obs[0]['center'] = obs[0]['center']+[.005,0.]
        obs[0]['prediction_centers'] += [.005,0.]
        obs[0]['guidance_motion']['stamp'] = .1
        obs[0]['guidance_motion']['models'][0]['position'] = obs[0]['center'].tolist()
        continuous_guidance(state,goals,obs,cfg,memory)
        self.assertEqual(memory['spatial_selected']['start'],selected['start'])
        self.assertEqual(memory['spatial_selected']['side'],selected['side'])
        self.assertTrue(memory['spatial_summary']['kept'])

    def test_budget_and_incompatible_modes_are_rejected(self):
        cfg,state,goals,obs,memory = inputs()
        with self.assertRaises(PreviewBudgetExceeded):
            continuous_guidance(state,goals,obs,dict(cfg,_preview_deadline=0.),memory)
        self.assertNotIn('progress_plan',memory)
        for key,value in [('execution_backup',False),('suppress_guidance',True),('rear_pass',True),
                          ('progress_guidance',True),('prediction_risk',True),('constraint_mode','ellipse')]:
            with self.assertRaises(ValueError): MPCSolverManager(dict(cfg,**{key:value}))
