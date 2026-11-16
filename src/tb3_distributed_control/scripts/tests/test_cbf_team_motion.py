"""Two-robot criteria keep safety/continuous motion distinct from detour roles."""
from pathlib import Path
import sys
import unittest
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码'),str(Path(__file__).resolve().parent)]
from evaluate_cbf_team_motion import evaluate_team
from qualify_cbf_team_scenes import qualify
from test_cbf_spatial_pass import inputs
from mpc_dcbf_cached_solver import continuous_guidance
from mpc_dcbf_cached_solver import CachedMPCProblem


class TeamMotionChecks(unittest.TestCase):
    def test_failed_long_preview_can_only_return_a_valid_labeled_hard_prefix(self):
        cfg,state,goals,obs,memory=inputs();cfg['spatial_prefix_retention']=True
        obs.append(dict(name='distant_approach',center=np.array([10.,0.]),a=3.,b=3.,theta=0.,
            dynamic=True,prediction_centers=np.column_stack((10.-.15*np.arange(11),np.zeros(11))),
            guidance_motion=dict(stamp=0.,weights=[1.],models=[dict(mode='cv',position=[10.,0.],
                velocity=[-1.5,0.],acceleration=[0.,0.],tau=1.)])))
        continuous_guidance(state,goals,obs,cfg,memory)
        self.assertEqual(memory['spatial_summary']['preview_scope'],'original_hard_prefix_only')
        self.assertLess(memory['spatial_summary']['extended_preview_min_h'],0.)
        self.assertTrue(CachedMPCProblem(cfg,2,False).validate(memory['progress_plan'],state,obs,
            [-.22,.22],memory['progress_previous']))

    def test_measured_transverse_motion_can_override_an_early_wrong_side(self):
        cfg,state,goals,obs,memory=inputs();cfg['spatial_rear_preference']=True
        center=np.array([2.3,.85]);velocity=np.array([-.04,-.1])
        obs[0].update(center=center,prediction_centers=center+np.arange(11)[:,None]*.1*velocity)
        obs[0]['guidance_motion']['models'][0].update(position=center.tolist(),velocity=velocity.tolist())
        memory['spatial_selected']=dict(name='actor',start=0.,y0=0.,dy0=0.,target=-.5,
            duration=6.,returning=False,side=-1)
        continuous_guidance(state,goals,obs,cfg,memory)
        self.assertEqual(memory['spatial_summary']['crossing_side_preference'],1)
        self.assertEqual(memory['spatial_summary']['side'],1)

    def test_rear_preference_does_not_force_an_infeasible_route(self):
        cfg,state,goals,obs,memory=inputs();cfg['spatial_rear_preference']=True
        center=np.array([1.35,.8]);velocity=np.array([-.04,-.1])
        obs[0].update(center=center,prediction_centers=center+np.arange(11)[:,None]*.1*velocity)
        obs[0]['guidance_motion']['models'][0].update(position=center.tolist(),velocity=velocity.tolist())
        memory['spatial_selected']=dict(name='actor',start=0.,y0=0.,dy0=0.,target=-.5,
            duration=6.,returning=False,side=-1)
        continuous_guidance(state,goals,obs,cfg,memory)
        self.assertEqual(memory['spatial_summary']['crossing_side_preference'],1)
        self.assertEqual(memory['spatial_summary']['rear_motion_candidates'],0)
        self.assertEqual(memory['spatial_summary']['side'],-1)

    @classmethod
    def setUpClass(cls):cls.q=qualify()

    def inputs(self,scenario='active_overtaking'):
        q=self.q;p=np.array(q['positions']);t=np.array(q['times']);proto=q['protocol']
        specs={'active_overtaking':([1.35,0.],[.05,0.]),'active_oblique_crossing':([2.3,.85],[-.04,-.10])}
        start,v=specs[scenario];actor=np.array(start)+t[:,None]*v
        return t,p,actor,proto

    def test_all_scenes_have_bounded_qualified_witness_and_correct_frozen_roles(self):
        self.assertTrue(self.q['qualified']);self.assertLess(self.q['maximum_speed'],.22)
        self.assertEqual(self.q['scenes']['active_oblique_crossing']['nominal_collision_roles'],[True,False])

    def test_peer_crossing_between_samples_is_rejected(self):
        t,p,actor,proto=self.inputs();i=100
        p[0,i]=[2.,.45];p[1,i]=[1.,.45];p[0,i+1]=[1.,.45];p[1,i+1]=[2.,.45]
        report=evaluate_team(t,p,actor,.18,proto,'active_overtaking')
        self.assertFalse(report['checks']['linear_swept_peer_proxy_clearance_positive'])

    def test_unthreatened_follower_can_keep_the_nominal_line(self):
        t,p,actor,proto=self.inputs('active_oblique_crossing');p[1,:,1]=0.
        report=evaluate_team(t,p,actor,.18,proto,'active_oblique_crossing')
        self.assertTrue(report['passed']);self.assertTrue(report['robots']['tb3_1']['accepted'])
        self.assertFalse(report['robots']['tb3_1']['motion']['checks']['spatial_detour_observed'])
        self.assertFalse(report['robots']['tb3_1']['spatial_required'])

    def test_fault_stop_or_missing_command_log_is_not_success(self):
        t,p,actor,proto=self.inputs()
        for counts in [(0,1),(0,None)]:
            self.assertFalse(evaluate_team(t,p,actor,.18,proto,'active_overtaking',counts)['passed'])

    def test_relative_formation_must_recover_not_only_individual_bounds(self):
        t,p,actor,proto=self.inputs();p[1,-1,0]-=.15
        report=evaluate_team(t,p,actor,.18,proto,'active_overtaking')
        self.assertFalse(report['checks']['relative_formation_recovered'])
        self.assertTrue(report['robots']['tb3_1']['motion']['checks']['tracking_recovered'])

    def test_short_peer_hard_constraint_still_rejects_invalid_spatial_prefix(self):
        cfg,state,goals,obs,memory=inputs()
        obs.append(dict(name='peer_1',peer=True,center=np.array([.4,.25]),a=.2,b=.2,theta=0.,
                        prediction_centers=np.tile([.4,.25],(11,1))))
        continuous_guidance(state,goals,obs,cfg,memory)
        self.assertNotIn('progress_plan',memory)
