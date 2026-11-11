"""Active-scene reachability and rejection of stopping/crawling false success."""
from pathlib import Path
import sys
import json
import unittest
import subprocess
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(Path(__file__).resolve().parent)]
from qualify_cbf_active_scenes import qualify
from evaluate_cbf_active_motion import evaluate_motion


class ActiveSceneChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q=qualify();cls.protocol=cls.q['protocol'];cls.t=np.array(cls.q['witness_times']);cls.p=np.array(cls.q['witness_positions'])
        spec=cls.q['scenes']['active_overtaking']['specification']
        cls.radius=spec['radius'];cls.actor=np.asarray(spec['start'])+cls.t[:,None]*np.asarray(spec['velocity'])

    def test_all_scenes_have_bounded_nonstopping_geometric_witness_and_real_conflict(self):
        self.assertTrue(self.q['qualified']);json.dumps(self.q,allow_nan=False)
        for r in self.q['scenes'].values():
            self.assertTrue(r['witness']['passed']);self.assertLessEqual(r['witness_speed_max'],.22)
            self.assertGreater(r['witness_speed_min'],.14);self.assertLessEqual(r['witness_yaw_rate_max'],2.84)
            self.assertIsNotNone(r['first_nominal_proxy_collision']);self.assertGreater(r['ideal_geometry_dcbf_residual_min'],0.)

    def test_harness_resolves_active_evaluator_from_an_unrelated_directory(self):
        script=Path(__file__).resolve().parent/'run_cbf_dynamic_gazebo.py'
        process=subprocess.run([sys.executable,str(script),'--help'],cwd='/tmp',stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.assertEqual(process.returncode,0,process.stderr);self.assertIn('active_head_on',process.stdout)

    def test_stopped_robot_and_slow_following_cannot_be_called_success(self):
        for r in self.q['scenes'].values():self.assertFalse(r['stopped_baseline']['passed'])
        self.assertLess(self.q['scenes']['active_head_on']['stopped_baseline']['sampled_proxy_clearance_min'],0.)
        following=self.q['scenes']['active_overtaking']['slow_following_baseline']
        self.assertFalse(following['passed']);self.assertFalse(following['checks']['passed_actor'])

    def test_one_second_stop_fails_even_when_final_error_and_clearance_pass(self):
        shifted=self.t-np.clip(self.t-8.,0.,1.)
        ego=np.column_stack([np.interp(shifted,self.t,self.p[:,j]) for j in (0,1)])
        r=evaluate_motion(self.t,ego,self.actor,self.radius,self.protocol)
        self.assertGreater(r['sampled_proxy_clearance_min'],0.);self.assertTrue(r['checks']['tracking_recovered'])
        self.assertFalse(r['checks']['no_sustained_standstill']);self.assertFalse(r['passed'])

    def test_crawling_without_full_stop_still_fails(self):
        shifted=self.t-.8*np.clip(self.t-8.,0.,2.)
        ego=np.column_stack([np.interp(shifted,self.t,self.p[:,j]) for j in (0,1)])
        r=evaluate_motion(self.t,ego,self.actor,self.radius,self.protocol)
        self.assertTrue(r['checks']['no_sustained_standstill']);self.assertTrue(r['checks']['tracking_recovered'])
        self.assertFalse(r['checks']['no_crawling_time_avoidance']);self.assertFalse(r['passed'])

    def test_missing_truth_interval_is_not_interpolated_into_success(self):
        keep=~((self.t>10.)&(self.t<11.))
        r=evaluate_motion(self.t[keep],self.p[keep],self.actor[keep],self.radius,self.protocol)
        self.assertFalse(r['checks']['truth_covers_motion_window']);self.assertFalse(r['passed'])

    def test_partial_run_has_no_final_success(self):
        keep=self.t<29.
        r=evaluate_motion(self.t[keep],self.p[keep],self.actor[keep],self.radius,self.protocol)
        self.assertIsNone(r['final_position_error']);self.assertFalse(r['passed'])

    def test_fault_stop_is_not_ignored_because_motion_was_continuous(self):
        r=evaluate_motion(self.t,self.p,self.actor,self.radius,self.protocol,fault_stop_count=1)
        self.assertFalse(r['checks']['no_fault_stop']);self.assertFalse(r['passed'])

    def test_straight_nonconflicting_motion_is_not_spatial_avoidance(self):
        ego=np.asarray(self.protocol['reference_start'])+self.t[:,None]*np.asarray(self.protocol['reference_velocity'])
        actor=self.actor.copy();actor[:,1]=.5
        r=evaluate_motion(self.t,ego,actor,self.radius,self.protocol)
        self.assertTrue(r['checks']['sampled_proxy_clearance_positive']);self.assertTrue(r['checks']['no_crawling_time_avoidance'])
        self.assertFalse(r['checks']['spatial_detour_observed']);self.assertFalse(r['passed'])

