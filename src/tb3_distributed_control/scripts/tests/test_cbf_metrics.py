#!/usr/bin/env python3
"""Evaluation semantics: common epochs, sign, causal plans and honest missing data."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock,patch
import threading
import time
import hashlib
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCRIPTS));sys.path.insert(0,str(SCRIPTS.parent/'纯仿真代码'))
from cbf_mpc_core import (LeaderMetricTimeline,accepted_plan_reference,sample_plan_reference,
    timeline_from_leader_history,wrap_angle,METRICS_SCHEMA_VERSION,TimedReference)
from cbf_experiment_results import export_results,finite_error_summary
from recompute_cbf_metrics import convert
from test_cbf_inline_geometry import inline
from mpc_dcbf_cached_solver import barrier_values

inline.accepted_plan_reference=accepted_plan_reference
inline.sample_plan_reference=sample_plan_reference
inline.wrap_angle=wrap_angle;inline.METRICS_SCHEMA_VERSION=METRICS_SCHEMA_VERSION
inline.barrier_values=barrier_values;inline.json=json


class MetricChecks(unittest.TestCase):
    def test_next_step_packets_are_evaluated_at_pose_stamp_not_control_stamp(self):
        t=LeaderMetricTimeline(.1);velocity=np.array([.2,.1])
        for k in range(3):
            logical=k*.1;t.observe_state(logical,velocity*(logical+.1),velocity,np.arctan2(.1,.2))
            t.observe_clock(logical,100.+logical)
        actual=t.evaluate(100.18)
        np.testing.assert_allclose(actual['position'],velocity*.18,atol=1e-12)
        self.assertEqual(actual['quality'],1)
        self.assertAlmostEqual(actual['logical_time'],.18)

    def test_hermite_is_exact_for_constant_acceleration_with_jitter(self):
        t=LeaderMetricTimeline();v=np.array([.1,.02]);a=np.array([.04,-.01])
        for logical in (0.,.103,.199,.304):
            u=logical+.1;velocity=v+a*u
            t.observe_clock(logical,20.+logical)
            t.observe_state(logical,v*u+.5*a*u*u,velocity,np.arctan2(velocity[1],velocity[0]))
        r=t.evaluate(20.25)
        np.testing.assert_allclose(r['position'],v*.25+.5*a*.25**2,atol=1e-12)
        np.testing.assert_allclose(r['velocity'],v+a*.25,atol=1e-12)

    def test_missing_pairs_old_gaps_and_before_start_remain_unavailable(self):
        t=LeaderMetricTimeline();t.observe_clock(0.,10.)
        self.assertEqual(t.evaluate(10.1)['quality'],0)
        t.observe_state(0.,[.1,0.],[1.,0.],0.)
        self.assertEqual(t.evaluate(9.99)['quality'],0)
        self.assertEqual(t.evaluate(10.3)['quality'],0)
        t.observe_clock(1.,11.);t.observe_state(1.,[1.1,0.],[1.,0.],0.)
        self.assertEqual(t.evaluate(10.5)['quality'],0)
        self.assertTrue(np.isnan(t.evaluate(10.5)['position']).all())

    def test_heading_sign_wrap_and_stationary_heading(self):
        self.assertAlmostEqual(wrap_angle(.6-.4),.2)
        self.assertAlmostEqual(wrap_angle(-np.pi+.01-(np.pi-.01)),.02)
        t=LeaderMetricTimeline()
        for k,heading in enumerate((np.pi-.01,-np.pi+.01)):
            t.observe_clock(k*.1,10.+k*.1);t.observe_state(k*.1,[0.,0.],[0.,0.],heading)
        self.assertAlmostEqual(abs(t.evaluate(10.15)['heading']),np.pi)

    def test_previous_plan_reference_and_no_current_anchor_fake_zero(self):
        q=np.tile([.2,-.1],(10,1));p=accepted_plan_reference(10.,[1.,2.],q,.1,10.03,7)
        self.assertEqual(sample_plan_reference(p,10.)['valid'],0)
        self.assertEqual(sample_plan_reference(p,10.02)['valid'],0)
        sample=sample_plan_reference(p,10.1)
        np.testing.assert_allclose(sample['position']-[1.01,2.],[.01,-.01])
        self.assertEqual(sample['cycle'],7)
        self.assertEqual(sample_plan_reference(p,10.2)['valid'],0)
        self.assertEqual(sample_plan_reference(None,10.1)['valid'],0)

    def test_history_decoder_uses_previous_publication_for_the_stored_value(self):
        rows=[]
        for logical in (0.,.11,.22):
            value_time=0. if logical==0 else (0.1 if logical==.11 else .21)
            rows.append(dict(trajectory_time=logical,ros_time=100.+logical,x0=.4+.15*value_time,
                y0=0.,u0x=.15,u0y=0.,theta0=0.))
        timeline=timeline_from_leader_history(rows)
        self.assertAlmostEqual(timeline.evaluate(100.18)['position'][0],.427)

    def test_history_gap_does_not_assign_a_late_pose_to_an_earlier_epoch(self):
        rows=[dict(trajectory_time=t,ros_time=100.+t,x0=.4+.15*t,y0=0.,
                   u0x=.15,u0y=0.,theta0=0.) for t in (0.,.2,.3)]
        timeline=timeline_from_leader_history(rows)
        self.assertEqual(timeline.evaluate(100.1)['quality'],0)
        self.assertAlmostEqual(timeline.evaluate(100.3)['position'][0],.445)

    def test_missing_values_are_not_zero_or_silently_last_valid(self):
        s=finite_error_summary([np.nan,.2,np.nan])
        self.assertEqual(s['valid_samples'],1);self.assertEqual(s['missing_samples'],2)
        self.assertAlmostEqual(s['mean'],.2);self.assertTrue(np.isnan(s['final']))
        self.assertTrue(np.isnan(finite_error_summary([np.nan])['rmse']))
        self.assertAlmostEqual(finite_error_summary([-.2,.2])['mae'],.2)

    def test_post_shutdown_samples_do_not_shift_the_evaluation_endpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'run_metadata.json').write_text(json.dumps(dict(timestamp='window',total_time=1.)))
            base=dict(x=.3,y=.2,theta=.1,v=.1,omega=0.,consensus_error_y=0.,observer_x=.1,
                observer_y=.2,solve_ms=20.,detected_ellipses=0,status='ok')
            rows=[dict(base,time=.9,consensus_error_x=.1),dict(base,time=1.1,consensus_error_x=10.)]
            with patch('cbf_experiment_results.plot_results'):export_results(p,{0:rows},[],{})
            stats=json.loads((p/'experiment_statistics.json').read_text())['0']
            self.assertAlmostEqual(stats['final_position_error'],.1)
            self.assertEqual(stats['excluded_window_rows'],1)

    def test_repeated_or_stale_pose_samples_are_counted_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'run_metadata.json').write_text(json.dumps(dict(timestamp='fresh')))
            base=dict(x=.3,y=.2,theta=.1,v=.1,omega=0.,consensus_error_x=.1,consensus_error_y=0.,
                observer_x=.1,observer_y=.2,solve_ms=20.,detected_ellipses=0,status='ok')
            rows=[dict(base,time=0.,evaluation_time=0.,evaluation_ros_time=1.,odom_age=.01),
                  dict(base,time=.1,evaluation_time=0.,evaluation_ros_time=1.,odom_age=.11),
                  dict(base,time=.8,evaluation_time=.1,evaluation_ros_time=1.1,odom_age=.7)]
            with patch('cbf_experiment_results.plot_results'):export_results(p,{0:rows},[],{})
            stats=json.loads((p/'experiment_statistics.json').read_text())['0']
            self.assertEqual(stats['evaluated_rows'],1);self.assertEqual(stats['excluded_measurement_rows'],2)

    def test_observer_reference_is_aligned_without_changing_raw_observer(self):
        n=self.node();n.metric_plan=None;n.metric_leader=LeaderMetricTimeline(.1)
        n.reference=TimedReference(0,radius=.5);velocity=np.array([.1,.05])
        for k in range(21):
            t=k*.1
            n.metric_leader.observe_clock(t,100.+t)
            n.metric_leader.observe_state(t,velocity*(t+.1),velocity,np.arctan2(velocity[1],velocity[0]))
            n.reference.observe(t,velocity*(t+.2),velocity)
        n.last_step=20;n.observer_source=102.;n.clock_sources[20]=102.;n.z=velocity*2.2
        old_z=n.z.copy();old_history=[(t,p.copy(),v.copy()) for t,p,v in n.reference.history]
        context=n.capture_metric_context(101.98,np.array([.8,.1,0.]))
        np.testing.assert_allclose(context['observer_aligned'],velocity*1.98,atol=1e-12)
        np.testing.assert_allclose(context['leader']['position'],velocity*1.98,atol=1e-12)
        np.testing.assert_array_equal(n.z,old_z)
        for old,new in zip(old_history,n.reference.history):
            self.assertEqual(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1]);np.testing.assert_array_equal(old[2],new[2])

    def test_stale_observer_is_not_reported_as_a_fresh_control_reference(self):
        n=self.node();n.metric_plan=None;n.metric_leader=LeaderMetricTimeline(.1)
        n.metric_leader.observe_clock(0.,99.9);n.metric_leader.observe_state(0.,[0.,0.],[.1,0.],0.)
        n.reference=TimedReference(0);n.reference.observe(0.,[0.,0.],[.1,0.])
        n.last_step=0;n.observer_source=98.;n.clock_sources={0:98.}
        context=n.capture_metric_context(100.,np.zeros(3))
        self.assertGreater(context['leader']['quality'],0)
        self.assertTrue(np.isnan(context['observer_aligned']).all())

    def node(self):
        inline.String=lambda **kw:SimpleNamespace(**kw)
        inline.RobotHistoryData=lambda **kw:SimpleNamespace(**kw)
        inline.rospy=SimpleNamespace(get_time=lambda:100.02,loginfo_throttle=Mock())
        n=inline.TurtleBot3CBFController.__new__(inline.TurtleBot3CBFController)
        n.closed=False;n.running=True;n.dt=.1;n.n=10;n.robot_id=0;n.name='tb3_0'
        velocity=.2*np.array([np.cos(.6),np.sin(.6)])
        n.latest_leader=(1,np.array([4.,*(np.array([1.,2.])+.12*velocity),*velocity,.6]),100.)
        n.clock_sources={1:100.};n.reference=SimpleNamespace(offset=lambda:np.array([.5,0.]))
        n.record_points=(np.empty((0,2)),100.);n.record_peer_count=0;n.stale=.6;n.cfg=dict(margin=.12,gamma=.3)
        n.manager=SimpleNamespace(last_records={});n.cycle_id=9;n.record_geometry_timing={};n.observed=[]
        n.snapshot_stamps=(99.98,99.99);n.raw_min_range=float('inf');n.dynamic_prediction=False
        n.input_ms=0.;n.manager_ms=0.;n.command_wall_ms=0.;n.cycle_validated_steps=10
        n.writer=Mock();n.rows=0;n.counts=Counter();n.stream=Mock();n.diagnostics_pub=Mock();n.history_pub=Mock()
        n.lock=threading.RLock();n.trace_lock=threading.RLock();n.trace_stream=Mock();n.cycle_cpu_begin=time.thread_time()
        n.applied_control=np.array([.1,0.]);n.last_command_time=100.02;n.z=np.array([9.,9.])
        n.metric_context=dict(stamp=99.98,leader=dict(position=np.array([1.,2.]),heading=.6,logical_time=3.98,quality=1),
            nominal=np.array([1.5,2.]),observer_raw=np.array([3.,4.]),observer_aligned=np.array([1.01,2.02]),observer_step=1,
            plan=dict(position=np.array([1.62,2.12]),valid=1,age=.1,cycle=8))
        return n

    def test_actual_record_separates_plan_nominal_and_observer_and_commits_only_after_sampling(self):
        n=self.node();state=np.array([1.6,2.1,.4]);solution=dict(success=True,plan_controls=np.tile([.2,0.],(10,1)))
        n.record(100.,state,None,[],.01,'ok',time.perf_counter(),solution,np.array([.1,0.]))
        row=n.writer.writerow.call_args[0][0]
        self.assertAlmostEqual(row['consensus_error_x'],.1);self.assertAlmostEqual(row['consensus_error_y'],.1)
        self.assertAlmostEqual(row['heading_error'],.2);self.assertAlmostEqual(row['legacy_heading_error'],-.2)
        self.assertAlmostEqual(row['plan_execution_error_x'],.02)
        self.assertAlmostEqual(row['observer_error_x'],.01)
        self.assertEqual(row['observer_x'],3.)
        self.assertEqual(n.history_pub.publish.call_args[0][0].zxeo_x,3.)
        self.assertEqual(n.metric_plan['cycle'],9)
        self.assertEqual(n.metric_plan['origin'],99.98)
        np.testing.assert_allclose(n.metric_plan['positions'][0],state[:2])
        trace=json.loads(n.trace_stream.write.call_args[0][0])
        self.assertEqual(trace['accepted_auxiliary_plan']['cycle'],9)
        np.testing.assert_allclose(trace['accepted_auxiliary_plan']['positions'][1],state[:2]+[.02,0.])

    def test_single_step_and_rejected_commands_do_not_create_auxiliary_reference(self):
        for status in ('validated_one_step','late_solution_stop','solver_failed_stop'):
            n=self.node();n.metric_plan={'old':True}
            n.record(100.,np.array([1.6,2.1,.4]),None,[],.01,status,time.perf_counter(),
                     dict(success=False,plan_controls=np.zeros((10,2))),np.array([.1,0.]))
            self.assertIsNone(n.metric_plan)

    def test_export_internal_metric_is_not_negative_formation_error(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'run_metadata.json').write_text(json.dumps(dict(timestamp='test',metrics_schema_version=2)))
            row=dict(time=1.,evaluation_time=.98,x=.6,y=.1,theta=.2,v=.1,omega=0.,
                consensus_error_x=.1,consensus_error_y=.1,observer_x=0.,observer_y=0.,solve_ms=20.,detected_ellipses=0,status='ok',
                reference_x=.5,reference_y=0.,plan_reference_x=.62,plan_reference_y=.11,
                plan_execution_error_x=.02,plan_execution_error_y=.01,plan_reference_valid=1,heading_error=-.2)
            with patch('cbf_experiment_results.plot_results'):data=export_results(p,{0:[row]},[],{})
            r=data['robots'][0]
            self.assertEqual(r['t'],[.98]);self.assertEqual(r['control_t'],[1.])
            self.assertEqual(r['xe'],[.1]);self.assertEqual(r['plan_execution_error_x'],[.02])
            self.assertEqual(r['xr'],[.5])
            self.assertEqual(data['chart_categories']['tracking_error_xy']['status'],'available')

    def test_legacy_nominal_alias_never_becomes_a_fake_internal_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'run_metadata.json').write_text(json.dumps(dict(timestamp='legacy')))
            row=dict(time=0.,x=.3,y=.2,theta=.1,v=.1,omega=0.,consensus_error_x=.04,
                consensus_error_y=.02,reference_x=.26,reference_y=.18,observer_x=.1,observer_y=.2,
                solve_ms=20.,detected_ellipses=0,status='ok')
            with patch('cbf_experiment_results.plot_results'):data=export_results(p,{0:[row]},[],{})
            self.assertEqual(data['chart_categories']['tracking_error_xy']['status'],'unavailable')
            self.assertTrue(np.isnan(data['robots'][0]['plan_execution_error_x'][0]))

    def test_legacy_converter_is_non_destructive_and_does_not_invent_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'old';source.mkdir();out=Path(folder)/'new'
            meta=dict(timestamp='old',method='MPC-D-CBF',control_dt=.1,formation_radius=0.,robot_names=['tb3_0'])
            (source/'run_metadata.json').write_text(json.dumps(meta))
            leader=[dict(trajectory_time=t,ros_time=10.+t,x0=.4+.15*max(0.,t),y0=0.,theta0=0.,u0x=.15,u0y=0.,v0=.15,w0=0.) for t in (0.,.1,.2)]
            def write(path,rows):
                with path.open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            write(source/'leader.csv',leader)
            row=dict(time=.2,ros_time=10.2,odom_age=.02,x=.45,y=0.,theta=.2,v=.15,omega=0.,
                consensus_error_x=.02,consensus_error_y=0.,heading_error=.2,reference_x=.43,reference_y=0.,
                observer_x=.4,observer_y=0.,solve_ms=20.,detected_ellipses=0,status='ok')
            write(source/'tb3_0.csv',[row]);before=hashlib.sha256((source/'tb3_0.csv').read_bytes()).hexdigest()
            convert(source,out,render=False)
            with (out/'tb3_0.csv').open() as stream:r=list(csv.DictReader(stream))[0]
            self.assertAlmostEqual(float(r['evaluation_time']),.18)
            self.assertAlmostEqual(float(r['consensus_error_x']),.023)
            self.assertAlmostEqual(float(r['heading_error']),-.2)
            self.assertEqual(float(r['plan_reference_valid']),0.)
            self.assertTrue(np.isnan(float(r['plan_execution_error_x'])))
            self.assertEqual(hashlib.sha256((source/'tb3_0.csv').read_bytes()).hexdigest(),before)
            with self.assertRaises(FileExistsError):convert(source,out,render=False)


if __name__=='__main__':unittest.main()
