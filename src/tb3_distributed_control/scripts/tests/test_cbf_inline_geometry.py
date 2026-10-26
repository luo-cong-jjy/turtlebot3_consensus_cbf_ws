#!/usr/bin/env python3
"""Numerical equivalence and direct scan-to-controller handoff, without ROS master."""
import ast
from collections import Counter, deque
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError, cKDTree

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE/'scripts'))
sys.path.insert(0, str(PACKAGE/'纯仿真代码'))
from cbf_mpc_core import ObservedEnvelopes, peer_obstacle


def definitions(path, name):
    # Load actual production definitions without ROS imports/node startup.
    module = ModuleType(name)
    sys.modules[name] = module
    module.__dict__.update(np=np, deque=deque, dataclass=dataclass, math=math,
        linear_sum_assignment=linear_sum_assignment, ConvexHull=ConvexHull,
        QhullError=QhullError, cKDTree=cKDTree, peer_obstacle=peer_obstacle,time=time)
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


inline = definitions(PACKAGE/'scripts/tb3_controller_node_cbf.py', '_inline_geometry_test')
reference = definitions(PACKAGE/'laser/mpc_dcbf_geometry/mpc_dcbf_geometry_core.py',
                        '_previous_geometry_test')


class Stamp:
    current = 1.0

    def __init__(self, seconds=0.): self.seconds = float(seconds)
    def to_sec(self): return self.seconds
    def __sub__(self, other): return Stamp(self.seconds-other.seconds)
    def __lt__(self, other): return self.seconds < other.seconds
    def __eq__(self, other): return isinstance(other, Stamp) and self.seconds == other.seconds
    @classmethod
    def now(cls): return cls(cls.current)


class GeometryChecks(unittest.TestCase):
    def test_fit_clustering_transform_and_id_variance_match_previous(self):
        rng = np.random.default_rng(42)
        angle = np.linspace(-np.pi, np.pi, 50)
        clouds = [np.column_stack((.4*np.cos(angle), .2*np.sin(angle)))+[1., .5],
                  np.column_stack((np.linspace(0, .6, 20), np.zeros(20))),
                  rng.normal(size=(80, 2))*[.07, .03]+[-1., 1.]]
        old_tracker, new_tracker = reference.EllipseTracker(), inline.EllipseTracker()
        for step in range(4):
            points = np.vstack([p+step*.005 for p in clouds])
            old_labels = reference.dbscan(points, .18, 3)
            new_labels = inline.dbscan(points, .18, 3)
            np.testing.assert_array_equal(new_labels, old_labels)
            old = [reference.enclosing_ellipse(points[old_labels == k])
                   for k in sorted(set(old_labels)-{-1})]
            new = [inline.enclosing_ellipse(points[new_labels == k])
                   for k in sorted(set(new_labels)-{-1})]
            old_tracker.update(old, 1.+step*.1); new_tracker.update(new, 1.+step*.1)
            for a, b in zip(old, new):
                self.assertEqual(a.__dict__, b.__dict__)
                local = (points[new_labels == new.index(b)]-[b.cx,b.cy])
                c,s = np.cos(b.theta),np.sin(b.theta)
                local = local@np.array([[c,-s],[s,c]])
                self.assertLessEqual(float(np.max(np.sum((local/[b.a,b.b])**2,axis=1))), 1.+1e-8)
            q = [.1, .2, .3, .9]
            np.testing.assert_array_equal(inline.transform_points(points, [2,3,.1],q),
                                          reference.transform_points(points,[2,3,.1],q))

    def controller(self):
        inline.rospy = SimpleNamespace(Time=Stamp, Duration=Stamp,
            logwarn_throttle=Mock(), loginfo_throttle=Mock())
        inline.tf2_ros = SimpleNamespace(LookupException=type('LookupFailure',(Exception,),{}),
            ConnectivityException=type('ConnectivityFailure',(Exception,),{}),
            ExtrapolationException=type('ExtrapolationFailure',(Exception,),{}))
        node = inline.TurtleBot3CBFController.__new__(inline.TurtleBot3CBFController)
        node.closed = False; node.lock = threading.RLock()
        node.robot_id = 0; node.peer_radius = .20
        node.names = ['tb3_'+str(i) for i in range(5)]
        node.poses = {i:(1., np.array([10.+i,10.,0.]), np.zeros(2)) for i in range(5)}
        node.pose_history = {i:deque([entry]) for i,entry in node.poses.items()}
        node.peer_shapes = {}; node.peer_shape_history = {}; node.peer_laser_fits = Counter()
        node.target_frame = 'world'; node.frame_prefix = 'tb3_0'; node.frame_override = ''
        node.min_range = .12; node.max_range = 3.5; node.voxel_size = .02
        node.scan_timeout = .5; node.tf_timeout = .05
        node.static_accumulation = True; node.static_map_voxel = .04
        node.static_points = np.empty((0,2)); node.static_map_stamp = None
        node.robot_exclusion_radius = .16; node.eps = .18; node.min_samples = 3
        node.min_axis = .025; node.padding = .03; node.max_extent = 3.
        node.fit_tolerance = .01; node.fit_iterations = 300
        node.peer_fit_tolerance=.02;node.peer_fit_iterations=80;node.peer_fit_period=.2
        node.tracker = inline.EllipseTracker(); node.envelopes = ObservedEnvelopes()
        node.observed = []; node.geometry_stamp = None; node.last_success = None
        node.processed_scan = None; node.publish_geometry = Mock()
        node.geometry_timing_sequence=0
        node.publish_geometry_estimates=Mock()
        node.latest_scan = SimpleNamespace(header=SimpleNamespace(stamp=Stamp(1.), frame_id='base_scan'),
            ranges=[1.]*41, angle_min=-.2, angle_increment=.01, range_min=.12, range_max=3.5)
        transform = SimpleNamespace(transform=SimpleNamespace(
            translation=SimpleNamespace(x=1.,y=0.,z=.1),
            rotation=SimpleNamespace(x=0.,y=0.,z=0.,w=1.)))
        node.tf_buffer = SimpleNamespace(lookup_transform=Mock(return_value=transform))
        Stamp.current = 1.05
        return node

    def test_direct_scan_uses_scan_time_tf_and_conditions_covariance_once(self):
        node = self.controller(); node.process_scan(None)
        target, source, stamp, timeout = node.tf_buffer.lookup_transform.call_args[0]
        self.assertEqual((target, source, stamp.to_sec()), ('world','tb3_0/base_scan',1.))
        self.assertEqual(len(node.observed),1)
        self.assertEqual(node.geometry_stamp,1.)
        self.assertGreater(node.observed[0]['center'][0],1.8)
        covariance = node.envelopes.tracks[1][1].copy()
        node.process_scan(None)
        self.assertEqual(node.publish_geometry.call_count,1)
        # A distinct ROS message can carry the same scan stamp (bag replay).
        import copy
        node.latest_scan=copy.deepcopy(node.latest_scan)
        node.process_scan(None)
        self.assertEqual(node.publish_geometry.call_count,1)
        np.testing.assert_array_equal(covariance,node.envelopes.tracks[1][1])
        Stamp.current = 1.7; node.process_scan(None)
        self.assertIsNone(node.geometry_stamp)
        self.assertEqual(node.observed,[])
        self.assertEqual(node.publish_geometry.call_args[0][2],[])

    def test_peer_mask_and_missing_peer_data_protect_static_map(self):
        node = self.controller()
        del node.poses[4]
        del node.pose_history[4]
        node.process_scan(None)
        self.assertIsNone(node.geometry_stamp)
        self.assertEqual(len(node.static_points),0)
        node.poses[4]=(1., np.array([2.,0.,0.]),np.zeros(2))
        node.pose_history[4]=deque([node.poses[4]])
        node.process_scan(None)
        self.assertTrue(np.all(np.linalg.norm(node.static_points-[2.,0.],axis=1)>.16))
        self.assertEqual(node.geometry_stamp,1.)
        self.assertIn(4,node.peer_shapes)
        self.assertGreater(node.peer_laser_fits[4],0)

    def test_robot_returns_have_unique_ids_and_do_not_enter_static_accumulation(self):
        poses={0:(np.array([0.,0.,0.]),np.zeros(2)),
               1:(np.array([.4,0.,0.]),np.zeros(2)),
               2:(np.array([.4,.4,0.]),np.zeros(2))}
        points=np.array([[0.,.01],[.4,.03],[.4,-.03],[.4,.43],[2.,2.]])
        static,peers=inline.split_robot_returns(points,poses,0,.16)
        np.testing.assert_array_equal(static,[[2.,2.]])
        np.testing.assert_array_equal(peers[1],points[1:3])
        np.testing.assert_array_equal(peers[2],points[3:4])
        self.assertNotIn(0,peers)

    def test_effective_peer_ellipse_contains_full_prior_and_measured_ellipse(self):
        angles=np.linspace(-np.pi,np.pi,100,endpoint=False)
        points=np.column_stack((.13*np.cos(angles)+.07,.04*np.sin(angles)))+[1.,2.]
        state=np.array([1.,2.,.4])
        shape=inline.fit_peer_shape(points,state,.20,.025,.03,.01,300)
        shape['stamp']=2.
        obs=inline.predicted_peer_envelope(1,2.,state,np.array([.1,.6]),2.,.1,10,.20,shape)
        theta=obs['theta'];c,s=np.cos(theta),np.sin(theta)
        rotation=np.array([[c,-s],[s,c]])
        raw=np.column_stack((shape['raw_a']*np.cos(angles),shape['raw_b']*np.sin(angles)))@rotation.T+obs['measured_center']
        prior=np.column_stack((.20*np.cos(angles),.20*np.sin(angles)))+state[:2]
        for boundary in [raw,prior]:
            local=(boundary-obs['center'])@rotation
            self.assertLessEqual(float(np.max(np.sum((local/[obs['a'],obs['b']])**2,axis=1))),1.+1e-9)
        np.testing.assert_allclose(obs['prediction_angles'],theta+.06*np.arange(11),atol=1e-12)
        self.assertEqual(obs['peer_source'],'laser')
        fallback=inline.predicted_peer_envelope(1,2.,state,np.array([.1,.6]),2.6,.1,10,.20,shape)
        self.assertEqual(fallback['peer_source'],'odometry')
        self.assertEqual((fallback['a'],fallback['b']),(.20,.20))
        self.assertNotIn('measured_center',fallback)

    def test_scan_time_pose_interpolation_handles_angle_wrap(self):
        history=[(1.,np.array([0.,0.,np.deg2rad(179)]),np.zeros(2)),
                 (1.2,np.array([.2,0.,np.deg2rad(-179)]),np.zeros(2))]
        state,velocity=inline.odometry_at_stamp(history,1.1,.5)
        np.testing.assert_allclose(state,[.1,0.,np.pi],atol=1e-12)
        self.assertIsNone(inline.odometry_at_stamp(history,2.,.5))

    def test_throttled_fit_keeps_measurement_time_and_expires_when_occluded(self):
        node=self.controller()
        poses={1:(np.array([1.,0.,0.]),np.zeros(2))}
        points=np.array([[1.03,-.03],[1.04,0.],[1.03,.03],[1.,.04]])
        node.update_peer_shapes({1:points},poses,1.)
        self.assertEqual(node.peer_shapes[1]['stamp'],1.)
        count=node.peer_laser_fits[1]
        node.update_peer_shapes({1:points},poses,1.1)
        self.assertEqual(node.peer_shapes[1]['stamp'],1.)
        self.assertEqual(node.peer_laser_fits[1],count)
        node.update_peer_shapes({},poses,1.6)
        self.assertNotIn(1,node.peer_shapes)

    def test_launch_and_controller_have_no_external_perception_dependency(self):
        import xml.etree.ElementTree as ET
        launch = ET.parse(PACKAGE/'launch/multi_turtlebot3_consensus_gazebo_cbf.launch')
        self.assertFalse(launch.findall('.//node[@type="laser_geometry_node.py"]'))
        source=(PACKAGE/'scripts/tb3_controller_node_cbf.py').read_text()
        self.assertNotIn('mpc_dcbf_geometry',source)
        self.assertNotIn('LaserObstacleArray',source)
        self.assertNotIn('mpc_dcbf_geometry', (PACKAGE/'launch/multi_turtlebot3_consensus_gazebo_cbf.launch').read_text())


if __name__ == '__main__': unittest.main()
