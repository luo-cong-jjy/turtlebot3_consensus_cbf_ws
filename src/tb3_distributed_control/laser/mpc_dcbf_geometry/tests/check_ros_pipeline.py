#!/usr/bin/env python3
"""Isolated ROS/Gazebo checks. Always creates its own loopback ROS master.

After building/sourcing this package:
  python3 .../check_ros_pipeline.py synthetic
  python3 .../check_ros_pipeline.py gazebo
No GUI or robot velocity commands are used. Only this script's process groups
are stopped in cleanup; the user's ROS master and Gazebo are not contacted.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import xmlrpc.client

import numpy as np


def free_port():
    with socket.socket() as stream:
        stream.bind(('127.0.0.1', 0))
        return stream.getsockname()[1]


def wait_until(predicate, seconds, description):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError('Timeout: ' + description)


def stop(process):
    if process.poll() is not None:
        return
    for sig, timeout in [(signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 2)]:
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
        except ProcessLookupError:
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['synthetic', 'gazebo', 'both'])
    parser.add_argument('--report', default='')
    args = parser.parse_args()
    if args.mode == 'both':
        for mode in ('synthetic', 'gazebo'):
            command = [sys.executable, str(Path(__file__).resolve()), mode]
            if args.report:
                target = Path(args.report)
                command += ['--report', str(target.with_name(target.stem + '_' + mode + target.suffix))]
            subprocess.run(command, check=True)
        return
    directory = Path(tempfile.mkdtemp(prefix='tb3_geometry_' + args.mode + '_'))
    env = os.environ.copy()
    env.pop('ROS_HOSTNAME', None)
    env.pop('ROS_NAMESPACE', None)
    port = free_port()
    env.update(ROS_MASTER_URI=f'http://127.0.0.1:{port}', ROS_IP='127.0.0.1',
               ROS_HOME=str(directory / 'ros'), ROS_LOG_DIR=str(directory / 'roslog'),
               GAZEBO_MASTER_URI=f'http://127.0.0.1:{free_port()}',
               GAZEBO_MODEL_DATABASE_URI='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    os.environ.pop('ROS_HOSTNAME', None)
    os.environ.pop('ROS_NAMESPACE', None)
    os.environ.update(env)
    processes, streams = [], []
    result = {'mode': args.mode, 'log_directory': str(directory), 'master': env['ROS_MASTER_URI']}

    def launch(command, label):
        stream = (directory / (label + '.log')).open('w')
        streams.append(stream)
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        processes.append(process)
        return process

    try:
        launch(['roscore', '-p', str(port)], 'master')
        master = xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
        def master_ready():
            try:
                return master.getPid('/geometry_check')[0] == 1
            except (OSError, xmlrpc.client.Error):
                return False
        wait_until(master_ready, 12, 'isolated ROS master')
        import rospy
        import tf2_ros
        from geometry_msgs.msg import TransformStamped
        from sensor_msgs.msg import LaserScan
        from sensor_msgs import point_cloud2
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import Float32MultiArray
        from visualization_msgs.msg import MarkerArray, Marker
        from tb3_distributed_control.msg import LaserObstacleArray
        rospy.set_param('/use_sim_time', args.mode == 'gazebo')
        rospy.init_node('geometry_check_client', disable_signals=True)
        observations, markers, clouds, compatibility, scans = [], [], [], [], []
        subscribers = [
            rospy.Subscriber('/tb3_0/laser_geometry/obstacles', LaserObstacleArray, observations.append),
            rospy.Subscriber('/tb3_0/laser_geometry/ellipses', MarkerArray, markers.append),
            rospy.Subscriber('/tb3_0/laser_geometry/points', PointCloud2, clouds.append),
            rospy.Subscriber('/tb3_0/laser_geometry/for_obs_track', Float32MultiArray, compatibility.append),
            rospy.Subscriber('/tb3_0/scan', LaserScan, scans.append),
        ]
        package = Path(__file__).resolve().parents[3]
        if args.mode == 'synthetic':
            rospy.set_param('/tb3_0/laser_geometry/robot_frame_prefix', 'tb3_0')
            rospy.set_param('/tb3_0/laser_geometry/scan_timeout', .4)
            launch([sys.executable, str(package / 'laser/mpc_dcbf_geometry/laser_geometry_node.py'),
                    '__ns:=/tb3_0', '__name:=laser_geometry'], 'geometry')
            broadcaster = tf2_ros.StaticTransformBroadcaster()
            transform = TransformStamped()
            transform.header.frame_id = 'world'
            transform.header.stamp = rospy.Time.now()
            transform.child_frame_id = 'tb3_0/base_scan'
            transform.transform.translation.x = 1.
            transform.transform.translation.y = 2.
            transform.transform.translation.z = .172
            transform.transform.rotation.z = np.sin(.2)
            transform.transform.rotation.w = np.cos(.2)
            broadcaster.sendTransform(transform)
            publisher = rospy.Publisher('/tb3_0/scan', LaserScan, queue_size=1)
            wait_until(lambda: publisher.get_num_connections() >= 2, 12, 'scan subscriptions')
            def send(valid=True, frame='base_scan'):
                scan = LaserScan()
                scan.header.stamp = rospy.Time.now()
                scan.header.frame_id = frame
                scan.angle_min, scan.angle_increment = -.2, .02
                scan.angle_max = .2
                scan.range_min, scan.range_max = .12, 3.5
                scan.ranges = [1.0 if valid else float('inf')] * 21
                publisher.publish(scan)
            deadline = time.monotonic() + 12
            while len([m for m in observations if m.obstacles]) < 4 and time.monotonic() < deadline:
                send()
                time.sleep(.12)
            nonempty = [m for m in observations if m.obstacles]
            assert len(nonempty) >= 4, 'No geometry output'
            assert all(len(m.obstacles) == 1 for m in nonempty)
            assert len({m.obstacles[0].id for m in nonempty}) == 1, 'Unstable static ID'
            obstacle = nonempty[-1].obstacles[0]
            assert obstacle.semi_major >= obstacle.semi_minor > 0
            assert nonempty[-1].header.frame_id == 'world'
            wait_until(lambda: clouds and clouds[-1].width == 21, 3, 'transformed cloud')
            xyz = np.array(list(point_cloud2.read_points(clouds[-1], field_names=('x', 'y', 'z'))))
            angles = -.2 + np.arange(21)*.02 + .4
            expected = np.column_stack((1 + np.cos(angles), 2 + np.sin(angles), np.full(21, .172)))
            np.testing.assert_allclose(xyz, expected, atol=2e-6)
            wait_until(lambda: compatibility and len(compatibility[-1].data) == 7, 3, 'compatibility fields')
            wait_until(lambda: markers and any(m.ns == 'ellipse' and m.action == Marker.ADD
                                               for m in markers[-1].markers), 3, 'ellipse RViz markers')
            for _ in range(4):
                send(False)
                time.sleep(.12)
            wait_until(lambda: observations and not observations[-1].obstacles, 3, 'empty scan cleanup')
            assert any(m.action == Marker.DELETE for array in markers for m in array.markers)
            for _ in range(5):
                send()
                time.sleep(.12)
            wait_until(lambda: observations[-1].obstacles, 3, 'scan restart')
            time.sleep(.8)
            wait_until(lambda: not observations[-1].obstacles, 3, 'stale scan cleanup')
            # No transform exists for this frame. No fabricated obstacle output.
            for _ in range(5):
                send(frame='missing_laser')
                time.sleep(.12)
            assert not observations[-1].obstacles
            result.update(checks=['real ROS scan subscription', 'timestamped full TF including offset',
                                  'DBSCAN and ellipse output', 'stable Hungarian ID',
                                  'native and 7-field messages', 'RViz marker publication/deletion',
                                  'empty and stale scan cleanup', 'missing TF does not fabricate estimates'],
                          obstacles=1, point_count=21)
        else:
            launch(['roslaunch', str(package / 'launch/multi_turtlebot3_consensus_gazebo_one_tb3.launch'),
                    'gui:=false', 'open_rviz:=false', 'model:=burger'], 'gazebo_launch')
            wait_until(lambda: len([m for m in observations if len(m.obstacles) >= 3]) >= 3,
                       55, 'Gazebo scan -> geometry pipeline')
            native = observations[-1]
            assert native.header.frame_id == 'world' and scans
            assert all(o.semi_major >= o.semi_minor > 0 for o in native.obstacles)
            assert all(np.isfinite([o.pose.x, o.pose.y, o.pose.theta, o.semi_major,
                                   o.semi_minor, o.shape_variance]).all() for o in native.obstacles)
            assert len({o.id for o in native.obstacles}) == len(native.obstacles)
            state = master.getSystemState('/geometry_check')[2]
            nodes = {node for category in state for _, participants in category for node in participants}
            assert not any('controller' in n or 'leader_node' in n or 'ego_planner' in n for n in nodes)
            result.update(checks=['headless single-robot launch', 'Gazebo LaserScan and TF',
                                  'multiple finite obstacle estimates', 'unique IDs',
                                  'no controller/leader/EgoPlanner nodes'],
                          obstacle_count=len(native.obstacles), laser_frame=scans[-1].header.frame_id,
                          sample=[{'id': o.id, 'x': o.pose.x, 'y': o.pose.y,
                                   'a': o.semi_major, 'b': o.semi_minor} for o in native.obstacles[:5]])
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        if 'rospy' in locals():
            rospy.signal_shutdown('isolated geometry check complete')
        for process in reversed(processes):
            stop(process)
        for stream in streams:
            stream.close()
        if args.report:
            Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
