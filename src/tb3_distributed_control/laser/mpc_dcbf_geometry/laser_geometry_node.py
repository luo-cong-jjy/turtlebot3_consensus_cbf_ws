#!/usr/bin/env python3
"""LaserScan obstacle geometry and RViz output; never publishes cmd_vel."""
from pathlib import Path
import sys
import threading

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs import point_cloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, Float32MultiArray, MultiArrayDimension
from visualization_msgs.msg import Marker, MarkerArray
from tb3_distributed_control.msg import LaserObstacle, LaserObstacleArray

# Support direct source execution and catkin's devel/install executable relay.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpc_dcbf_geometry_core import (EllipseTracker, dbscan, enclosing_ellipse,
                                    scan_points, transform_points, voxel_downsample)


class LaserGeometryNode:
    COLORS = ((0.1, 0.8, 1.0), (1.0, 0.35, 0.2), (0.3, 1.0, 0.3),
              (1.0, 0.3, 0.85), (0.8, 0.7, 0.2), (0.6, 0.4, 1.0))

    def __init__(self):
        get = rospy.get_param
        self.target_frame = get('~target_frame', 'world').lstrip('/')
        self.frame_prefix = get('~robot_frame_prefix', 'tb3_0').strip('/')
        self.frame_override = get('~scan_frame_override', '').lstrip('/')
        self.min_range = float(get('~min_range', 0.12))
        self.max_range = float(get('~max_range', 3.5))
        self.voxel_size = float(get('~voxel_size', 0.02))
        self.eps = float(get('~cluster_eps', 0.18))
        self.min_samples = int(get('~cluster_min_points', 3))
        self.max_extent = float(get('~max_cluster_extent', 3.0))
        self.min_axis = float(get('~min_semi_axis', 0.025))
        self.padding = float(get('~ellipse_padding', 0.03))
        self.fit_tolerance = float(get('~fit_tolerance', 0.01))
        self.fit_iterations = int(get('~fit_max_iterations', 300))
        self.display_margin = float(get('~display_margin', 0.10))
        self.scan_timeout = float(get('~scan_timeout', 0.5))
        self.tf_timeout = float(get('~tf_timeout', 0.05))
        rate = float(get('~processing_rate', 10.0))
        if (not self.target_frame or self.max_range <= self.min_range
                or self.min_range < 0 or self.eps <= 0 or self.min_samples < 1
                or self.max_extent <= 0 or self.min_axis <= 0 or self.padding < 0
                or self.display_margin < 0 or self.fit_tolerance <= 0
                or self.fit_iterations < 1 or self.scan_timeout <= 0
                or self.tf_timeout < 0 or not 0 < rate <= 100):
            raise ValueError('Invalid laser geometry configuration')
        self.tracker = EllipseTracker(float(get('~association_distance', 0.6)),
                                      float(get('~track_timeout', 1.0)),
                                      int(get('~shape_history_size', 10)))
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.cloud_pub = rospy.Publisher('~points', PointCloud2, queue_size=1)
        self.cluster_pub = rospy.Publisher('~clusters', MarkerArray, queue_size=1)
        self.ellipse_pub = rospy.Publisher('~ellipses', MarkerArray, queue_size=1)
        self.obstacle_pub = rospy.Publisher('~obstacles', LaserObstacleArray, queue_size=1)
        self.compat_pub = rospy.Publisher('~for_obs_track', Float32MultiArray, queue_size=1)
        self.lock = threading.Lock()
        self.latest_scan = None
        # Optional multi-robot integration: remove only measured peer-body returns.
        # Peer safety is then handled from odometry predictions by the controller.
        # Disabled for the original standalone geometry test.
        self.robot_exclusion_radius = float(get('~robot_exclusion_radius', 0.14))
        self.robot_positions = {}
        self.static_accumulation = bool(get('~static_accumulation', False))
        self.static_map_voxel = float(get('~static_map_voxel', 0.04))
        self.static_points = np.empty((0, 2))
        self.static_map_stamp = None
        self.robot_subscribers = [rospy.Subscriber(topic, Odometry, self.robot_callback,
            callback_args=topic, queue_size=1) for topic in get('~robot_exclusion_topics', [])]
        self.processed_scan = None
        self.last_success = None
        self.visible_ids = set()
        self.subscriber = rospy.Subscriber(get('~scan_topic', 'scan'), LaserScan,
                                           self.scan_callback, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self.process, reset=True)
        rospy.loginfo('Laser geometry: %s -> %s, DBSCAN eps=%.3f m; visible-contour ellipses',
                      self.subscriber.resolved_name, self.target_frame, self.eps)

    def scan_callback(self, scan):
        with self.lock:
            self.latest_scan = scan

    def robot_callback(self, message, key):
        # This option is for Gazebo's world-source odometry, as configured by
        # the CBF launch. Other frame conventions require explicit TF conversion.
        p = message.pose.pose.position
        with self.lock:
            self.robot_positions[key] = (message.header.stamp.to_sec(), np.array([p.x, p.y]))

    def process(self, _event):
        with self.lock:
            scan = self.latest_scan
        if scan is None:
            return
        now = rospy.Time.now()
        stamp = scan.header.stamp
        if stamp == rospy.Time(0):
            rospy.logwarn_throttle(5.0, 'Dropping zero-stamped scan: timestamped TF is required')
            self.clear_stale(now)
            return
        if now < stamp:
            return
        if (now - stamp).to_sec() > self.scan_timeout:
            self.clear_stale(now)
            return
        if scan is self.processed_scan:
            return
        source_frame = self.frame_override or scan.header.frame_id.lstrip('/')
        if not source_frame:
            rospy.logwarn_throttle(5.0, 'LaserScan has no frame_id')
            self.clear_stale(now)
            return
        if '/' not in source_frame and source_frame != self.target_frame and self.frame_prefix:
            source_frame = self.frame_prefix + '/' + source_frame
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, source_frame, stamp, rospy.Duration(self.tf_timeout))
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            points = scan_points(scan.ranges, scan.angle_min, scan.angle_increment,
                                 max(self.min_range, scan.range_min),
                                 min(self.max_range, scan.range_max))
            world_points = transform_points(points, [translation.x, translation.y, translation.z],
                                           [rotation.x, rotation.y, rotation.z, rotation.w])
            xy = voxel_downsample(world_points[:, :2], self.voxel_size)
            with self.lock:
                robot_positions = list(self.robot_positions.values())
            if self.static_accumulation and (len(robot_positions) < len(self.robot_subscribers)
                    or any(abs(stamp.to_sec()-s)>self.scan_timeout for s,p in robot_positions)):
                # Do not commit a robot body to the static map before peer masks
                # are available. This mode is explicitly for static-world tests.
                return
            for robot_stamp, position in robot_positions:
                if abs(stamp.to_sec()-robot_stamp) < self.scan_timeout:
                    xy = xy[np.linalg.norm(xy-position, axis=1) > self.robot_exclusion_radius]
            if self.static_accumulation:
                if self.static_map_stamp is not None and stamp.to_sec()<self.static_map_stamp:
                    self.static_points = np.empty((0, 2))
                self.static_points = voxel_downsample(np.vstack((self.static_points,xy)),self.static_map_voxel)
                xy = self.static_points
                self.static_map_stamp = stamp.to_sec()
            labels = dbscan(xy, self.eps, self.min_samples)
            fits, clusters = [], []
            for label in sorted(set(labels) - {-1}):
                cluster = xy[labels == label]
                if np.ptp(cluster, axis=0).max() > self.max_extent:
                    continue
                ellipse = enclosing_ellipse(cluster, self.min_axis, self.padding,
                                             self.fit_tolerance, self.fit_iterations)
                fits.append(ellipse)
                clusters.append(cluster)
            self.tracker.update(fits, stamp.to_sec())
            self.publish(stamp, world_points, fits, clusters)
            self.processed_scan = scan
            self.last_success = now
            rospy.loginfo_throttle(2.0, 'Laser geometry: %d returns, %d obstacle clusters',
                                  len(points), len(fits))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(3.0, 'Waiting for scan-time TF %s <- %s: %s',
                                  self.target_frame, source_frame, exc)
            self.clear_stale(now)
        except (ValueError, np.linalg.LinAlgError) as exc:
            rospy.logwarn_throttle(3.0, 'Invalid scan or geometry: %s', exc)
            self.clear_stale(now)

    def clear_stale(self, now):
        if self.last_success is not None and ((now - self.last_success).to_sec() > self.scan_timeout
                                             or now < self.last_success):
            self.tracker.update([], now.to_sec())
            self.publish(now, np.empty((0, 3)), [], [])
            self.last_success = None

    def marker(self, stamp, track_id, namespace, marker_type):
        marker = Marker()
        marker.header = Header(stamp=stamp, frame_id=self.target_frame)
        marker.ns, marker.id, marker.type = namespace, track_id, marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = rospy.Duration(max(0.6, self.scan_timeout))
        return marker

    def publish(self, stamp, points, ellipses, clusters):
        header = Header(stamp=stamp, frame_id=self.target_frame)
        self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(header, points.tolist()))
        cluster_markers, ellipse_markers = MarkerArray(), MarkerArray()
        message = LaserObstacleArray(header=header)
        compatibility = Float32MultiArray()
        for ellipse, cluster in zip(ellipses, clusters):
            color = self.COLORS[(ellipse.track_id - 1) % len(self.COLORS)]
            marker = self.marker(stamp, ellipse.track_id, 'clusters', Marker.POINTS)
            marker.scale.x = marker.scale.y = 0.045
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 1.0
            marker.points = [Point(x=float(p[0]), y=float(p[1]), z=0.07) for p in cluster]
            cluster_markers.markers.append(marker)
            for namespace, margin, alpha in [('ellipse', 0.0, 1.0),
                                              ('display_margin', self.display_margin, 0.4)]:
                if namespace == 'display_margin' and margin == 0:
                    continue
                marker = self.marker(stamp, ellipse.track_id, namespace, Marker.LINE_STRIP)
                marker.scale.x = 0.025
                marker.color.r, marker.color.g, marker.color.b = color
                marker.color.a = alpha
                marker.points = [Point(x=float(p[0]), y=float(p[1]), z=0.09)
                                 for p in ellipse.contour(margin=margin)]
                ellipse_markers.markers.append(marker)
            marker = self.marker(stamp, ellipse.track_id, 'labels', Marker.TEXT_VIEW_FACING)
            marker.pose.position.x, marker.pose.position.y = ellipse.cx, ellipse.cy
            marker.pose.position.z = 0.35
            marker.scale.z = 0.14
            marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.0
            marker.text = f'#{ellipse.track_id} a={ellipse.a:.2f} b={ellipse.b:.2f}'
            ellipse_markers.markers.append(marker)
            obstacle = LaserObstacle()
            obstacle.id = ellipse.track_id
            obstacle.pose.x, obstacle.pose.y, obstacle.pose.theta = ellipse.cx, ellipse.cy, ellipse.theta
            obstacle.semi_major, obstacle.semi_minor = ellipse.a, ellipse.b
            obstacle.shape_variance = ellipse.variance
            obstacle.point_count = ellipse.point_count
            message.obstacles.append(obstacle)
            compatibility.data.extend([ellipse.cx, ellipse.cy, ellipse.a, ellipse.b,
                                        ellipse.theta, ellipse.track_id, ellipse.variance])
        current_ids = {e.track_id for e in ellipses}
        for track_id in self.visible_ids - current_ids:
            for namespace, output in [('clusters', cluster_markers), ('ellipse', ellipse_markers),
                                       ('display_margin', ellipse_markers), ('labels', ellipse_markers)]:
                marker = self.marker(stamp, track_id, namespace, Marker.LINE_STRIP)
                marker.action = Marker.DELETE
                output.markers.append(marker)
        self.visible_ids = current_ids
        compatibility.layout.dim = [MultiArrayDimension(label='obstacles', size=len(ellipses),
                                                        stride=7 * len(ellipses)),
                                     MultiArrayDimension(label='cx,cy,a,b,theta,id,variance', size=7, stride=7)]
        self.cluster_pub.publish(cluster_markers)
        self.ellipse_pub.publish(ellipse_markers)
        self.obstacle_pub.publish(message)
        self.compat_pub.publish(compatibility)


if __name__ == '__main__':
    rospy.init_node('laser_geometry')
    try:
        node = LaserGeometryNode()
        rospy.spin()
    except (ValueError, rospy.ROSInterruptException) as exc:
        rospy.logfatal('Laser geometry stopped: %s', exc)
        if not isinstance(exc, rospy.ROSInterruptException):
            sys.exit(1)
