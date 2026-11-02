#!/usr/bin/env python3
"""Distributed observer + unified MPC-D-CBF, driven by Gazebo odometry/laser geometry.

The original PDNN/CAPA node is intentionally untouched. No world obstacle truth is
read here. Optional peer motion predictions use ordinary odometry messages.
"""
import csv
import json
import os
from pathlib import Path
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
import math

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import rospy
import rospkg
import tf2_ros
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError, cKDTree
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Float64MultiArray, Bool, String, Header
from tb3_distributed_control.msg import RobotHistoryData, LeaderHistoryData
from tf.transformations import euler_from_quaternion

PACKAGE = Path(rospkg.RosPack().get_path("tb3_distributed_control"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if (PACKAGE / "纯仿真代码").is_dir():
    sys.path.insert(0, str(PACKAGE / "纯仿真代码"))
from cbf_mpc_core import (default_config, TimedReference, ObservedEnvelopes,
                          peer_obstacle, padded_obstacles,LeaderMetricTimeline,
                          accepted_plan_reference,sample_plan_reference,wrap_angle,METRICS_SCHEMA_VERSION)
from mpc_dcbf_cached_solver import MPCSolverManager, CachedMPCProblem, barrier_values
from mpc_dcbf_static_obstacles import reference_covariance_matrices, reference_axis_inflation


# Inlined 2-D geometry: DBSCAN -> enclosing ellipse -> ID association.
# Same numerical implementation as the previous standalone perception path;
# no perception module import, subprocess, or obstacle-message subscription.
def scan_points(ranges, angle_min, angle_increment, range_min, range_max):
    """Finite returns in [range_min, range_max), in the laser frame."""
    values = np.asarray(ranges, dtype=float)
    if values.ndim != 1 or not np.isfinite([angle_min, angle_increment]).all():
        raise ValueError('Invalid scan angles or range array')
    if range_min < 0 or range_max <= range_min or angle_increment == 0:
        raise ValueError('Invalid scan range limits or angle increment')
    angles = angle_min + np.arange(values.size) * angle_increment
    valid = np.isfinite(values) & (values >= range_min) & (values < range_max)
    return np.column_stack((values[valid] * np.cos(angles[valid]),
                            values[valid] * np.sin(angles[valid])))


def transform_points(points, translation, quaternion):
    """Full rigid laser-to-target transform; quaternion order is [x,y,z,w]."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    translation = np.asarray(translation, dtype=float).reshape(3)
    q = np.asarray(quaternion, dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if not np.isfinite(translation).all() or not np.isfinite(q).all() or norm < 1e-12:
        raise ValueError('Invalid rigid transform')
    x, y, z, w = q / norm
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    xyz = np.column_stack((points, np.zeros(len(points))))
    return xyz @ rotation.T + translation


def voxel_downsample(points, size):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(points) or size <= 0:
        return points.copy()
    _, inverse = np.unique(np.floor(points / size), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    return np.column_stack([np.bincount(inverse, weights=points[:, d]) / counts
                            for d in range(2)])


def dbscan(points, eps=0.18, min_samples=3):
    """Euclidean DBSCAN; min_samples includes the query point itself."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if eps <= 0 or min_samples < 1:
        raise ValueError('DBSCAN eps and min_samples must be positive')
    if not np.isfinite(points).all():
        raise ValueError('DBSCAN requires finite points')
    labels = np.full(len(points), -1, dtype=int)
    if not len(points):
        return labels
    neighbours = cKDTree(points).query_ball_point(points, eps)
    visited = np.zeros(len(points), dtype=bool)
    cluster_id = 0
    for seed in range(len(points)):
        if visited[seed]:
            continue
        visited[seed] = True
        if len(neighbours[seed]) < min_samples:
            continue
        labels[seed] = cluster_id
        pending = deque(neighbours[seed])
        queued = set(neighbours[seed])
        while pending:
            index = pending.popleft()
            if not visited[index]:
                visited[index] = True
                if len(neighbours[index]) >= min_samples:
                    for other in neighbours[index]:
                        if other not in queued:
                            pending.append(other)
                            queued.add(other)
            if labels[index] == -1:
                labels[index] = cluster_id
        cluster_id += 1
    return labels


@dataclass
class Ellipse:
    cx: float
    cy: float
    a: float
    b: float
    theta: float
    point_count: int
    track_id: int = 0
    variance: float = 0.0

    def contour(self, count=80, margin=0.0):
        angle = np.linspace(0.0, 2.0 * np.pi, count + 1)
        local = np.column_stack(((self.a + margin) * np.cos(angle),
                                 (self.b + margin) * np.sin(angle)))
        c, s = math.cos(self.theta), math.sin(self.theta)
        return local @ np.array([[c, s], [-s, c]]) + [self.cx, self.cy]


def enclosing_ellipse(points, min_axis=0.025, padding=0.03,
                      tolerance=0.01, max_iterations=300):
    """Approximate MVEE with a finite enclosing fallback for point/line scans.

    a >= b are semi-axis lengths; theta is the major axis's CCW angle modulo pi.
    An ellipse encloses observed returns, not the unseen portion of the object.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('Ellipse requires nonempty finite points')
    if min_axis <= 0 or padding < 0 or tolerance <= 0 or max_iterations < 1:
        raise ValueError('Invalid ellipse fitting parameters')
    unique = np.unique(points, axis=0)
    origin = unique.mean(axis=0)
    scale = max(float(np.ptp(unique, axis=0).max()), min_axis)
    normalized = (unique - origin) / scale
    _, singular, vt = np.linalg.svd(normalized, full_matrices=True)
    rank_two = len(singular) >= 2 and singular[1] > max(1e-10, singular[0] * 1e-6)
    fitted = False
    if rank_two and len(unique) >= 3:
        try:
            support = normalized[ConvexHull(normalized).vertices]
            q = np.vstack((support.T, np.ones(len(support))))
            weights = np.full(len(support), 1.0 / len(support))
            for _ in range(max_iterations):
                matrix = (q * weights) @ q.T
                leverage = np.sum(q * np.linalg.solve(matrix, q), axis=0)
                index = int(np.argmax(leverage))
                maximum = float(leverage[index])
                if maximum <= 3.0 * (1.0 + tolerance):
                    break
                step = (maximum - 3.0) / (3.0 * (maximum - 1.0))
                weights *= 1.0 - step
                weights[index] += step
            center = support.T @ weights
            covariance = (support.T * weights) @ support - np.outer(center, center)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            if np.min(eigenvalues) > 1e-14:
                axes = np.sqrt(2.0 * eigenvalues[::-1]) * scale
                rotation = eigenvectors[:, ::-1]
                center = origin + center * scale
                fitted = True
        except (QhullError, np.linalg.LinAlgError):
            pass
    if not fitted:
        # PCA only regularizes degenerate observations; it is not the main fit.
        rotation = vt.T
        projected = (unique - origin) @ rotation
        lo, hi = projected.min(axis=0), projected.max(axis=0)
        center = origin + ((lo + hi) * 0.5) @ rotation.T
        axes = (hi - lo) * 0.5
    axes = np.maximum(axes, min_axis)
    local = (unique - center) @ rotation
    radius = float(np.sqrt(np.sum((local / axes) ** 2, axis=1)).max())
    axes *= max(1.0, radius) * (1.0 + 1e-9)
    axes += padding
    if axes[1] > axes[0]:
        axes = axes[::-1]
        rotation = rotation[:, ::-1]
    theta = math.atan2(rotation[1, 0], rotation[0, 0])
    theta = (theta + np.pi * 0.5) % np.pi - np.pi * 0.5
    return Ellipse(float(center[0]), float(center[1]), float(axes[0]),
                   float(axes[1]), float(theta), len(points))


class EllipseTracker:
    """Bounded-memory, one-to-one Hungarian matching in a fixed frame."""
    def __init__(self, max_distance=0.6, timeout=1.0, history_size=10):
        if max_distance <= 0 or timeout <= 0 or history_size < 1:
            raise ValueError('Invalid tracker parameters')
        self.max_distance = max_distance
        self.timeout = timeout
        self.history_size = history_size
        self.tracks = {}
        self.next_id = 1
        self.last_stamp = None

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.last_stamp = None

    def update(self, ellipses, stamp):
        if self.last_stamp is not None and stamp < self.last_stamp:
            self.reset()
        self.last_stamp = stamp
        self.tracks = {i: t for i, t in self.tracks.items()
                       if stamp - t['stamp'] <= self.timeout}
        track_ids = sorted(self.tracks)
        assigned = {}
        if ellipses and track_ids:
            centers = np.array([[e.cx, e.cy] for e in ellipses])
            old = np.array([self.tracks[i]['center'] for i in track_ids])
            distance = np.linalg.norm(centers[:, None] - old[None, :], axis=2)
            penalty = (len(ellipses) + 1) * self.max_distance
            cost = np.full((len(ellipses), len(track_ids) + len(ellipses)), penalty)
            cost[:, :len(track_ids)] = np.where(distance <= self.max_distance,
                                               distance, penalty * 100)
            rows, columns = linear_sum_assignment(cost)
            assigned = {r: track_ids[c] for r, c in zip(rows, columns)
                        if c < len(track_ids) and distance[r, c] <= self.max_distance}
        for index, ellipse in enumerate(ellipses):
            track_id = assigned.get(index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {'history': deque(maxlen=self.history_size)}
            track = self.tracks[track_id]
            track.update(center=np.array([ellipse.cx, ellipse.cy]), stamp=stamp)
            track['history'].append([ellipse.a, ellipse.b])
            history = np.asarray(track['history'])
            variance = float(np.var(history, axis=0).sum()) ** 1.3
            ellipse.track_id = track_id
            ellipse.variance = variance + max(0, 3 - len(history)) * 0.0005
        return ellipses


def constant_twist_pose(state, velocity, elapsed):
    """Align timestamped odometry in SE(2), also for short backward alignment."""
    result = np.asarray(state, float).copy()
    v, w = velocity
    theta = result[2]
    half_turn = .5*w*elapsed
    distance = v*elapsed*np.sinc(half_turn/np.pi)
    result[:2] += distance*np.array([np.cos(theta+half_turn), np.sin(theta+half_turn)])
    result[2] += w*elapsed
    return result


def odometry_at_stamp(history, stamp, max_extrapolation):
    """Interpolate scan-time poses; bound extrapolation when the next odom is late."""
    if not history:
        return None
    times = np.array([entry[0] for entry in history])
    right = int(np.searchsorted(times, stamp, side='right'))
    if 0 < right < len(history):
        t0, p0, v0 = history[right-1]
        t1, p1, v1 = history[right]
        weight = (stamp-t0)/max(t1-t0, 1e-12)
        delta = p1-p0
        delta[2] = (delta[2]+np.pi)%(2*np.pi)-np.pi
        return p0+weight*delta, (1-weight)*v0+weight*v1
    source, pose, velocity = history[0 if right == 0 else -1]
    if abs(stamp-source) > max_extrapolation:
        return None
    return constant_twist_pose(pose, velocity, stamp-source), velocity.copy()


def split_robot_returns(points, robot_poses, own_id, association_radius):
    """Assign each return to at most one known robot; only the rest is static."""
    ids = sorted(robot_poses)
    if not ids or not len(points):
        return points.copy(), {}
    centers = np.array([robot_poses[i][0][:2] for i in ids])
    distance = np.linalg.norm(points[:, None, :]-centers[None, :, :], axis=2)
    closest = np.argmin(distance, axis=1)
    matched = distance[np.arange(len(points)), closest] <= association_radius
    peers = {i: points[matched & (closest == j)] for j,i in enumerate(ids) if i != own_id}
    return points[~matched], peers


def fit_peer_shape(points, state, radius, min_axis, padding, tolerance, iterations):
    """Fit the visible contour, then conservatively recenter it on odometry.

    The effective ellipse contains BOTH the complete measured ellipse and the
    previous radius-sized body prior. A partial LDS-shell arc cannot shrink the
    robot to the size of its visible sensor. Shape is stored in the body frame.
    """
    fit = enclosing_ellipse(points, min_axis, padding, tolerance, iterations)
    c,s = np.cos(fit.theta), np.sin(fit.theta)
    local_offset = (np.array([fit.cx,fit.cy])-state[:2])@np.array([[c,-s],[s,c]])
    axes = np.maximum([fit.a,fit.b], radius)
    # Triangle inequality bounds every point of the translated measured ellipse.
    scale = max(1., np.linalg.norm(local_offset/axes)+max(fit.a/axes[0],fit.b/axes[1]))
    axes *= scale
    c,s = np.cos(state[2]), np.sin(state[2])
    return dict(a=float(axes[0]), b=float(axes[1]),
        body_angle=float((fit.theta-state[2]+np.pi/2)%np.pi-np.pi/2),
        body_offset=(np.array([fit.cx,fit.cy])-state[:2])@np.array([[c,-s],[s,c]]),
        raw_a=fit.a, raw_b=fit.b, point_count=fit.point_count)


def predicted_peer_envelope(peer_id, stamp, state, velocity, now, dt, horizon,
                            radius, shape=None, shape_timeout=.5):
    """Odometry drives translation/rotation; fresh laser data supplies the shape."""
    initial = constant_twist_pose(state, velocity, max(0., now-stamp))
    obstacle = peer_obstacle(peer_id, initial, velocity, dt, horizon, radius)
    obstacle['peer_source'] = 'odometry'
    if shape is not None and 0 <= now-shape['stamp'] <= shape_timeout:
        angles = initial[2]+np.arange(horizon+1)*dt*velocity[1]+shape['body_angle']
        c,s = np.cos(initial[2]),np.sin(initial[2])
        obstacle.update(a=shape['a'], b=shape['b'], theta=float(angles[0]),
            prediction_angles=angles, peer_source='laser',
            measured_center=initial[:2]+shape['body_offset']@np.array([[c,s],[-s,c]]),
            measured_a=shape['raw_a'], measured_b=shape['raw_b'],
            measured_point_count=shape['point_count'], measurement_age=now-shape['stamp'])
    return obstacle


class DynamicEllipseTracker:
    """Local measured ellipses -> predictive association -> upstream 9-state KF.

    CV state order and Q/R/P follow MPC-D-CBF obs_param/obs_kf.cpp. CA uses
    the two previously unused states as ax/ay with a white-jerk process model.
    Unlike its
    fixed-rate callback, actual scan stamps determine elapsed time. Predictions
    include node zero and are re-aligned to the control stamp without updating
    the filter twice. No simulation or peer truth enters this estimator.
    """
    def __init__(self, dt=.1, horizon=10, association_distance=.65, timeout=.65,
                 history_size=10, speed_threshold=.05, mode='cv',reaction_time=.35,extra_margin=.02,
                 ca_jerk_density=.01,ca_acceleration_variance=.04,singer_tau=1.,
                 acceleration_buffer=True,stage_velocity_guidance=True):
        if dt<=0 or horizon<1 or association_distance<=0 or timeout<=0 or mode not in ('cv','frozen','ca','singer'):
            raise ValueError('Invalid dynamic tracker configuration')
        self.dt,self.horizon,self.max_distance,self.timeout=dt,horizon,association_distance,timeout
        self.history_size,self.speed_threshold,self.mode=history_size,speed_threshold,mode
        if reaction_time<0 or extra_margin<0:raise ValueError('Negative execution margin')
        self.reaction_time,self.extra_margin=reaction_time,extra_margin
        self.A,self.Q,self.R,self.H,self.P0=reference_covariance_matrices(dt)
        if not all(np.isfinite(v) and v>0 for v in (ca_jerk_density,ca_acceleration_variance)):
            raise ValueError('CA jerk density and acceleration prior must be finite and positive')
        self.ca_jerk_density=ca_jerk_density
        self.ca_acceleration_variance=ca_acceleration_variance
        if not np.isfinite(singer_tau) or singer_tau<=0:raise ValueError('Singer tau must be finite and positive')
        self.singer_tau=singer_tau
        self.acceleration_buffer=bool(acceleration_buffer)
        self.stage_velocity_guidance=bool(stage_velocity_guidance)
        if mode in ('ca','singer'):self.P0[7,7]=self.P0[8,8]=ca_acceleration_variance
        # Positive quadrature weights retain PSD of the diffusion integral.
        self.quadrature_nodes,self.quadrature_weights=np.polynomial.legendre.leggauss(12)
        self.tracks={};self.next_id=1;self.last_stamp=None

    def reset(self):
        self.tracks.clear();self.last_stamp=None;self.next_id=1

    def propagate(self,state,covariance,elapsed):
        if not np.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid propagation time')
        a=np.eye(9);a[0,5]=a[1,6]=elapsed
        if self.mode=='singer':
            h=elapsed;tau=self.singer_tau
            a[0,7]=a[1,8]=tau*tau*(h/tau+np.expm1(-h/tau))
            a[5,7]=a[6,8]=-tau*np.expm1(-h/tau)
            a[7,7]=a[8,8]=np.exp(-h/tau)
            u=.5*h*(self.quadrature_nodes+1.)
            g=np.column_stack((tau*tau*(u/tau+np.expm1(-u/tau)),
                               -tau*np.expm1(-u/tau),np.exp(-u/tau)))
            block=self.ca_jerk_density*.5*h*(g.T@(self.quadrature_weights[:,None]*g))
            if h/tau>1.:
                # Closed-form integral avoids resolving a narrow exponential
                # boundary layer when the elapsed time greatly exceeds tau.
                z=h/tau;e=np.exp(-z);d=-np.expm1(-z);d2=-np.expm1(-2*z)
                block=self.ca_jerk_density*np.array([
                    [tau**5*(z**3/3-z*z+z-2*z*e+.5*d2),
                     tau**4*(z*z/2-z+z*e+d-.5*d2),tau**3*(-z*e+.5*d2)],
                    [tau**4*(z*z/2-z+z*e+d-.5*d2),
                     tau**3*(z-2*d+.5*d2),tau*tau*(d-.5*d2)],
                    [tau**3*(-z*e+.5*d2),tau*tau*(d-.5*d2),tau*.5*d2]])
            q=np.zeros((9,9))
            for indices in ([0,5,7],[1,6,8]):q[np.ix_(indices,indices)]=block
            for i in (2,3,4):q[i,i]=self.Q[i,i]*(h/self.dt)
            return a@state,a@covariance@a.T+q
        if self.mode=='ca':
            h=elapsed
            a[0,7]=a[1,8]=.5*h*h;a[5,7]=a[6,8]=h
            # Exact discretization of independent continuous white jerk. The
            # geometric states retain the upstream random-walk covariance.
            block=self.ca_jerk_density*np.array([[h**5/20,h**4/8,h**3/6],
                [h**4/8,h**3/3,h*h/2],[h**3/6,h*h/2,h]])
            q=np.zeros((9,9))
            for indices in ([0,5,7],[1,6,8]):q[np.ix_(indices,indices)]=block
            for i in (2,3,4):q[i,i]=self.Q[i,i]*(h/self.dt)
            return a@state,a@covariance@a.T+q
        return a@state,a@covariance@a.T+self.Q*(max(0.,elapsed)/self.dt)

    def update(self,ellipses,stamp,forced_ids=None):
        if forced_ids is not None and (len(forced_ids)!=len(ellipses) or len(set(forced_ids))!=len(forced_ids)):
            raise ValueError('Common association IDs must be unique per frame')
        if not np.isfinite(stamp):raise ValueError('Non-finite scan time')
        if self.last_stamp is not None and stamp<self.last_stamp:self.reset()
        if self.last_stamp is not None and stamp<=self.last_stamp:return ellipses
        self.last_stamp=stamp
        self.tracks={i:t for i,t in self.tracks.items() if stamp-t['seen']<=self.timeout}
        ids=sorted(self.tracks)
        for track in self.tracks.values():
            track['x'],track['P']=self.propagate(track['x'],track['P'],stamp-track['stamp'])
            track['stamp']=stamp;track['missed']+=1
        assigned={}
        if forced_ids is not None:assigned=dict(enumerate(forced_ids))
        elif ellipses and ids:
            centers=np.array([[e.cx,e.cy] for e in ellipses])
            predicted=np.array([self.tracks[i]['x'][:2] for i in ids])
            distances=np.linalg.norm(centers[:,None,:]-predicted[None,:,:],axis=2)
            # Dummy columns permit unmatched measurements; gated pairs cannot
            # steal the only valid association of another detection.
            cost=np.full((len(ellipses),len(ids)+len(ellipses)),self.max_distance*2)
            cost[:,:len(ids)]=np.where(distances<=self.max_distance,distances,self.max_distance*100)
            rows,cols=linear_sum_assignment(cost)
            assigned={r:ids[c] for r,c in zip(rows,cols) if c<len(ids) and distances[r,c]<=self.max_distance}
        for index,ellipse in enumerate(ellipses):
            key=assigned.get(index)
            z=np.array([ellipse.cx,ellipse.cy,ellipse.a,ellipse.b,ellipse.theta])
            if key is None or key not in self.tracks:
                if key is None:key=self.next_id;self.next_id+=1
                else:self.next_id=max(self.next_id,key+1)
                x=np.zeros(9);x[:5]=z
                self.tracks[key]=dict(x=x,P=self.P0.copy(),stamp=stamp,seen=stamp,
                    history=deque(maxlen=self.history_size),count=0,moving_votes=0,dynamic=False,missed=0)
            track=self.tracks[key];track['history'].append([ellipse.a,ellipse.b])
            variance=float(np.var(np.asarray(track['history']),axis=0).sum())**1.3
            variance+=max(0,3-len(track['history']))*.0005
            r=self.R.copy();r[0,0]*=3000.**np.log10(np.clip(variance,.005,.05)/.005);r[1,1]=r[0,0]
            # Major-axis direction is modulo pi, not modulo 2*pi.
            delta=z[4]-track['x'][4]
            z[4]=track['x'][4]+.5*np.arctan2(np.sin(2*delta),np.cos(2*delta))
            p=track['P'];gain=np.linalg.solve(self.H@p@self.H.T+r,self.H@p).T
            track['x']+=gain@(z-self.H@track['x'])
            residual=np.eye(9)-gain@self.H
            track['P']=residual@p@residual.T+gain@r@gain.T  # Joseph form
            track['P']=(track['P']+track['P'].T)/2
            track['x'][2:4]=np.maximum(track['x'][2:4],.025)
            track.update(seen=stamp,missed=0,variance=variance,count=track['count']+1,
                         raw_a=ellipse.a,raw_b=ellipse.b,raw_center=z[:2].copy(),raw_theta=z[4],point_count=ellipse.point_count)
            speed=float(np.linalg.norm(track['x'][5:7]))
            track['moving_votes']=max(-3,min(3,track['moving_votes']+(1 if speed>self.speed_threshold else -1)))
            if track['moving_votes']>=2:track['dynamic']=True
            elif track['moving_votes']<=-2:track['dynamic']=False
            ellipse.track_id=key;ellipse.variance=variance
        return ellipses

    def snapshot(self,now):
        return [dict(t,id=i,x=t['x'].copy(),P=t['P'].copy()) for i,t in sorted(self.tracks.items())
                if -1e-9<=now-t['seen']<=self.timeout]

    def forecast(self,track,now):
        if not -1e-9<=now-track['seen']<=self.timeout:return None
        x,p=self.propagate(track['x'],track['P'],max(0.,now-track['stamp']))
        if self.mode=='frozen':x[5:7]=0.
        motion=dict(mode=self.mode,position=x[:2].tolist(),velocity=x[5:7].tolist(),
            acceleration=x[7:9].tolist(),tau=self.singer_tau)
        centers=[];axes=[];velocities=[];position_covariances=[];margins=[]
        acceleration=x[7:9].copy() if self.mode in ('ca','singer') else np.zeros(2)
        # Enclose the most recent measured ellipse even if the mean lags it.
        c,s=np.cos(x[4]-track['raw_theta']),np.sin(x[4]-track['raw_theta'])
        rotation=np.array([[c,-s],[s,c]])
        base=np.maximum(x[2:4],[track['raw_a'],track['raw_b']])
        c,s=np.cos(x[4]),np.sin(x[4]);local=(track['raw_center']-x[:2])@np.array([[c,-s],[s,c]])
        shape=np.diag(1./base)@rotation@np.diag([track['raw_a'],track['raw_b']])
        scale=max(1.,np.linalg.norm(local/base)+np.linalg.norm(shape,2))
        shape_guard=base*scale
        # Separate from upstream stochastic inflation: a deterministic execution
        # buffer for model/command tracking lag. Uniform radial scaling adds
        # exactly this margin to the existing b*(rho-1) barrier, not an
        # unverified a+=d,b+=d approximation of an ellipse Minkowski sum.
        execution_margin=self.extra_margin+self.reaction_time*np.linalg.norm(track['x'][5:7])
        for j in range(self.horizon+1):
            if j:x,p=self.propagate(x,p,self.dt)
            centers.append(x[:2].copy())
            velocities.append(x[5:7].copy());position_covariances.append(p[:2,:2].copy())
            margin=execution_margin
            if self.mode in ('ca','singer') and self.acceleration_buffer:
                # Reaction displacement under estimated acceleration, an
                # engineering buffer, NOT a certified acceleration bound.
                margin=self.extra_margin+self.reaction_time*np.linalg.norm(x[5:7])+\
                    .5*self.reaction_time**2*np.linalg.norm(x[7:9])
            margins.append(float(margin))
            inflated=shape_guard+reference_axis_inflation(p)
            axes.append(inflated*(1.+margin/inflated[1]))
        return dict(id=track['id'],name='tracked_'+str(track['id']),center=np.asarray(centers[0]),
            a=float(axes[0][0]),b=float(axes[0][1]),theta=float(x[4]),
            prediction_centers=np.asarray(centers),prediction_axes=np.asarray(axes),
            prediction_model=self.mode,stage_velocity_guidance=self.stage_velocity_guidance,
            guidance_motion=dict(stamp=now,weights=[1.],models=[motion]),
            prediction_velocities=np.asarray(velocities),
            prediction_position_covariances=np.asarray(position_covariances),
            prediction_execution_margins=np.asarray(margins),acceleration=acceleration,
            velocity=track['x'][5:7].copy(),dynamic=track['dynamic'],
            execution_margin=margins[0],
            guidance=not track['dynamic'],seen=track['seen'],measurement_age=now-track['seen'],
            raw_a=track['raw_a'],raw_b=track['raw_b'],point_count=track['point_count'],missed=track['missed'])

    def obstacles(self,snapshot,now):
        return [obs for obs in (self.forecast(t,now) for t in snapshot) if obs is not None]


def mixture_moments(means,covariances,weights):
    """Moment matching includes between-model scatter; not a confidence bound."""
    means=np.asarray(means);weights=np.asarray(weights)
    mean=np.einsum('i,ij->j',weights,means);delta=means-mean
    covariance=np.einsum('i,ijk->jk',weights,np.asarray(covariances)+delta[:,:,None]*delta[:,None,:])
    return mean,(covariance+covariance.T)/2


class ReliabilityEllipseTracker:
    """Causal delayed forecast scoring, shared IDs and smooth expert selection.

    Independent CV/CA/Singer filters see exactly the same associated detections.
    No interacting-filter state mixing, model probabilities or truth input.
    Scores are physical squared forecast errors against later measurements.
    """
    MODES=('cv','ca','singer')

    def __init__(self,*args,reliability_horizon=1.,score_tau=.5,weight_tau=.3,
                 score_floor=.015,switch_ratio=.8,switch_hold=.6,**kwargs):
        values=(reliability_horizon,score_tau,weight_tau,score_floor,switch_hold)
        if not all(np.isfinite(v) and v>0 for v in values) or not 0<switch_ratio<1:
            raise ValueError('Invalid reliability parameters')
        self.filters={m:DynamicEllipseTracker(*args,mode=m,**kwargs) for m in self.MODES}
        base=self.filters['cv']
        for key in ('dt','horizon','max_distance','timeout','reaction_time','extra_margin','speed_threshold','stage_velocity_guidance'):
            setattr(self,key,getattr(base,key))
        self.mode='adaptive';self.score_horizon=reliability_horizon
        self.score_tau,self.weight_tau,self.score_floor=score_tau,weight_tau,score_floor
        self.switch_ratio,self.switch_hold=switch_ratio,switch_hold
        self.tracks={};self.next_id=1;self.last_stamp=None

    def reset(self):
        for model in self.filters.values():model.reset()
        self.tracks.clear();self.next_id=1;self.last_stamp=None

    def _score(self,track,measurement,stamp):
        elapsed=stamp-track['weight_stamp'];track['weight_stamp']=stamp
        mature=None
        while track['queue'] and track['queue'][0]['stamp']<=stamp-self.score_horizon+1e-9:
            mature=track['queue'].popleft()
        if mature is None or stamp-mature['stamp']>self.score_horizon+max(.15,2*self.dt):return
        errors=[]
        for mode in self.MODES:
            x,p=mature['banks'][mode]
            predicted,_=self.filters[mode].propagate(x,p,stamp-mature['stamp'])
            errors.append(min(float(np.sum((predicted[:2]-measurement)**2)),.25**2))
        errors=np.asarray(errors)
        if track['score_count']==0:track['scores']=errors
        else:
            alpha=-np.expm1(-(stamp-track['score_stamp'])/self.score_tau)
            track['scores']+=alpha*(errors-track['scores'])
        track['score_count']+=1;track['score_stamp']=stamp;track['score_origin']=mature['stamp']
        if track['score_count']<3:return
        scores=track['scores'];best=int(np.argmin(scores));current=track['selected']
        # Prefer the simpler CV expert when measured skill is indistinguishable.
        if scores[0]<=scores[best]+self.score_floor**2:best=0
        improvement=scores[current]-scores[best]
        qualified=(best!=current and ((best==0 and scores[0]<=scores[current]+self.score_floor**2)
            or (scores[best]<self.switch_ratio*scores[current] and improvement>self.score_floor**2)))
        if qualified:
            track['confirm']=track['confirm']+1 if track['challenger']==best else 1
            track['challenger']=best
            if track['confirm']>=3 and stamp-track['switch_stamp']>=self.switch_hold:
                track['selected']=best;track['switch_stamp']=stamp;track['switches']+=1
                track['confirm']=0;track['challenger']=-1
        else:track['confirm']=0;track['challenger']=-1
        target=np.eye(3)[track['selected']]
        alpha=-np.expm1(-max(0.,elapsed)/self.weight_tau)
        track['weights']+=alpha*(target-track['weights'])
        track['weights']/=track['weights'].sum()

    def update(self,ellipses,stamp):
        if not np.isfinite(stamp):raise ValueError('Non-finite scan time')
        if self.last_stamp is not None and stamp<self.last_stamp:self.reset()
        if self.last_stamp is not None and stamp<=self.last_stamp:return ellipses
        self.last_stamp=stamp
        self.tracks={i:t for i,t in self.tracks.items() if stamp-t['seen']<=self.timeout}
        ids=sorted(self.tracks);assigned={}
        if ids and ellipses:
            predicted=[]
            for key in ids:
                points=[]
                for mode in self.MODES:
                    bank=self.filters[mode].tracks[key]
                    x,_=self.filters[mode].propagate(bank['x'],bank['P'],stamp-bank['stamp'])
                    points.append(x[:2])
                predicted.append(self.tracks[key]['weights']@np.asarray(points))
            centers=np.array([[e.cx,e.cy] for e in ellipses])
            distance=np.linalg.norm(centers[:,None,:]-np.asarray(predicted)[None,:,:],axis=2)
            cost=np.full((len(ellipses),len(ids)+len(ellipses)),2*self.max_distance)
            cost[:,:len(ids)]=np.where(distance<=self.max_distance,distance,100*self.max_distance)
            rr,cc=linear_sum_assignment(cost)
            assigned={r:ids[c] for r,c in zip(rr,cc) if c<len(ids) and distance[r,c]<=self.max_distance}
        keys=[]
        for index,ellipse in enumerate(ellipses):
            key=assigned.get(index)
            if key is None:
                key=self.next_id;self.next_id+=1
                self.tracks[key]=dict(seen=stamp,weights=np.array([1.,0.,0.]),scores=np.full(3,np.nan),
                    score_count=0,selected=0,switch_stamp=stamp,weight_stamp=stamp,score_stamp=None,
                    score_origin=None,switches=0,challenger=-1,confirm=0,moving_votes=0,dynamic=False,
                    queue=deque(maxlen=int(np.ceil(self.score_horizon/self.dt))+8))
            self._score(self.tracks[key],np.array([ellipse.cx,ellipse.cy]),stamp)
            keys.append(key);ellipse.track_id=key
        for mode,model in self.filters.items():
            copies=[Ellipse(e.cx,e.cy,e.a,e.b,e.theta,e.point_count) for e in ellipses]
            model.update(copies,stamp,forced_ids=keys)
        for key,track in self.tracks.items():
            banks=[self.filters[m].tracks[key] for m in self.MODES]
            x,p=mixture_moments([b['x'] for b in banks],[b['P'] for b in banks],track['weights'])
            track.update(x=x,P=p,stamp=stamp,missed=banks[0]['missed'])
        for ellipse,key in zip(ellipses,keys):
            track=self.tracks[key];base=self.filters['cv'].tracks[key]
            track['seen']=stamp
            moving=np.linalg.norm(track['x'][5:7])>self.speed_threshold
            track['moving_votes']=max(-3,min(3,track['moving_votes']+(1 if moving else -1)))
            if track['moving_votes']>=2:track['dynamic']=True
            elif track['moving_votes']<=-2:track['dynamic']=False
            ellipse.variance=base['variance']
            track['queue'].append(dict(stamp=stamp,banks={m:(self.filters[m].tracks[key]['x'].copy(),
                self.filters[m].tracks[key]['P'].copy()) for m in self.MODES}))
        return ellipses

    def snapshot(self,now):
        snapshots=[]
        for key,t in sorted(self.tracks.items()):
            if not -1e-9<=now-t['seen']<=self.timeout:continue
            item={k:v for k,v in t.items() if k!='queue'}
            item.update(id=key,x=t['x'].copy(),P=t['P'].copy(),weights=t['weights'].copy(),scores=t['scores'].copy())
            item['banks']={}
            for mode,model in self.filters.items():
                b=model.tracks[key]
                item['banks'][mode]=dict(b,id=key,x=b['x'].copy(),P=b['P'].copy())
            snapshots.append(item)
        return snapshots

    def forecast(self,track,now):
        if not -1e-9<=now-track['seen']<=self.timeout:return None
        experts=[self.filters[m].forecast(track['banks'][m],now) for m in self.MODES]
        w=track['weights'];centers=[];covariances=[]
        for j in range(self.horizon+1):
            mean,p=mixture_moments([e['prediction_centers'][j] for e in experts],
                [e['prediction_position_covariances'][j] for e in experts],w)
            centers.append(mean);covariances.append(p)
        states=[];ps=[]
        for mode in self.MODES:
            bank=track['banks'][mode]
            x,p=self.filters[mode].propagate(bank['x'],bank['P'],max(0.,now-bank['stamp']))
            states.append(x);ps.append(p)
        x,p=mixture_moments(states,ps,w);raw=track['banks']['cv']
        c,s=np.cos(x[4]-raw['raw_theta']),np.sin(x[4]-raw['raw_theta'])
        rotation=np.array([[c,-s],[s,c]]);base=np.maximum(x[2:4],[raw['raw_a'],raw['raw_b']])
        c,s=np.cos(x[4]),np.sin(x[4]);local=(raw['raw_center']-x[:2])@np.array([[c,-s],[s,c]])
        shape=np.diag(1./base)@rotation@np.diag([raw['raw_a'],raw['raw_b']])
        guard=base*max(1.,np.linalg.norm(local/base)+np.linalg.norm(shape,2))
        margins=np.einsum('i,ij->j',w,[e['prediction_execution_margins'] for e in experts]);axes=[]
        for j,position_p in enumerate(covariances):
            geo=p[2:4,2:4]+j*self.filters['cv'].Q[2:4,2:4]
            inflation=.95*np.sqrt(max(np.diag(position_p)))+.95*np.sqrt(max(np.diag(geo)))
            inflated=guard+inflation;axes.append(inflated*(1.+margins[j]/inflated[1]))
        velocity=np.einsum('i,ijk->jk',w,[e['prediction_velocities'] for e in experts])
        return dict(id=track['id'],name='tracked_'+str(track['id']),center=np.asarray(centers[0]),
            a=float(axes[0][0]),b=float(axes[0][1]),theta=float(x[4]),prediction_model='adaptive',
            stage_velocity_guidance=self.stage_velocity_guidance,prediction_centers=np.asarray(centers),
            prediction_axes=np.asarray(axes),prediction_velocities=velocity,
            prediction_position_covariances=np.asarray(covariances),prediction_execution_margins=margins,
            velocity=velocity[0],acceleration=np.einsum('i,ij->j',w,[e['acceleration'] for e in experts]),
            dynamic=track['dynamic'],guidance=not track['dynamic'],execution_margin=float(margins[0]),
            seen=track['seen'],measurement_age=now-track['seen'],missed=track['missed'],
            raw_a=raw['raw_a'],raw_b=raw['raw_b'],point_count=raw['point_count'],
            model_weights=w.copy(),model_scores=[float(s) if np.isfinite(s) else None for s in track['scores']],
            guidance_motion=dict(stamp=now,weights=w.tolist(),models=[e['guidance_motion']['models'][0] for e in experts]),
            selected_model=self.MODES[track['selected']],score_count=track['score_count'],
            score_origin=track['score_origin'],model_switches=track['switches'])

    def obstacles(self,snapshot,now):
        return [o for o in (self.forecast(t,now) for t in snapshot) if o is not None]


def make_dynamic_tracker(*args,mode='cv',**kwargs):
    return ReliabilityEllipseTracker(*args,**kwargs) if mode=='adaptive' else DynamicEllipseTracker(*args,mode=mode,**kwargs)


class TurtleBot3CBFController:
    COLORS = ((0.1, 0.8, 1.0), (1.0, 0.35, 0.2), (0.3, 1.0, 0.3),
              (1.0, 0.3, 0.85), (0.8, 0.7, 0.2), (0.6, 0.4, 1.0))
    FIELDS = ["time","ros_time","robot","status","x","y","theta","v","omega",
        "observer_x","observer_y","goal_x","goal_y","consensus_error_x","consensus_error_y",
        "observer_step","leader_step","observer_age","odom_age","geometry_age",
        "detected_ellipses","static_constraints","peer_constraints","min_h",
        "solve_ms","solver_success","solver_attempts","guide_level","guide_obstacle","recovery_barriers",
        "laser_peer_constraints","odometry_peer_constraints","geometry_ms","peer_fit_ms",
        "reference_x","reference_y","heading_error","raw_min_range",
        "closest_obstacle_distance","detected_peer_ellipses","total_clusters_count",
        "cycle_id","input_ms","manager_ms","command_wall_ms","record_ms",
        "ipopt_status","ipopt_iterations","ipopt_wall_ms","ipopt_cpu_ms","fallback_ms",
        "validated_steps","dual_warm","budget_hit","solver_seed","geometry_sequence",
        "geometry_sample_stamp","geometry_cpu_ms","geometry_peer_cpu_ms",
        "dynamic_tracks","missed_tracks","max_obstacle_speed","motion_prediction_enabled",
        "ca_prediction_enabled","singer_prediction_enabled","adaptive_prediction_enabled","adaptive_switches_active","max_obstacle_acceleration",
        "progress_enabled","progress_mode","progress_decision_id","progress_decision_ms",
        "detected_static_ellipses","detected_dynamic_ellipses",
        "metrics_schema_version","evaluation_time","evaluation_ros_time","reference_quality",
        "leader_reference_x","leader_reference_y","leader_reference_heading",
        "nominal_reference_x","nominal_reference_y","plan_reference_x","plan_reference_y",
        "plan_execution_error_x","plan_execution_error_y","plan_reference_valid","plan_reference_age","plan_source_cycle",
        "observer_aligned_x","observer_aligned_y","observer_error_x","observer_error_y",
        "legacy_consensus_error_x","legacy_consensus_error_y","legacy_heading_error"]

    def __init__(self):
        self.robot_id=int(rospy.get_param("~robot_id"))
        self.names=rospy.get_param("/cbf/robot_names",["tb3_"+str(i) for i in range(5)])
        self.name=self.names[self.robot_id]
        self.cfg=default_config();self.cfg.update(rospy.get_param("/cbf/mpc",{}))
        self.cfg['progress_guidance']=bool(rospy.get_param('/cbf/progress_guidance',False))
        self.cfg['moving_frame_projection']=bool(rospy.get_param('/cbf/moving_frame_projection',False))
        self.cfg['feasible_seed_search']=bool(rospy.get_param('/cbf/feasible_seed_search',False))
        self.cfg['realtime']=bool(rospy.get_param('/cbf/realtime_solver',True))
        self.control_budget=float(rospy.get_param('/cbf/control_budget_ms',100.))/1000.
        self.cfg['nlp_wall_time']=float(rospy.get_param('/cbf/nlp_budget_ms',65.))/1000.
        self.cfg['manager_wall_time']=max(.010,self.control_budget-.015)
        self.cfg['fallback_reserve']=.010
        self.cfg['dual_warm_start']=bool(rospy.get_param('/cbf/dual_warm_start',True))
        self.cfg['stagger_callbacks']=bool(rospy.get_param('/cbf/stagger_callbacks',True))
        if not .005 < self.cfg['nlp_wall_time'] < self.control_budget <= .2:
            raise ValueError('Expected 5ms < NLP budget < control budget <= 200ms')
        self.dt=self.cfg["dt"];self.n=self.cfg["horizon"]
        self.slots=int(rospy.get_param("/cbf/obstacle_slots",12))
        self.max_static=self.slots-(len(self.names)-1)
        self.peer_safety=bool(rospy.get_param("/cbf/peer_safety",True))
        self.stale=float(rospy.get_param("/cbf/data_timeout",.6))
        self.radius=float(rospy.get_param("/cbf/formation_radius",.5))
        self.peer_radius=float(rospy.get_param("/cbf/peer_radius",.20))
        if not np.isfinite(self.peer_radius) or self.peer_radius <= 0:
            raise ValueError('Expected a positive finite peer envelope radius')
        self.recovery_gamma=float(rospy.get_param('/cbf/recovery_gamma',.03))
        if not 0 < self.recovery_gamma <= self.cfg['gamma'] <= 1:
            raise ValueError('Expected 0 < recovery_gamma <= gamma <= 1')
        self.slot_names=[None]*self.max_static
        self.last_slot_signature=None
        self.last_command_time=None
        self.last_command_wall=None;self.applied_control=np.zeros(2)
        self.last_solution_stamp=None;self.cycle_id=0;self.trace_lock=threading.RLock()
        self.watchdog_stops=0
        self.lock=threading.RLock();self.closed=False;self.running=False;self.warmed=False
        self.poses={};self.pose_history={};self.neighbors={};self.pending={};self.last_step=-1
        self.z=np.zeros(2);self.dz=np.zeros(2)
        self.metric_leader=LeaderMetricTimeline(self.dt)
        self.metric_plan=None;self.metric_context={}
        self.A=np.asarray(rospy.get_param("/cbf/adjacency",[[0,0,0,0,0],[1,0,0,0,0],
            [0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]),float)
        self.B=np.asarray(rospy.get_param("/cbf/pinning",[1,0,0,0,0]),float)
        self.eta=float(rospy.get_param("/cbf/observer_eta",.7));self.alpha=.8
        self.neighbor_ids=list(np.flatnonzero(self.A[self.robot_id]))
        self.reference=TimedReference(self.robot_id,self.dt,self.n,self.radius,robot_count=len(self.names))
        self.envelopes=ObservedEnvelopes(self.dt,self.n)
        self.observed=[];self.geometry_stamp=None;self.geometry_sequence=None
        self.latest_leader=None;self.clock_sources={};self.last_solution=None
        self.manager=MPCSolverManager(self.cfg,recovery=bool(rospy.get_param('/cbf/stall_recovery',True)))
        self.problem_key=(self.slots,bool(self.cfg['realtime'] and self.cfg['delta_limit']))
        self.manager.problems[self.problem_key]=CachedMPCProblem(self.cfg,*self.problem_key)
        self.manager.builds=1
        self.cmd_pub=rospy.Publisher('/'+self.name+'/cmd_vel',Twist,queue_size=1)
        self.observer_pub=rospy.Publisher('/'+self.name+'/observer_state',Float64MultiArray,
                                         queue_size=10,latch=True)
        self.history_pub=rospy.Publisher('/'+self.name+'/history',RobotHistoryData,queue_size=10)
        self.diagnostics_pub=rospy.Publisher('/'+self.name+'/cbf_diagnostics',String,queue_size=10)
        for i,name in enumerate(self.names):
            rospy.Subscriber('/'+name+'/odom',Odometry,self.odom_callback,callback_args=i,queue_size=5)
            if i!=self.robot_id:
                rospy.Subscriber('/'+name+'/observer_state',Float64MultiArray,
                    self.observer_callback,callback_args=i,queue_size=100)
        self.init_geometry()
        leader_ns=rospy.get_param('/cbf/leader_namespace','tb3_virtual_leader')
        rospy.Subscriber('/'+leader_ns+'/state',Float64MultiArray,self.leader_callback,queue_size=100)
        rospy.Subscriber('/leader/history',LeaderHistoryData,self.clock_callback,queue_size=100)
        rospy.Subscriber('/system/shutdown',Bool,self.stop_callback,queue_size=1)
        run_id=rospy.get_param('/cbf/run_id','')
        if run_id:
            key='/cbf/results/'+run_id+'/directory'
            deadline=time.monotonic()+20.
            while not rospy.has_param(key) and not rospy.is_shutdown():
                if time.monotonic()>deadline:raise RuntimeError('CBF result recorder did not initialize')
                time.sleep(.05)
            self.output=Path(rospy.get_param(key))
        else:
            self.output=Path(rospy.get_param('/cbf/output_dir','/tmp/tb3_cbf_gazebo'))
        self.output.mkdir(parents=True,exist_ok=True)
        self.stream=(self.output/(self.name+'.csv')).open('w',newline='')
        self.writer=csv.DictWriter(self.stream,fieldnames=self.FIELDS);self.writer.writeheader()
        self.trace_stream=(self.output/(self.name+'_solver_trace.jsonl')).open('w')
        self.counts=Counter();self.rows=0
        rospy.loginfo('%s: experiment CSV %s',self.name,self.output/(self.name+'.csv'))
        self.publish_observer(-1)
        rospy.set_param('/cbf/ready/'+self.name,False)
        rospy.on_shutdown(self.close)
        self.observer_timer=rospy.Timer(rospy.Duration(.02),self.advance_observer,reset=True)
        if self.cfg['realtime'] and self.cfg['stagger_callbacks']:
            phase=self.robot_id*self.dt/len(self.names)
            self.control_start_timer=rospy.Timer(rospy.Duration(.001+phase),self.start_control_timer,oneshot=True,reset=True)
            self.geometry_start_timer=rospy.Timer(rospy.Duration(.001+phase+.5/self.geometry_rate),
                                                 self.start_geometry_timer,oneshot=True,reset=True)
        else:
            self.timer=rospy.Timer(rospy.Duration(self.dt),self.control,reset=True)
            self.geometry_timer=rospy.Timer(rospy.Duration(1.0/self.geometry_rate),self.process_scan,reset=True)
        self.watchdog_timer=rospy.Timer(rospy.Duration(.05),self.watchdog,reset=True)
        rospy.loginfo('%s: unified MPC, N=%d, %d slots; CAPA absent',self.name,self.n,self.slots)

    def start_control_timer(self,_event):
        if not self.closed:self.timer=rospy.Timer(rospy.Duration(self.dt),self.control,reset=True)

    def start_geometry_timer(self,_event):
        if not self.closed:self.geometry_timer=rospy.Timer(rospy.Duration(1./self.geometry_rate),self.process_scan,reset=True)

    def init_geometry(self):
        """Direct LaserScan input; all fitting and rendering live in this file."""
        get = rospy.get_param
        self.target_frame = 'world'
        self.frame_prefix = self.name
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
        # Robot shape changes slowly in its body frame. Bound the additional
        # MVEE work; every returned fit still encloses all assigned returns.
        self.peer_fit_iterations = int(get('~peer_fit_max_iterations', 80))
        self.peer_fit_tolerance = float(get('~peer_fit_tolerance', 0.02))
        self.peer_fit_period = 1./float(get('~peer_fit_rate', 5.0))
        self.peer_display_period = 1./float(get('~peer_display_rate', 5.0))
        self.display_margin = float(get('~display_margin', 0.10))
        self.scan_timeout = float(get('~scan_timeout', 0.5))
        self.tf_timeout = float(get('~tf_timeout', 0.05))
        self.geometry_rate = float(get('~processing_rate', 10.0))
        self.robot_exclusion_radius = float(get('~robot_exclusion_radius', 0.16))
        self.static_accumulation = bool(get('~static_accumulation', True))
        self.dynamic_prediction = bool(get('/cbf/dynamic_prediction',False))
        if self.dynamic_prediction:
            # External targets are associated frame by frame; never accumulate
            # their old locations into the static world representation.
            self.static_accumulation=False
        dynamic_mode=get('/cbf/dynamic/prediction_mode','cv')
        self.dynamic_tracker=make_dynamic_tracker(self.dt,self.n,
            float(get('/cbf/dynamic/association_distance',.65)),
            float(get('/cbf/dynamic/track_timeout',.65)),
            speed_threshold=float(get('/cbf/dynamic/speed_threshold',.05)),
            mode=dynamic_mode,
            reaction_time=float(get('/cbf/dynamic/reaction_time',.35)),
            extra_margin=float(get('/cbf/dynamic/extra_margin',.02)),
            ca_jerk_density=float(get('/cbf/dynamic/ca_jerk_density',.01)),
            ca_acceleration_variance=float(get('/cbf/dynamic/ca_acceleration_variance',.04)),
            singer_tau=float(get('/cbf/dynamic/singer_tau',1.)),
            acceleration_buffer=bool(get('/cbf/dynamic/acceleration_buffer',True)),
            stage_velocity_guidance=bool(get('/cbf/dynamic/stage_velocity_guidance',True)),
            **(get('/cbf/dynamic/reliability',{}) if dynamic_mode=='adaptive' else {}))
        self.dynamic_snapshot=[]
        self.static_map_voxel = float(get('~static_map_voxel', 0.04))
        if (self.max_range <= self.min_range or self.min_range < 0
                or self.voxel_size < 0 or self.eps <= 0 or self.min_samples < 1
                or self.max_extent <= 0 or self.min_axis <= 0 or self.padding < 0
                or self.display_margin < 0 or self.fit_tolerance <= 0
                or self.fit_iterations < 1 or self.scan_timeout <= 0
                or self.tf_timeout < 0 or not 0 < self.geometry_rate <= 100
                or self.robot_exclusion_radius <= 0 or self.static_map_voxel <= 0):
            raise ValueError('Invalid in-controller laser geometry configuration')
        if (self.peer_fit_iterations < 1 or self.peer_fit_tolerance <= 0
                or not .01 <= self.peer_fit_period <= self.scan_timeout
                or not .01 <= self.peer_display_period <= self.scan_timeout):
            raise ValueError('Peer fit/display rate must be positive and faster than scan timeout')
        self.tracker = EllipseTracker(float(get('~association_distance', 0.6)),
                                      float(get('~track_timeout', 1.0)),
                                      int(get('~shape_history_size', 10)))
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        geometry_ns = '/' + self.name + '/cbf_geometry'
        self.cloud_pub = rospy.Publisher(geometry_ns+'/points', PointCloud2, queue_size=1)
        self.cluster_pub = rospy.Publisher(geometry_ns+'/clusters', MarkerArray, queue_size=1)
        self.ellipse_pub = rospy.Publisher(geometry_ns+'/ellipses', MarkerArray, queue_size=1)
        self.peer_pub = rospy.Publisher(geometry_ns+'/peers', MarkerArray, queue_size=1)
        self.estimates_pub = rospy.Publisher(geometry_ns+'/estimates',String,queue_size=5)
        self.predictions_pub = rospy.Publisher(geometry_ns+'/predictions',String,queue_size=2)
        self.geometry_debug = bool(get('/cbf/visualization/debug',False))
        selected = int(get('/cbf/visualization/observer',0)) == self.robot_id
        self.display_ellipse_pub = (rospy.Publisher('/cbf/geometry/ellipses',MarkerArray,queue_size=1)
                                    if selected else None)
        self.display_peer_pub = (rospy.Publisher('/cbf/geometry/peers',MarkerArray,queue_size=1)
                                 if selected else None)
        self.raw_min_range = float('nan')
        self.obstacle_points = np.empty((0,2))
        self.obstacle_points_stamp = None
        self.peer_shapes = {}
        self.peer_shape_history = {}
        self.peer_laser_fits = Counter()
        self.peer_visible_markers = set()
        self.peer_marker_stamp = None
        self.geometry_ms = 0.
        self.peer_fit_ms = 0.
        self.geometry_timing={};self.geometry_timing_sequence=0
        self.latest_scan = None
        self.processed_scan = None
        self.last_success = None
        self.visible_ids = set()
        self.static_points = np.empty((0, 2))
        self.static_map_stamp = None
        self.scan_subscriber = rospy.Subscriber(get('~scan_topic', '/'+self.name+'/scan'),
                                               LaserScan, self.scan_callback, queue_size=1)
        rospy.loginfo('%s: embedded LaserScan geometry %s -> world',
                      self.name, self.scan_subscriber.resolved_name)

    def scan_callback(self, scan):
        with self.lock:
            self.latest_scan = scan

    def process_scan(self, _event):
        if self.closed:
            return
        geometry_begin = time.perf_counter()
        geometry_cpu_begin=time.thread_time()
        with self.lock:
            scan = self.latest_scan
        if scan is None:
            return
        now = rospy.Time.now()
        stamp = scan.header.stamp
        if stamp == rospy.Time(0):
            rospy.logwarn_throttle(5.0, 'Dropping zero-stamped scan: timestamped TF is required')
            self.clear_stale_geometry(now)
            return
        if now < stamp:
            return
        if (now - stamp).to_sec() > self.scan_timeout:
            self.clear_stale_geometry(now)
            return
        if scan is self.processed_scan:
            return
        if self.geometry_stamp is not None and abs(stamp.to_sec()-self.geometry_stamp)<1e-9:
            return  # A replayed stamp must not update the KF or create ID=-1 fits.
        source_frame = self.frame_override or scan.header.frame_id.lstrip('/')
        if not source_frame:
            rospy.logwarn_throttle(5.0, 'LaserScan has no frame_id')
            self.clear_stale_geometry(now)
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
            self.raw_min_range = float(np.linalg.norm(points,axis=1).min()) if len(points) else float('inf')
            world_points = transform_points(points, [translation.x, translation.y, translation.z],
                                           [rotation.x, rotation.y, rotation.z, rotation.w])
            xy = voxel_downsample(world_points[:, :2], self.voxel_size)
            with self.lock:
                histories = {i:list(history) for i,history in self.pose_history.items()}
            robot_poses = {i:odometry_at_stamp(history, stamp.to_sec(), self.scan_timeout)
                           for i,history in histories.items()}
            robot_poses = {i:item for i,item in robot_poses.items() if item is not None}
            if self.static_accumulation and len(robot_poses) < len(self.names):
                # Do not commit a robot body to the static map before peer masks
                # are available. This mode is explicitly for static-world tests.
                return
            xy, peer_points = split_robot_returns(xy, robot_poses, self.robot_id,
                                                   self.robot_exclusion_radius)
            # Diagnostics use this frame's measured points, including peers,
            # before static accumulation. This is separate from CBF h values.
            live_obstacle_points = np.vstack([xy]+list(peer_points.values()))
            peer_begin = time.perf_counter()
            peer_cpu_begin=time.thread_time()
            self.update_peer_shapes(peer_points, robot_poses, stamp.to_sec())
            self.peer_fit_ms = 1000.*(time.perf_counter()-peer_begin)
            peer_wall_ms=self.peer_fit_ms
            peer_cpu_ms=1000*(time.thread_time()-peer_cpu_begin)
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
            moving=getattr(self,'dynamic_prediction',False)
            if moving:self.dynamic_tracker.update(fits,stamp.to_sec())
            else:self.tracker.update(fits, stamp.to_sec())
            raw = [dict(id=e.track_id, center=np.array([e.cx, e.cy]), theta=e.theta,
                        a=e.a, b=e.b, variance=e.variance) for e in fits]
            with self.lock:
                if self.closed:
                    return
                if self.geometry_stamp is not None and stamp.to_sec() < self.geometry_stamp:
                    self.envelopes.tracks.clear()
                if moving:
                    self.dynamic_snapshot=self.dynamic_tracker.snapshot(stamp.to_sec())
                    self.observed=self.dynamic_tracker.obstacles(self.dynamic_snapshot,stamp.to_sec())
                else:self.observed = self.envelopes.update(raw, stamp.to_sec())
                self.geometry_stamp = stamp.to_sec()
                self.obstacle_points = live_obstacle_points
                self.obstacle_points_stamp = stamp.to_sec()
                observed = list(self.observed)
            self.publish_geometry_estimates(stamp.to_sec(),fits,observed)
            if moving:
                rows=[dict(id=o['id'],dynamic=o['dynamic'],velocity=o['velocity'].tolist(),
                    prediction_model=o['prediction_model'],acceleration=o['acceleration'].tolist(),
                    velocities=o['prediction_velocities'].tolist(),
                    position_covariances=o['prediction_position_covariances'].tolist(),
                    execution_margins=o['prediction_execution_margins'].tolist(),
                    measurement_age=o['measurement_age'],missed=o['missed'],
                    execution_margin=o['execution_margin'],
                    centers=o['prediction_centers'].tolist(),axes=o['prediction_axes'].tolist(),
                    theta=o['theta']) for o in observed]
                if self.dynamic_tracker.mode=='adaptive':
                    for row,o in zip(rows,observed):row.update(model_weights=o['model_weights'].tolist(),
                        model_scores=o['model_scores'],selected_model=o['selected_model'],
                        score_count=o['score_count'],score_origin=o['score_origin'],model_switches=o['model_switches'])
                self.predictions_pub.publish(String(data=json.dumps(dict(stamp=stamp.to_sec(),
                    observer=self.name,dt=self.dt,mode=self.dynamic_tracker.mode,tracks=rows,
                    measurements=[dict(center=[e.cx,e.cy],a=e.a,b=e.b,theta=e.theta,
                        point_count=e.point_count) for e in fits]))))
            self.publish_geometry(stamp, world_points, fits, clusters)
            self.processed_scan = scan
            self.last_success = now
            self.geometry_ms = 1000.*(time.perf_counter()-geometry_begin)
            self.geometry_timing_sequence+=1
            self.geometry_timing=dict(sequence=self.geometry_timing_sequence,stamp=stamp.to_sec(),
                wall_ms=self.geometry_ms,peer_wall_ms=peer_wall_ms,
                cpu_ms=1000*(time.thread_time()-geometry_cpu_begin),peer_cpu_ms=peer_cpu_ms)
            rospy.loginfo_throttle(2.0, 'Laser geometry: %d returns, %d obstacle clusters',
                                  len(points), len(fits))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(3.0, 'Waiting for scan-time TF %s <- %s: %s',
                                  self.target_frame, source_frame, exc)
            self.clear_stale_geometry(now)
        except (ValueError, np.linalg.LinAlgError) as exc:
            rospy.logwarn_throttle(3.0, 'Invalid scan or geometry: %s', exc)
            self.clear_stale_geometry(now)

    def update_peer_shapes(self, groups, robot_poses, stamp):
        """Per-frame robot fits are cached briefly and NEVER enter the static map."""
        updates = {}
        for peer_id, points in groups.items():
            if len(points) < self.min_samples:
                continue
            previous = self.peer_shape_history.get(peer_id)
            if previous is not None and 0 <= stamp-previous[0] < self.peer_fit_period-1e-6:
                continue
            shape = fit_peer_shape(points, robot_poses[peer_id][0], self.peer_radius,
                self.min_axis, self.padding, self.peer_fit_tolerance, self.peer_fit_iterations)
            history = (previous[1] if previous is not None and stamp > previous[0]
                       else deque(maxlen=self.tracker.history_size))
            history.append([shape['raw_a'],shape['raw_b']])
            shape.update(stamp=stamp, variance=float(np.var(np.asarray(history),axis=0).sum())**1.3
                         +max(0,3-len(history))*.0005)
            self.peer_shape_history[peer_id] = (stamp,history)
            updates[peer_id] = shape
            self.peer_laser_fits[peer_id] += 1
        with self.lock:
            self.peer_shapes = {i:shape for i,shape in self.peer_shapes.items()
                                if 0 <= stamp-shape['stamp'] <= self.scan_timeout}
            self.peer_shapes.update(updates)

    def publish_geometry_estimates(self, stamp, fits, observed):
        """Record measured and effective semi-axes separately, independent of RViz."""
        rows=[];by_id={o['id']:o for o in observed}
        for fit in fits:
            effective=by_id[fit.track_id]
            rows.append(dict(ros_time=stamp,observer=self.name,kind='dynamic' if effective.get('dynamic',False) else 'static',id=fit.track_id,
                source='laser',raw_a=fit.a,raw_b=fit.b,effective_a=effective['a'],
                effective_b=effective['b'],point_count=fit.point_count,measurement_age=0.,size_floor_dominates=False))
        with self.lock:shapes=dict(self.peer_shapes)
        for i in range(len(self.names)):
            if i==self.robot_id:continue
            shape=shapes.get(i)
            valid=shape is not None and 0 <= stamp-shape['stamp'] <= self.scan_timeout
            a,b=(shape['a'],shape['b']) if valid else (self.peer_radius,self.peer_radius)
            rows.append(dict(ros_time=stamp,observer=self.name,kind='peer',id=i,
                source='laser' if valid else 'odometry',raw_a=shape['raw_a'] if valid else None,
                raw_b=shape['raw_b'] if valid else None,effective_a=a,effective_b=b,
                point_count=shape['point_count'] if valid else 0,
                measurement_age=stamp-shape['stamp'] if valid else None,
                size_floor_dominates=bool(abs(a-self.peer_radius)<1e-9 and abs(b-self.peer_radius)<1e-9)))
        self.estimates_pub.publish(String(data=json.dumps(dict(rows=rows))))

    def publish_peer_geometry(self, stamp, peers):
        """Render the very same robot envelopes supplied to this MPC cycle."""
        if self.closed:
            return
        if not (self.peer_pub.get_num_connections() or
                self.display_peer_pub is not None and self.display_peer_pub.get_num_connections()):
            return
        if (self.peer_marker_stamp is not None
                and 0 <= stamp.to_sec()-self.peer_marker_stamp < self.peer_display_period-1e-6):
            return
        self.peer_marker_stamp = stamp.to_sec()
        output = MarkerArray()
        visible = set()
        for peer_id, obs in peers:
            sensed = obs['peer_source'] == 'laser'
            color = (0.15,1.,0.45) if sensed else (.65,.65,.65)
            safe_scale = 1.+self.cfg['margin']/obs['b']
            contours = [('peer_body', obs['center'], obs['a'], obs['b'], obs['theta'], color, 1.)]
            if self.geometry_debug:
                contours += [('peer_safe', obs['center'], obs['a']*safe_scale, obs['b']*safe_scale,
                 obs['theta'], color, .35),
                ('peer_future', obs['prediction_centers'][-1], obs['a']*safe_scale,
                 obs['b']*safe_scale, obs.get('prediction_angles',[obs['theta']])[-1],color,.25)]
            if sensed:
                contours.append(('peer_raw',obs['measured_center'],obs['measured_a'],
                    obs['measured_b'],obs['theta'],(1.,.7,.1),1.))
            for namespace,center,a,b,theta,rgb,alpha in contours:
                marker = self.geometry_marker(stamp,peer_id,namespace,Marker.LINE_STRIP)
                marker.scale.x = .018 if namespace == 'peer_raw' else .012
                marker.color.r,marker.color.g,marker.color.b = rgb
                marker.color.a = alpha
                ellipse = Ellipse(float(center[0]),float(center[1]),a,b,theta,0)
                marker.points = [Point(x=float(p[0]),y=float(p[1]),z=.12)
                                 for p in ellipse.contour(count=40)]
                output.markers.append(marker);visible.add((namespace,peer_id))
            if not self.geometry_debug:continue
            marker = self.geometry_marker(stamp,peer_id,'peer_path',Marker.LINE_STRIP)
            marker.scale.x = .01
            marker.color.r,marker.color.g,marker.color.b = color
            marker.color.a = .6
            marker.points = [Point(x=float(p[0]),y=float(p[1]),z=.12)
                             for p in obs['prediction_centers']]
            output.markers.append(marker);visible.add(('peer_path',peer_id))
            marker = self.geometry_marker(stamp,peer_id,'peer_label',Marker.TEXT_VIEW_FACING)
            marker.pose.position.x,marker.pose.position.y = obs['center']
            marker.pose.position.z = .50
            marker.scale.z = .11
            marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.
            source = ('laser %d pts' % obs['measured_point_count']) if sensed else 'odom fallback'
            raw=(' raw=%.4f/%.4f'%(obs['measured_a'],obs['measured_b'])) if sensed else ''
            marker.text = '%s %s%s used=%.4f/%.4f' % (self.names[peer_id],source,raw,obs['a'],obs['b'])
            output.markers.append(marker);visible.add(('peer_label',peer_id))
        for namespace,peer_id in self.peer_visible_markers-visible:
            marker = self.geometry_marker(stamp,peer_id,namespace,Marker.LINE_STRIP)
            marker.action = Marker.DELETE;output.markers.append(marker)
        self.peer_visible_markers = visible
        if self.peer_pub.get_num_connections():self.peer_pub.publish(output)
        if self.display_peer_pub is not None:self.display_peer_pub.publish(output)

    def clear_stale_geometry(self, now):
        if self.last_success is not None and ((now - self.last_success).to_sec() > self.scan_timeout
                                             or now < self.last_success):
            self.tracker.update([], now.to_sec())
            with self.lock:
                self.observed = []
                self.peer_shapes = {}
                # A stale scan must never become a fresh empty observation.
                self.geometry_stamp = None
            self.publish_geometry(now, np.empty((0, 3)), [], [])
            self.last_success = None

    def geometry_marker(self, stamp, track_id, namespace, marker_type):
        marker = Marker()
        marker.header = Header(stamp=stamp, frame_id=self.target_frame)
        marker.ns, marker.id, marker.type = namespace, track_id, marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = rospy.Duration(max(0.6, self.scan_timeout))
        return marker

    def publish_geometry(self, stamp, points, ellipses, clusters):
        header = Header(stamp=stamp, frame_id=self.target_frame)
        if self.cloud_pub.get_num_connections():
            self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(header, points.tolist()))
        if not (self.cluster_pub.get_num_connections() or self.ellipse_pub.get_num_connections() or
                self.display_ellipse_pub is not None and self.display_ellipse_pub.get_num_connections()):
            return
        cluster_markers, ellipse_markers = MarkerArray(), MarkerArray()
        with self.lock:by_id={o['id']:o for o in self.observed}
        for ellipse, cluster in zip(ellipses, clusters):
            effective=by_id.get(ellipse.track_id)
            color = (1.,.45,.1) if effective is not None and effective.get('dynamic',False) else (.15,.65,1.)
            marker = self.geometry_marker(stamp, ellipse.track_id, 'clusters', Marker.POINTS)
            marker.scale.x = marker.scale.y = 0.045
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 1.0
            marker.points = [Point(x=float(p[0]), y=float(p[1]), z=0.07) for p in cluster]
            cluster_markers.markers.append(marker)
            layers=[('ellipse',ellipse.a,ellipse.b,1.)]
            if self.geometry_debug and effective is not None:
                scale=1.+self.cfg['margin']/effective['b']
                layers.append(('display_margin',effective['a']*scale,effective['b']*scale,.35))
            for namespace, a, b, alpha in layers:
                marker = self.geometry_marker(stamp, ellipse.track_id, namespace, Marker.LINE_STRIP)
                marker.scale.x = 0.025
                marker.color.r, marker.color.g, marker.color.b = color
                marker.color.a = alpha
                contour=Ellipse(ellipse.cx,ellipse.cy,a,b,ellipse.theta,ellipse.point_count)
                marker.points = [Point(x=float(p[0]), y=float(p[1]), z=0.09)
                                 for p in contour.contour()]
                ellipse_markers.markers.append(marker)
            if not self.geometry_debug:continue
            marker = self.geometry_marker(stamp, ellipse.track_id, 'labels', Marker.TEXT_VIEW_FACING)
            marker.pose.position.x, marker.pose.position.y = ellipse.cx, ellipse.cy
            marker.pose.position.z = 0.35
            marker.scale.z = 0.14
            marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.0
            marker.text = f'#{ellipse.track_id} raw a={ellipse.a:.4f} b={ellipse.b:.4f}'
            ellipse_markers.markers.append(marker)
        current_ids = {e.track_id for e in ellipses}
        for track_id in self.visible_ids - current_ids:
            for namespace, output in [('clusters', cluster_markers), ('ellipse', ellipse_markers),
                                       ('display_margin', ellipse_markers), ('labels', ellipse_markers)]:
                marker = self.geometry_marker(stamp, track_id, namespace, Marker.LINE_STRIP)
                marker.action = Marker.DELETE
                output.markers.append(marker)
        self.visible_ids = current_ids
        for obs in by_id.values():
            if not obs.get('dynamic',False):continue
            marker=self.geometry_marker(stamp,obs['id'],'dynamic_prediction',Marker.LINE_STRIP)
            marker.scale.x=.025;marker.color.r=1.;marker.color.g=.45;marker.color.a=.9
            marker.points=[Point(x=float(p[0]),y=float(p[1]),z=.12) for p in obs['prediction_centers']]
            ellipse_markers.markers.append(marker)
            if self.geometry_debug:
                for j in (0,self.n):
                    a,b=obs['prediction_axes'][j];center=obs['prediction_centers'][j]
                    marker=self.geometry_marker(stamp,obs['id'],'dynamic_envelope_'+str(j),Marker.LINE_STRIP)
                    marker.scale.x=.015;marker.color.r=1.;marker.color.g=.45;marker.color.a=.5
                    marker.points=[Point(x=float(p[0]),y=float(p[1]),z=.12) for p in
                        Ellipse(center[0],center[1],a,b,obs['theta'],0).contour()]
                    ellipse_markers.markers.append(marker)
        self.cluster_pub.publish(cluster_markers)
        self.ellipse_pub.publish(ellipse_markers)
        if self.display_ellipse_pub is not None:self.display_ellipse_pub.publish(ellipse_markers)

    def odom_callback(self,msg,i):
        q=msg.pose.pose.orientation;p=msg.pose.pose.position
        state=np.array([p.x,p.y,euler_from_quaternion([q.x,q.y,q.z,q.w])[2]])
        stamp=msg.header.stamp.to_sec()
        if not np.isfinite(state).all() or stamp<=0:return
        with self.lock:
            velocity=np.array([msg.twist.twist.linear.x,msg.twist.twist.angular.z])
            if not np.isfinite(velocity).all():return
            self.poses[i]=(stamp,state,velocity)
            history=self.pose_history.setdefault(i,deque(maxlen=100))
            if history and stamp < history[-1][0]:history.clear()
            if not history or stamp > history[-1][0]:history.append((stamp,state,velocity))

    def clock_callback(self,msg):
        with self.lock:
            self.metric_leader.observe_clock(msg.trajectory_time,msg.ros_timestamp)
            self.clock_sources[int(round(msg.trajectory_time/self.dt))]=msg.ros_timestamp
            if len(self.clock_sources)>100:
                del self.clock_sources[min(self.clock_sources)]

    def leader_callback(self,msg):
        values=np.asarray(msg.data,float)
        if len(values)<6 or not np.isfinite(values[:6]).all():return
        step=int(round(values[0]/self.dt));now=rospy.get_time()
        with self.lock:
            self.metric_leader.observe_state(values[0],values[1:3],values[3:5],values[5])
            self.pending[step]=(values[:6].copy(),now)
            self.latest_leader=(step,values[:6].copy(),now)
            self.running=True

    def observer_callback(self,msg,i):
        if len(msg.data)<6:return
        values=np.asarray(msg.data[:4],float);step=int(round(msg.data[5]))
        if not np.isfinite(values).all():return
        with self.lock:
            history=self.neighbors.setdefault(i,{})
            history[step]=values
            if len(history)>100:del history[min(history)]

    def publish_observer(self,step):
        self.observer_pub.publish(Float64MultiArray(data=[*self.z,*self.dz,rospy.get_time(),float(step)]))

    def advance_observer(self,_event):
        if self.closed:return
        with self.lock:
            if not self.running:
                self.publish_observer(-1);return
            for unused in range(10):
                step=self.last_step+1
                if step not in self.pending:return
                if any(step-1 not in self.neighbors.get(i,{}) for i in self.neighbor_ids):return
                leader,received=self.pending.pop(step)
                sum_velocity=np.zeros(2);sum_error=np.zeros(2)
                for i in self.neighbor_ids:
                    item=self.neighbors[i][step-1];weight=self.A[self.robot_id,i]
                    sum_velocity+=weight*item[2:4];sum_error+=weight*(self.z-item[:2])
                pin=self.B[self.robot_id];degree=self.A[self.robot_id].sum()+pin
                ss=sum_error+pin*(self.z-leader[1:3])
                self.dz=(sum_velocity+pin*leader[3:5]-self.eta*np.sign(ss)*np.abs(ss)**self.alpha)/degree
                self.z=self.z+self.dt*self.dz
                self.last_step=step;self.observer_received=received
                self.observer_source=self.clock_sources.get(step,received)
                self.observer_logical=leader[0]
                self.reference.observe(leader[0],self.z,self.dz)
                self.publish_observer(step)

    def stop_callback(self,msg):
        if msg.data:self.close()

    def stop_motion(self):
        self.cmd_pub.publish(Twist())
        self.applied_control=np.zeros(2)
        self.metric_plan=None

    def capture_metric_context(self,stamp,state):
        """Freeze evaluation inputs before solving, under the control lock."""
        reference=self.metric_leader.evaluate(stamp)
        aligned=np.full(2,np.nan)
        if self.reference.history:
            source=self.clock_sources.get(self.last_step,self.observer_source)
            if abs(stamp-source)<=self.stale:
                aligned=self.reference.goals(stamp-source,state[2])[0,:2]-self.reference.offset()
        return dict(stamp=stamp,leader=reference,nominal=reference['position']+self.reference.offset(),
            observer_raw=self.z.copy(),observer_aligned=aligned,observer_step=self.last_step,
            plan=sample_plan_reference(self.metric_plan,stamp))

    def watchdog(self,_event):
        # Independent of the possibly busy NLP timer. Never hold an old moving
        # command indefinitely after a slow solve or a callback failure.
        if (not self.closed and self.running and self.last_command_time is not None
                and (rospy.get_time()-self.last_command_time>.25 or
                     self.last_command_wall is not None and time.perf_counter()-self.last_command_wall>.25)):
            self.stop_motion()
            self.watchdog_stops+=1

    def control(self,_event):
        if self.closed:return
        begin=time.perf_counter();now=rospy.get_time();self.cycle_id+=1
        self.cycle_cpu_begin=time.thread_time();self.input_ms=0.;self.manager_ms=0.;self.command_wall_ms=0.
        self.cycle_validated_steps=0
        with self.lock:
            self.metric_context={}
            poses={k:(s,x.copy(),v.copy()) for k,(s,x,v) in self.poses.items()}
            observed=list(self.observed);stamp=self.geometry_stamp
            if self.dynamic_prediction:
                observed=self.dynamic_tracker.obstacles(self.dynamic_snapshot,now)
            peer_shapes=dict(self.peer_shapes)
            self.record_geometry_timing=dict(self.geometry_timing)
            self.record_points=(self.obstacle_points,self.obstacle_points_stamp)
            self.record_peer_count=sum(0 <= now-shape['stamp'] <= self.scan_timeout for shape in peer_shapes.values())
            if self.robot_id not in poses or stamp is None:
                self.stop_motion();return
            odom_stamp,state,_=poses[self.robot_id]
            self.snapshot_stamps=(odom_stamp,stamp)
            self.metric_context=self.capture_metric_context(odom_stamp,state)
            if now-odom_stamp>self.stale or now-stamp>self.stale:
                self.stop_motion();self.record(now,state,None,observed,0.,'stale_data',begin);return
            if len(poses)<len(self.names):
                self.stop_motion();return
            if not self.warmed:
                dummy=padded_obstacles([],self.slots)
                problem=self.manager.problems[self.problem_key]
                try:
                    problem.solve(state,np.tile(state,(self.n+1,1)),dummy,(-.22,.22),None,
                        np.zeros((self.n,2)),np.zeros((self.n,2)))
                except RuntimeError:
                    self.stop_motion();return
                self.warmed=True
                rospy.set_param('/cbf/ready/'+self.name,True)
            if not self.running or not self.reference.history:
                self.stop_motion();return
            source=self.clock_sources.get(self.last_step,self.observer_source)
            age=max(0.,now-source)
            goals=self.reference.goals(age,state[2])
            if age>self.stale:
                self.stop_motion();self.record(now,state,goals,observed,age,'stale_observer',begin);return
        # All optimizer inputs are immutable snapshots. ROS observation callbacks remain live.
        nearby=sorted(observed,key=lambda o:np.linalg.norm(state[:2]-o['center'])-o['a'])[:self.max_static]
        by_name={o['name']:o for o in nearby}
        self.slot_names=[name if name in by_name else None for name in self.slot_names]
        for obs in nearby:
            if obs['name'] not in self.slot_names:
                self.slot_names[self.slot_names.index(None)]=obs['name']
        # Keep static track IDs and peer IDs in fixed slots. Nearest-distance
        # sorting must not silently retarget the persistent bypass memory.
        obstacles=[]
        for i,name in enumerate(self.slot_names):
            obstacles.append(by_name[name] if name is not None else
                dict(center=np.array([1000.+i,1000.]),a=.1,b=.1,theta=0.,name='inactive',guidance=False))
        peers=[]
        if self.peer_safety:
            for i,(s,x,v) in poses.items():
                if i==self.robot_id:continue
                if now-s>self.stale:
                    self.stop_motion();self.record(now,state,goals,obstacles,age,'stale_peer',begin);return
                peer=predicted_peer_envelope(i,s,x,v,now,self.dt,self.n,self.peer_radius,
                    peer_shapes.get(i),self.scan_timeout)
                peers.append((i,peer));obstacles.append(peer)
        self.publish_peer_geometry(rospy.Time.from_sec(now),peers)
        actual_obstacles=[o for o in obstacles if o['name']!='inactive']
        # Sensor envelopes may grow around an already physically free robot.
        # Forward invariance only applies when h0>=0. Outside that modeled set,
        # request slower positive recovery instead of an impossible .3-rate jump.
        # Keep the original gamma unchanged for all currently safe barriers.
        h0=barrier_values(np.tile(state,(self.n+1,1)),actual_obstacles,self.cfg['margin'])
        recovery_names={o['name'] for o,h in zip(actual_obstacles,h0[:,0]) if h < 0.}
        obstacles=[dict(o,gamma=self.recovery_gamma) if o['name'] in recovery_names else o for o in obstacles]
        actual_obstacles=[o for o in obstacles if o['name']!='inactive']
        obstacles=padded_obstacles(obstacles,self.slots)
        signature=[o['name'] for o in obstacles]
        memory=self.manager.robot_memory.get(self.name,{})
        index=memory.get('guide_index')
        if (index is not None and self.last_slot_signature is not None
                and signature[index]!=self.last_slot_signature[index]):
            memory['guide_index']=None;memory['guide_level']=0.
            memory.pop('guide_vector',None)
        self.last_slot_signature=signature
        self.input_ms=1000*(time.perf_counter()-begin)
        manager_begin=time.perf_counter()
        last=self.last_solution
        shift=1
        if self.last_solution_stamp is not None:
            age_seed=max(0.,now-self.last_solution_stamp)
            shift=max(1,int(round(age_seed/self.dt)))
            if age_seed>self.stale:last=None
        try:
            solution=self.manager.solve(state,goals,obstacles,
                (-.22,.22),last,self.name,
                deadline=begin+self.control_budget-.003 if self.cfg['realtime'] else None,
                applied_previous=self.applied_control.copy(),seed_shift=shift,cycle_token=self.cycle_id)
            control=solution['controls'][0].copy()
            if not solution['success']:
                status=('feasible_iterate' if solution.get('validated_steps',0)==self.n
                        else 'validated_one_step') if self.cfg['realtime'] else 'validated_fallback'
            else:status='ok'
            self.last_solution=solution;self.last_solution_stamp=now
            self.cycle_validated_steps=solution.get('validated_steps',self.n if solution['success'] else 1)
        except RuntimeError as exc:
            rospy.logwarn_throttle(2.,'%s: MPC stopped: %s',self.name,str(exc).splitlines()[-1])
            solution=None;control=np.zeros(2);status='solver_failed_stop'
            if not self.cfg['realtime']:self.last_solution=None
        self.manager_ms=1000*(time.perf_counter()-manager_begin)
        if self.closed:return
        late=(time.perf_counter()-begin>self.control_budget) if self.cfg['realtime'] else rospy.get_time()-now>.18
        if late:
            control=np.zeros(2)
            if not self.cfg['realtime']:self.last_solution=None
            # Preserve a real solver failure in diagnostics; do not relabel it
            # as a mere timeout just because the failed solve was also slow.
            if solution is not None:
                status='late_solution_stop' if solution['success'] else 'late_fallback_stop'
        cmd=Twist();cmd.linear.x=float(control[0]);cmd.angular.z=float(control[1]);self.cmd_pub.publish(cmd)
        self.applied_control=control.copy()
        self.last_command_time=rospy.get_time()
        self.last_command_wall=time.perf_counter();self.command_wall_ms=1000*(self.last_command_wall-begin)
        self.record(now,state,goals,actual_obstacles,age,status,begin,solution,control)

    def record(self,now,state,goals,obstacles,age,status,begin,solution=None,control=None):
        if self.closed or not self.running:return
        record_begin=time.perf_counter()
        control=np.zeros(2) if control is None else control
        step,leader,received=self.latest_leader
        source=self.clock_sources.get(step,received)
        clock=leader[0]+now-source
        # Diagnostic only: align the legacy next-step leader sample to wall/sim time.
        legacy_nominal=leader[1:3]+(now-source-self.dt)*leader[3:5]+self.reference.offset()
        metrics=self.metric_context
        reference=metrics['leader'];nominal=metrics['nominal'];plan=metrics['plan']
        formation_error=state[:2]-nominal
        plan_error=plan['position']-state[:2]
        observer_error=metrics['observer_aligned']-reference['position']
        evaluation_time=reference['logical_time'] if np.isfinite(reference['logical_time']) else clock+metrics['stamp']-now
        points,point_stamp=self.record_points
        peer_count=self.record_peer_count
        closest=float('nan')
        if point_stamp is not None and 0 <= now-point_stamp <= self.stale:
            closest=float(np.linalg.norm(points-nominal,axis=1).min()) if len(points) else float('inf')
        h=barrier_values(np.tile(state,(self.n+1,1)),[o for o in obstacles if o['name']!='inactive'],self.cfg['margin'])
        detail=self.manager.last_records.get(self.name,{})
        if detail.get('cycle_token')!=self.cycle_id:detail={}
        attempts=detail.get('attempt_trace',[])
        geometry=self.record_geometry_timing
        row=dict(time=clock,ros_time=now,robot=self.name,status=status,x=state[0],y=state[1],
            theta=state[2],v=control[0],omega=control[1],observer_x=metrics['observer_raw'][0],observer_y=metrics['observer_raw'][1],
            goal_x=goals[0,0] if goals is not None else np.nan,
            goal_y=goals[0,1] if goals is not None else np.nan,
            consensus_error_x=formation_error[0],consensus_error_y=formation_error[1],
            observer_step=metrics['observer_step'],leader_step=step,observer_age=age,
            odom_age=now-self.snapshot_stamps[0],geometry_age=now-self.snapshot_stamps[1],
            detected_ellipses=len(self.observed),static_constraints=sum(o['name']!='inactive' and not o.get('peer',False) and not o.get('dynamic',False) for o in obstacles),
            peer_constraints=sum(o.get('peer',False) for o in obstacles),
            min_h=float(h[:,0].min()) if h.size else np.nan,
            solve_ms=1000*(time.perf_counter()-begin),solver_success=int(solution is not None and solution['success']),
            solver_attempts=detail.get('attempts',0),guide_level=detail.get('guide_level',0.),
            guide_obstacle=detail.get('guide_obstacle',''),
            recovery_barriers=sum(o.get('gamma',self.cfg['gamma'])!=self.cfg['gamma'] for o in obstacles),
            laser_peer_constraints=sum(o.get('peer_source')=='laser' for o in obstacles),
            odometry_peer_constraints=sum(o.get('peer_source')=='odometry' for o in obstacles),
            geometry_ms=geometry.get('wall_ms',0.),peer_fit_ms=geometry.get('peer_wall_ms',0.),
            reference_x=nominal[0],reference_y=nominal[1],
            heading_error=wrap_angle(reference['heading']-state[2]),
            raw_min_range=self.raw_min_range,closest_obstacle_distance=closest,
            detected_peer_ellipses=peer_count,total_clusters_count=len(self.observed)+peer_count,
            cycle_id=self.cycle_id,input_ms=self.input_ms,manager_ms=self.manager_ms,
            command_wall_ms=self.command_wall_ms,record_ms=1000*(time.perf_counter()-record_begin),
            ipopt_status=attempts[-1].get('status','not_called') if attempts else 'not_called',
            ipopt_iterations=sum(max(0,t.get('iterations',0)) for t in attempts),
            ipopt_wall_ms=sum(t.get('nlp_ms',0.) for t in attempts),
            ipopt_cpu_ms=sum(t.get('nlp_cpu_ms',0.) for t in attempts),fallback_ms=detail.get('fallback_ms',0.),
            validated_steps=self.cycle_validated_steps,dual_warm=int(detail.get('dual_warm',False)),
            budget_hit=int(detail.get('budget_expired',False) or any(t.get('deadline_stop',False) or
                'WallTime' in t.get('status','') for t in attempts)),solver_seed=detail.get('seed',''),
            geometry_sequence=geometry.get('sequence',0),geometry_sample_stamp=geometry.get('stamp',0.),
            geometry_cpu_ms=geometry.get('cpu_ms',0.),geometry_peer_cpu_ms=geometry.get('peer_cpu_ms',0.),
            dynamic_tracks=sum(o.get('dynamic',False) for o in obstacles),
            missed_tracks=sum(o.get('missed',0)>0 for o in obstacles),
            max_obstacle_speed=max([float(np.linalg.norm(o.get('velocity',[0.,0.]))) for o in obstacles]+[0.]),
            motion_prediction_enabled=int(self.dynamic_prediction and self.dynamic_tracker.mode in ('cv','ca','singer','adaptive')),
            ca_prediction_enabled=int(self.dynamic_prediction and self.dynamic_tracker.mode=='ca'),
            singer_prediction_enabled=int(self.dynamic_prediction and self.dynamic_tracker.mode=='singer'),
            adaptive_prediction_enabled=int(self.dynamic_prediction and self.dynamic_tracker.mode=='adaptive'),
            adaptive_switches_active=sum(o.get('model_switches',0) for o in obstacles),
            progress_enabled=int(self.cfg.get('progress_guidance',False)),
            progress_mode={'direct':1,'yield':2,'ccw':3,'cw':4}.get(detail.get('progress',{}).get('mode'),0),
            progress_decision_id=detail.get('progress',{}).get('decision_id',0),
            progress_decision_ms=detail.get('progress_decision',{}).get('decision_ms',0.),
            max_obstacle_acceleration=max([float(np.linalg.norm(o.get('acceleration',[0.,0.]))) for o in obstacles]+[0.]),
            detected_static_ellipses=sum(not o.get('dynamic',False) for o in self.observed),
            detected_dynamic_ellipses=sum(o.get('dynamic',False) for o in self.observed),
            metrics_schema_version=METRICS_SCHEMA_VERSION,evaluation_time=evaluation_time,evaluation_ros_time=metrics['stamp'],
            reference_quality=reference['quality'],leader_reference_x=reference['position'][0],
            leader_reference_y=reference['position'][1],leader_reference_heading=reference['heading'],
            nominal_reference_x=nominal[0],nominal_reference_y=nominal[1],
            plan_reference_x=plan['position'][0],plan_reference_y=plan['position'][1],
            plan_execution_error_x=plan_error[0],plan_execution_error_y=plan_error[1],
            plan_reference_valid=plan['valid'],plan_reference_age=plan['age'],plan_source_cycle=plan['cycle'],
            observer_aligned_x=metrics['observer_aligned'][0],observer_aligned_y=metrics['observer_aligned'][1],
            observer_error_x=observer_error[0],observer_error_y=observer_error[1],
            legacy_consensus_error_x=state[0]-legacy_nominal[0],legacy_consensus_error_y=state[1]-legacy_nominal[1],
            legacy_heading_error=wrap_angle(state[2]-np.arctan2(leader[4],leader[3])))
        self.writer.writerow(row);self.rows+=1;self.counts[status]+=1
        if self.rows%10==0:self.stream.flush()
        self.diagnostics_pub.publish(String(data=json.dumps(row)))
        msg=RobotHistoryData(robot_id=self.robot_id,trajectory_time=clock,ros_timestamp=now,
            xc=float(state[0]),yc=float(state[1]),thetac=float(state[2]),vc=float(control[0]),wc=float(control[1]),
            xr=float(nominal[0]),yr=float(nominal[1]),zxeo_x=float(metrics['observer_raw'][0]),zxeo_y=float(metrics['observer_raw'][1]),
            xe=float(state[0]-nominal[0]),ye=float(state[1]-nominal[1]),thetae=row['heading_error'],
            computation_time=float(row['solve_ms']/1000.),full_cycle_time=float(row['solve_ms']/1000.),
            raw_min_obstacle_distance=self.raw_min_range,closest_obstacle_distance=closest,
            avoidance_force_magnitude=float('nan'),total_clusters_count=row['total_clusters_count'])
        self.history_pub.publish(msg)
        with self.lock:
            if (solution is not None and status in ('ok','feasible_iterate') and self.cycle_validated_steps>=self.n
                    and np.allclose(self.applied_control,control,atol=1e-9,rtol=0)):
                self.metric_plan=accepted_plan_reference(metrics['stamp'],state[:2],solution['plan_controls'],
                    self.dt,self.last_command_time,self.cycle_id)
            else:self.metric_plan=None
            plan_for_trace=self.metric_plan
        with self.trace_lock:
            if not self.closed:
                self.trace_stream.write(json.dumps(dict(cycle_id=self.cycle_id,trajectory_time=clock,
                    metrics_schema_version=METRICS_SCHEMA_VERSION,
                    accepted_auxiliary_plan=None if plan_for_trace is None else dict(
                        origin=plan_for_trace['origin'],issued_at=plan_for_trace['issued_at'],dt=plan_for_trace['dt'],
                        positions=plan_for_trace['positions'].tolist(),cycle=plan_for_trace['cycle']),
                    status=status,command=control.tolist(),control_wall_ms=row['solve_ms'],
                    command_wall_ms=self.command_wall_ms,cycle_end_ms=1000*(time.perf_counter()-begin),
                    cycle_cpu_ms=1000*(time.thread_time()-self.cycle_cpu_begin),manager=detail))+'\n')
                if self.rows%10==0:self.trace_stream.flush()
        rospy.loginfo_throttle(5.,'%s t=%.1f status=%s obs=%d e=%.3f solve=%.1fms',self.name,
            clock,status,len(obstacles),np.linalg.norm(state[:2]-nominal),row['solve_ms'])

    def close(self):
        if self.closed:return
        self.closed=True
        if hasattr(self,'timer'):self.timer.shutdown()
        if hasattr(self,'observer_timer'):self.observer_timer.shutdown()
        if hasattr(self,'watchdog_timer'):self.watchdog_timer.shutdown()
        if hasattr(self,'geometry_timer'):self.geometry_timer.shutdown()
        if hasattr(self,'control_start_timer'):self.control_start_timer.shutdown()
        if hasattr(self,'geometry_start_timer'):self.geometry_start_timer.shutdown()
        self.stop_motion()
        self.stream.flush();self.stream.close()
        with self.trace_lock:self.trace_stream.flush();self.trace_stream.close()
        (self.output/(self.name+'_summary.json')).write_text(json.dumps(
            dict(robot=self.name,rows=self.rows,status=dict(self.counts),watchdog_stops=self.watchdog_stops,
                 peer_laser_fits=dict(self.peer_laser_fits),solver=self.manager.summary(),
                 config=self.cfg,source='in-controller LaserScan geometry + odometry; no world truth'),indent=2))


if __name__=='__main__':
    rospy.init_node('tb3_controller_cbf')
    controller=TurtleBot3CBFController()
    rospy.spin()
