#!/usr/bin/env python3
"""ROS-independent 2-D adaptation of MPC-D-CBF obstacle geometry.

DBSCAN -> enclosing ellipse -> Hungarian association -> rolling shape variance.
The upstream CGAL ellipse solver is replaced by bounded numerical Khachiyan
iterations. Returned ellipses are enlarged to contain all measured points.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError, cKDTree


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
