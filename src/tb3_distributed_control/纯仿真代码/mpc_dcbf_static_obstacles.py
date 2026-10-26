#!/usr/bin/env python3
"""Shared static-obstacle adaptation of MPC-D-CBF's geometry and uncertainty.

No ROS, data association, obstacle state estimation, or nominal motion prediction.
MVEE solves the same minimum-area enclosing-ellipse problem as CGAL numerically.
Only the upstream covariance recursion is retained to generate axis inflation.
Angles use the upstream convention: counterclockwise, modulo pi.
"""
from collections import deque

import numpy as np
from scipy.spatial import ConvexHull


def minimum_enclosing_ellipse(points, tolerance=1e-9, max_iterations=50000):
    """Fit a full-rank 2-D point set, with an optimality and containment check.

    Khachiyan's dual algorithm; no PCA replacement, added padding, or artificial
    minimum semi-axis. Degenerate/non-convergent fits raise rather than silently
    changing the geometric problem. Hull reduction preserves the exact optimum.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite obstacle points of shape (n, 2)")
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("Invalid MVEE convergence settings")
    unique = np.unique(points, axis=0)
    if len(unique) < 3:
        raise ValueError("A nondegenerate ellipse needs at least three points")
    origin = unique.mean(axis=0)
    scale = float(np.ptp(unique, axis=0).max())
    if scale <= 0:
        raise ValueError("Degenerate obstacle")
    normalized = (unique - origin) / scale
    if np.linalg.matrix_rank(normalized) < 2:
        raise ValueError("Collinear obstacle points")
    support = normalized[ConvexHull(normalized).vertices]
    q = np.vstack((support.T, np.ones(len(support))))
    weights = np.full(len(support), 1.0 / len(support))
    for iteration in range(max_iterations):
        moment = (q * weights) @ q.T
        leverage = np.sum(q * np.linalg.solve(moment, q), axis=0)
        index = int(np.argmax(leverage))
        maximum = float(leverage[index])
        if maximum <= 3.0 * (1.0 + tolerance):
            break
        step = (maximum - 3.0) / (3.0 * (maximum - 1.0))
        weights *= 1.0 - step
        weights[index] += step
    else:
        raise RuntimeError("Minimum enclosing ellipse did not converge")
    local_center = support.T @ weights
    covariance = (support.T * weights) @ support - np.outer(local_center, local_center)
    values, vectors = np.linalg.eigh(covariance)
    if values.min() <= 0:
        raise ValueError("Degenerate ellipse covariance")
    axes = np.sqrt(2.0 * values[::-1]) * scale
    rotation = vectors[:, ::-1]
    center = origin + local_center * scale
    radial_max = float(np.sqrt(np.sum(((points - center) @ rotation / axes)**2, axis=1)).max())
    # Only roundoff/convergence correction, not a safety margin.
    axes *= max(1.0, radial_max) * (1.0 + 8.0*np.finfo(float).eps)
    theta = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    theta = (theta + np.pi/2) % np.pi - np.pi/2
    if abs(axes[0] - axes[1]) <= 1e-10 * axes[0]:
        # The major-axis direction of a circle is not identifiable.
        theta = 0.0
    return {"center": center, "a": float(axes[0]), "b": float(axes[1]),
            "theta": theta, "point_count": len(points),
            "fit_iterations": iteration + 1,
            "fit_dual_maximum": maximum,
            "fit_containment_max": float(np.max(np.sum(
                ((points-center) @ rotation / axes)**2, axis=1)))}


def reference_shape_variance(axis_history):
    """ellipse.hpp::ab_variance_calculation, population covariance (divide by n)."""
    history = np.asarray(axis_history, dtype=float)
    if history.ndim != 2 or history.shape[1] != 2 or not len(history):
        raise ValueError("Expected a nonempty history of two semi-axes")
    centered = history - history.mean(axis=0)
    trace = float(np.sum(centered**2) / len(history))
    return trace**1.3 + 0.0005 * max(0, 3-len(history))


def reference_covariance_matrices(dt):
    """obs_kf.cpp parameters and ordering: [x,y,a,b,theta,vx,vy,*,*]."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Expected a positive sample interval")
    transition = np.eye(9)
    transition[0, 5] = transition[1, 6] = dt
    return (transition, .00008*np.eye(9),
            np.diag([.0008, .0008, .008, .008, .0008*1e-8]),
            np.eye(5, 9), .025*np.eye(9))


def reference_axis_inflation(covariance):
    """Literal upstream rule; 0.95 is a multiplier, not a 95% quantile."""
    diagonal = np.maximum(np.diag(covariance), 0.0)
    pos_conf = .95 * np.sqrt(max(diagonal[0], diagonal[1]))
    axis_conf = .95 * np.sqrt(max(diagonal[2], diagonal[3]))
    return float(pos_conf + axis_conf)


class SharedStaticObstacleModel:
    """Fit once; advance one shared uncertainty state per simulation sample.

    Repeated observations are the same exact static ellipse, not noisy scans.
    Centers, nominal axes and orientations never undergo a Kalman state update.
    A/Q/R/H/P and covariance-only measurement updates match obs_kf.cpp.
    """
    def __init__(self, point_groups, names, horizon_steps, dt):
        if horizon_steps < 1 or len(point_groups) != len(names) or not len(names):
            raise ValueError("Invalid obstacle groups or MPC horizon")
        self.horizon_steps = int(horizon_steps)
        self.raw_obstacles = []
        for points, name in zip(point_groups, names):
            fitted = minimum_enclosing_ellipse(points)
            fitted["name"] = str(name)
            self.raw_obstacles.append(fitted)
        self.transition, self.process, self.measurement, self.observation, prior = reference_covariance_matrices(dt)
        self.covariances = [prior.copy() for _ in names]
        self.axis_histories = [deque(maxlen=10) for _ in names]
        self.frame_count = 0
        self.envelopes = None

    def advance(self):
        """Return N+1 covariance-inflated static ellipses, shared by all robots.

        Upstream prediction[0] follows one Predict_Cov() call. Preserve that
        indexing; N+1 entries cover all N transitions in this MPC formulation.
        The upstream first-observation branch emits nothing; because complete
        geometry is known here, startup uses P0 immediately instead of omitting
        obstacle constraints until the second frame.
        """
        envelopes = []
        h = self.observation
        for index, raw in enumerate(self.raw_obstacles):
            self.axis_histories[index].append([raw["a"], raw["b"]])
            variance = reference_shape_variance(self.axis_histories[index])
            measurement = self.measurement.copy()
            clipped = np.clip(variance, .005, .05)
            fraction = np.log10(clipped/.005) / np.log10(.05/.005)
            covariance_scale = 3000.0**fraction
            measurement[0, 0] *= covariance_scale
            measurement[1, 1] *= covariance_scale
            posterior = self.covariances[index]
            if self.frame_count:
                predicted = self.transition @ posterior @ self.transition.T + self.process
                innovation = h @ predicted @ h.T + measurement
                gain = np.linalg.solve(innovation, h @ predicted).T
                posterior = predicted - gain @ h @ predicted
                posterior = .5*(posterior + posterior.T)
            self.covariances[index] = posterior
            forecast = posterior.copy()
            axes, predicted_covariances = [], []
            for _ in range(self.horizon_steps + 1):
                forecast = self.transition @ forecast @ self.transition.T + self.process
                inflation = reference_axis_inflation(forecast)
                axes.append([raw["a"] + inflation, raw["b"] + inflation])
                predicted_covariances.append(forecast.copy())
            axes = np.asarray(axes)
            axes.setflags(write=False)
            obstacle = dict(raw)
            obstacle.update(a=float(axes[0, 0]), b=float(axes[0, 1]),
                            nominal_a=raw["a"], nominal_b=raw["b"],
                            prediction_axes=axes, shape_variance=variance,
                            measurement_covariance_scale=float(covariance_scale),
                            posterior_covariance=posterior.copy(),
                            prediction_covariances=np.asarray(predicted_covariances),
                            observation_frame=self.frame_count+1)
            obstacle["center"] = raw["center"].copy()
            obstacle["center"].setflags(write=False)
            envelopes.append(obstacle)
        self.frame_count += 1
        self.envelopes = tuple(envelopes)
        return self.envelopes


def obstacle_axes(obstacle, step=0):
    """Read the envelope for the same stage as the corresponding robot state."""
    if "prediction_axes" not in obstacle:
        return float(obstacle["a"]), float(obstacle["b"])
    if step < 0 or step >= len(obstacle["prediction_axes"]):
        raise IndexError("Obstacle prediction stage outside the MPC window")
    a, b = obstacle["prediction_axes"][step]
    return float(a), float(b)
