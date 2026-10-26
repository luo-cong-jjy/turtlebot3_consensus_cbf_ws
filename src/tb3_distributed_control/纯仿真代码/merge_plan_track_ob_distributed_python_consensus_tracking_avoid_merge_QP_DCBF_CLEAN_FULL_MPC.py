#!/usr/bin/env python3
import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import math
from collections import deque
from mpc_dcbf_static_obstacles import SharedStaticObstacleModel, obstacle_axes


def env_flag(name, default):
    """Read a boolean experiment switch without editing the simulation source."""
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def experiment_output_path(default_path):
    """Insert MPC_EXPERIMENT_TAG before the suffix to preserve comparison runs."""
    tag = os.getenv("MPC_EXPERIMENT_TAG", "").strip()
    path = Path(default_path)
    return str(path.with_name(f"{path.stem}_{tag}{path.suffix}")) if tag else str(path)

try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    ca = None
    CASADI_AVAILABLE = False

def _casadi_obstacle_barrier(pos_xy, obs, safe_margin, prediction_step=0):
    center = np.asarray(obs["center"], dtype=float).reshape(2)
    a, b = obstacle_axes(obs, prediction_step)
    theta = float(obs.get("theta", 0.0))
    c = ca.cos(theta)
    s = ca.sin(theta)
    dx = pos_xy[0] - center[0]
    dy = pos_xy[1] - center[1]
    # Fitted ellipses use the upstream CCW convention, independently of R1/R3.
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    ellipse_norm = (local_x / a) ** 2 + (local_y / b) ** 2
    return b * (ca.sqrt(ellipse_norm + 1e-12) - 1.0) - safe_margin


def _numeric_obstacle_barrier(pos_xy, obs, safe_margin, prediction_step=0):
    pos_xy = np.asarray(pos_xy, dtype=float).reshape(2)
    center = np.asarray(obs["center"], dtype=float).reshape(2)
    a, b = obstacle_axes(obs, prediction_step)
    theta = float(obs.get("theta", 0.0))
    c = np.cos(theta)
    s = np.sin(theta)
    dx, dy = pos_xy - center
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    ellipse_norm = (local_x / a) ** 2 + (local_y / b) ** 2
    return float(b * (np.sqrt(max(ellipse_norm, 0.0)) - 1.0) - safe_margin)


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def formation_offset(robot_index, time_value):
    """返回第robot_index个机器人的编队偏移，robot_index从0开始。"""
    angle = aa * time_value + robot_index * 2 * np.pi / Rnum
    return np.array([bb * 0.4 * np.cos(angle), bb * 0.4 * np.sin(angle)])


_mpc_reference_history = {}


def build_full_mpc_goal_states(
    robot_index, observer_pos, observer_vel, time_value, current_theta, static_avoidance=None
):
    """将本地观测序列对齐到MPC状态时刻，再预测领导者位置与编队偏移。

    保留原观测器：它用z0[k+1]校正z_hat[k]，匀速稳态下
    z_hat[k]=z0[k+1]。本函数接收刚更新的z_hat[k+1]，而MPC的
    x[0]是机器人k时刻状态，因此输入的有效时刻约为t[k]+2*dt。
    两拍校正是参考时标转换；不改动观测器或误差绘图数据。
    """
    goal_states = np.zeros((FULL_MPC_HORIZON_STEPS + 1, 3))
    observer_pos = np.asarray(observer_pos, dtype=float).reshape(2)
    observer_vel = np.asarray(observer_vel, dtype=float).reshape(2)
    static_avoidance = (
        np.zeros(2) if static_avoidance is None
        else np.asarray(static_avoidance, dtype=float).reshape(2)
    )

    history = _mpc_reference_history.setdefault(robot_index, deque(maxlen=11))
    if history and time_value < history[-1][0] - 1e-9:
        history.clear()
    if history and abs(time_value - history[-1][0]) < 1e-9:
        history.pop()
    history.append((float(time_value), observer_pos.copy()))
    # 用过去1秒的本地位置估计曲率，避免用匀速直线外推整个1秒窗口。
    # 前3秒观测器仍在收敛，保持线性预测；随后0.5秒平滑接入拟合。
    coefficients = None
    fit_weight = 0.0
    if len(history) == history.maxlen:
        times = np.array([s[0] - time_value for s in history])
        samples = np.array([s[1] for s in history])
        design = np.column_stack((np.ones(len(times)), times, times**2))
        coefficients = np.linalg.lstsq(design, samples, rcond=None)[0]
        fit_weight = np.clip((time_value - 3.0) / 0.5, 0.0, 1.0)

    def target_at(tau):
        relative_time = tau - 2 * t1
        predicted = observer_pos + observer_vel * relative_time
        if coefficients is not None:
            curved = np.array([1., relative_time, relative_time**2]) @ coefficients
            predicted = (1.0 - fit_weight) * predicted + fit_weight * curved
        return (predicted + formation_offset(robot_index, time_value + tau)
                - np.exp(-FULL_MPC_AVOIDANCE_GUIDE_DECAY * tau) * static_avoidance)

    for step in range(FULL_MPC_HORIZON_STEPS + 1):
        goal_states[step, 0:2] = target_at(step * t1)

    previous_heading = float(current_theta)
    # 当前离散模型先更新theta，再用theta[j+1]推进位置：
    # theta_ref[j+1]必须对应pos_ref[j] -> pos_ref[j+1]这段弦的方向。
    previous_pos = target_at(-t1)
    for step in range(FULL_MPC_HORIZON_STEPS + 1):
        delta = goal_states[step, 0:2] - previous_pos
        if np.linalg.norm(delta) > 1e-8:
            previous_heading = np.arctan2(delta[1], delta[0])
        goal_states[step, 2] = previous_heading
        previous_pos = goal_states[step, 0:2]
    return goal_states


def rollout_unicycle_state(current_state, controls, dt):
    """用小车模型滚动预测状态，主要用于初值和失败兜底。"""
    current_state = np.asarray(current_state, dtype=float).reshape(3)
    controls = np.asarray(controls, dtype=float).reshape(-1, 2)
    states = np.zeros((controls.shape[0] + 1, 3))
    states[0] = current_state
    for step, control in enumerate(controls):
        v, omega = control
        theta = states[step, 2]
        next_theta = wrap_angle(theta + dt * omega)
        states[step + 1, 0] = states[step, 0] + dt * v * np.cos(next_theta)
        states[step + 1, 1] = states[step, 1] + dt * v * np.sin(next_theta)
        states[step + 1, 2] = next_theta
    return states


def full_mpc_velocity_bounds(current_state, goal_states):
    """返回full MPC速度上下界。

    开源项目对固定终点局部规划采用“远处只前进、终点附近允许倒车”。
    本脚本是连续滚动轨迹跟踪，默认保留双向速度，避免D-CBF绕障时可行域过窄。
    """
    if not FULL_MPC_FORWARD_SPEED_BOUND:
        return -FULL_MPC_V_MAX, FULL_MPC_V_MAX
    if not FULL_MPC_ALLOW_REVERSE_NEAR_GOAL:
        return FULL_MPC_V_MIN, FULL_MPC_V_MAX

    current_state = np.asarray(current_state, dtype=float).reshape(3)
    goal_states = np.asarray(goal_states, dtype=float)
    local_goal = goal_states[-1, 0:2]
    distance_to_local_goal = np.linalg.norm(local_goal - current_state[0:2])
    goal_motion = np.linalg.norm(goal_states[-1, 0:2] - goal_states[0, 0:2])

    if (
        distance_to_local_goal <= FULL_MPC_REVERSE_ALLOW_DISTANCE
        and goal_motion <= FULL_MPC_REVERSE_ALLOW_MOTION
    ):
        return -FULL_MPC_V_MIN, FULL_MPC_V_MAX
    return FULL_MPC_V_MIN, FULL_MPC_V_MAX


def fallback_full_mpc_control(current_state, goal_states):
    """IPOPT失败时的轻量兜底控制，避免仿真中断。"""
    current_state = np.asarray(current_state, dtype=float).reshape(3)
    target = goal_states[min(1, len(goal_states) - 1), 0:2]
    diff = target - current_state[0:2]
    target_heading = np.arctan2(diff[1], diff[0]) if np.linalg.norm(diff) > 1e-8 else goal_states[0, 2]
    heading_error = wrap_angle(target_heading - current_state[2])
    distance = np.linalg.norm(diff)
    v_lower, v_upper = full_mpc_velocity_bounds(current_state, goal_states)
    v = np.clip(FULL_MPC_FALLBACK_K_V * distance * np.cos(heading_error), v_lower, v_upper)
    omega = np.clip(FULL_MPC_FALLBACK_K_W * heading_error, -FULL_MPC_OMEGA_MAX, FULL_MPC_OMEGA_MAX)
    controls = np.tile(np.array([v, omega]), (FULL_MPC_HORIZON_STEPS, 1))
    states = rollout_unicycle_state(current_state, controls, t1)
    plan_controls = np.diff(goal_states[:, 0:2], axis=0) / t1
    plan_controls = np.clip(plan_controls, -FULL_MPC_PLAN_V_MAX, FULL_MPC_PLAN_V_MAX)
    return controls[0], plan_controls[0], states, controls, plan_controls


def shifted_last_solution_from_current(current_state, goal_states, last_controls):
    """参考开源项目的失败处理：上一轮控制序列右移，并从当前状态重新预测。"""
    controls = np.asarray(last_controls, dtype=float)
    if controls.shape != (FULL_MPC_HORIZON_STEPS, 2):
        return None, None

    v_lower, v_upper = full_mpc_velocity_bounds(current_state, goal_states)
    shifted_controls = np.vstack((controls[1:], controls[-1:]))
    shifted_controls[:, 0] = np.clip(shifted_controls[:, 0], v_lower, v_upper)
    shifted_controls[:, 1] = np.clip(shifted_controls[:, 1], -FULL_MPC_OMEGA_MAX, FULL_MPC_OMEGA_MAX)
    shifted_states = rollout_unicycle_state(current_state, shifted_controls, t1)
    return shifted_states, shifted_controls


_mpc_solver_manager = None


def get_mpc_solver_manager():
    """一次构建共享参数化NLP；各机器人保留独立的热启动和停滞记录。"""
    global _mpc_solver_manager
    if _mpc_solver_manager is None:
        from mpc_dcbf_cached_solver import MPCSolverManager
        config = dict(
            horizon=FULL_MPC_HORIZON_STEPS, dt=t1, margin=DCBF_SAFE_MARGIN,
            gamma=DCBF_GAMMA, omega_max=FULL_MPC_OMEGA_MAX,
            plan_v_max=FULL_MPC_PLAN_V_MAX,
            delta_limit=FULL_MPC_ENABLE_DELTA_CONTROL_LIMIT,
            delta_v=FULL_MPC_DELTA_V_MAX, delta_w=FULL_MPC_DELTA_OMEGA_MAX,
            q_state=FULL_MPC_Q_STATE, q_terminal=FULL_MPC_Q_TERMINAL,
            q_plan=FULL_MPC_Q_PLAN, q_coupling=FULL_MPC_Q_PLAN_TRACK,
            r_control=FULL_MPC_R_CONTROL, r_plan=FULL_MPC_R_PLAN_CONTROL,
            r_delta=FULL_MPC_R_DELTA_CONTROL, r_delta_plan=FULL_MPC_R_DELTA_PLAN_CONTROL,
            max_iter=FULL_MPC_IPOPT_MAX_ITER,
            fallback_kv=FULL_MPC_FALLBACK_K_V, fallback_kw=FULL_MPC_FALLBACK_K_W,
            continuous_guidance=ENABLE_CONTINUOUS_GUIDANCE,
            guide_velocity_weight=FULL_MPC_GUIDE_VELOCITY_WEIGHT,
            guide_heading_weight=FULL_MPC_GUIDE_HEADING_WEIGHT,
            guide_delta_scale=FULL_MPC_GUIDE_DELTA_SCALE,
        )
        _mpc_solver_manager = MPCSolverManager(config, ENABLE_STALL_RECOVERY)
    return _mpc_solver_manager


def solve_full_mpc_dcbf(current_state, goal_states, obstacles,
                        last_solution=None, robot_label="robot"):
    """复用95维NLP，保留名义目标和硬安全约束，加入局部绕行软引导。"""
    if not CASADI_AVAILABLE:
        raise RuntimeError("Full MPC-D-CBF requires CasADi/IPOPT; no unchecked control will be applied")
    current_state = np.asarray(current_state, dtype=float).reshape(3)
    goal_states = np.asarray(goal_states, dtype=float).reshape(FULL_MPC_HORIZON_STEPS + 1, 3)
    active_obstacles = obstacles if ENABLE_DCBF else ()
    result = get_mpc_solver_manager().solve(
        current_state, goal_states, active_obstacles,
        full_mpc_velocity_bounds(current_state, goal_states), last_solution, robot_label,
    )
    if not result['success']:
        dcbf_mpc_failure_counts[robot_label] = dcbf_mpc_failure_counts.get(robot_label, 0) + 1
        if dcbf_mpc_failure_counts[robot_label] <= 3:
            print(f"[{robot_label}] MPC未成功，使用经单步安全核验的兜底: {result['status']}")
    return (result['controls'][0], result['plan_controls'][0], result['states'],
            result['controls'], result['plan_controls'], result['success'],
            result['obstacle_count'], result['min_h'])


def store_full_mpc_solution(
    control,
    plan_control,
    states_sol,
    k,
    x_hat_arr,
    y_hat_arr,
    u_arr,
    uz_arr,
):
    """把full MPC-D-CBF的一步解写回原脚本使用的数组。"""
    plan_vel = np.asarray(plan_control, dtype=float).reshape(2)
    # The planning state starts from the measured pose in the unified NLP.
    # Keep x_hat/y_hat on that optimized planning layer, not on the plant state.
    next_plan_pos = np.asarray(states_sol[0, 0:2], dtype=float) + t1 * plan_vel
    control = np.asarray(control, dtype=float).reshape(2)
    x_hat_arr[0, k + 1] = next_plan_pos[0]
    y_hat_arr[0, k + 1] = next_plan_pos[1]
    u_arr[:, k + 1] = plan_vel
    uz_arr[:, k + 1] = control


def plot_static_obstacles(ax):
    """绘制物理边界、一次拟合的MBE和最近一轮第0步安全包络。"""
    ax.plot(xobs[:200], yobs[:200], 'k.', markersize=2, alpha=0.7, label='Obstacle 1 boundary')
    ax.plot(xobs[200:400], yobs[200:400], '.', color='dimgray', markersize=2, alpha=0.7, label='Obstacle 2 boundary')
    ax.plot(xobs[400:600], yobs[400:600], '.', color='gray', markersize=2, alpha=0.7, label='Obstacle 3 boundary')
    ax.plot(xobs[600:800], yobs[600:800], '.', color='darkgray', markersize=2, alpha=0.7, label='Obstacle 4 boundary')

    if ENABLE_DCBF:
        theta_plot = np.linspace(0, 2 * np.pi, 240)
        for obs_idx, obs in enumerate(static_obstacle_model.raw_obstacles, 1):
            local_x = obs["a"] * np.cos(theta_plot)
            local_y = obs["b"] * np.sin(theta_plot)
            c, s = np.cos(obs["theta"]), np.sin(obs["theta"])
            ax.plot(obs["center"][0] + c*local_x - s*local_y,
                    obs["center"][1] + s*local_x + c*local_y,
                    ':', color='tab:blue', linewidth=1,
                    label='Minimum enclosing ellipses' if obs_idx == 1 else None)
        for obs_idx, obs in enumerate(dcbf_obstacles, 1):
            center = np.asarray(obs["center"], dtype=float).reshape(2)
            a = float(obs["a"])
            b = float(obs["b"])
            theta_obs = float(obs.get("theta", 0.0))
            safe_scale = 1.0 + DCBF_SAFE_MARGIN / max(b, 1e-6)
            local_x = safe_scale * a * np.cos(theta_plot)
            local_y = safe_scale * b * np.sin(theta_plot)
            safe_x = center[0] + local_x * np.cos(theta_obs) - local_y * np.sin(theta_obs)
            safe_y = center[1] + local_x * np.sin(theta_obs) + local_y * np.cos(theta_obs)
            ax.plot(
                safe_x,
                safe_y,
                '--',
                color='black',
                linewidth=1.2,
                alpha=0.8,
                label=f'Latest D-CBF safe boundary {obs_idx}',
            )


def recorded_obstacle_barriers(pos_xy, sample_index):
    """按实际执行该步控制时的包络计算h，避免用末帧包络回算全程。"""
    if not ENABLE_DCBF:
        return []
    if sample_index == 0 or not dcbf_obstacle_envelope_history:
        envelopes = dcbf_obstacle_envelope_history[0] if dcbf_obstacle_envelope_history else dcbf_obstacles
        stage = 0
    else:
        envelopes = dcbf_obstacle_envelope_history[min(sample_index - 1, len(dcbf_obstacle_envelope_history) - 1)]
        stage = 1
    return [_numeric_obstacle_barrier(pos_xy, obs, DCBF_SAFE_MARGIN, stage)
            for obs in envelopes]


def build_full_mpc_diagnostic_rows():
    """汇总full MPC-D-CBF的安全裕度、队形误差和控制平滑性指标。"""
    rows = []
    robot_data = [
        ("Robot 1", 0, xc1[0, :], yc1[0, :], x_hat1[0, :], y_hat1[0, :], zxeo_1, u1, uz1),
        ("Robot 2", 1, xc2[0, :], yc2[0, :], x_hat2[0, :], y_hat2[0, :], zxeo_2, u2, uz2),
        ("Robot 3", 2, xc3[0, :], yc3[0, :], x_hat3[0, :], y_hat3[0, :], zxeo_3, u3, uz3),
        ("Robot 4", 3, xc4[0, :], yc4[0, :], x_hat4[0, :], y_hat4[0, :], zxeo_4, u4, uz4),
        ("Robot 5", 4, xc5[0, :], yc5[0, :], x_hat5[0, :], y_hat5[0, :], zxeo_5, u5, uz5),
    ]

    for robot_name, robot_index, x_arr, y_arr, plan_x_arr, plan_y_arr, zxeo_arr, u_arr, uz_arr in robot_data:
        sample_count = min(len(t), len(x_arr), len(y_arr), len(plan_x_arr), len(plan_y_arr), zxeo_arr.shape[1], uz_arr.shape[1])
        actual_pos = np.column_stack((x_arr[:sample_count], y_arr[:sample_count]))
        plan_pos = np.column_stack((plan_x_arr[:sample_count], plan_y_arr[:sample_count]))
        nominal_pos = np.zeros_like(actual_pos)
        for sample_idx in range(sample_count):
            nominal_pos[sample_idx, :] = zxeo_arr[:, sample_idx] + formation_offset(robot_index, t[sample_idx])

        formation_error = np.linalg.norm(actual_pos - nominal_pos, axis=1)
        # 保留历史formation_error口径（相对原始观测器）；另存与一致误差图
        # 完全相同的actual - leader - formation_offset，避免混淆两类指标。
        leader_nominal = np.array([
            z0[:, s] + formation_offset(robot_index, t[s])
            for s in range(sample_count)
        ])
        consensus_error = actual_pos - leader_nominal
        if ENABLE_DCBF and len(dcbf_obstacles) > 0:
            h_values = np.array([
                min(recorded_obstacle_barriers(pos, sample_idx))
                for sample_idx, pos in enumerate(actual_pos)
            ])
        else:
            h_values = np.full(sample_count, np.nan)

        v_values = uz_arr[0, :sample_count]
        omega_values = uz_arr[1, :sample_count]
        plan_tracking_error = np.linalg.norm(actual_pos - plan_pos, axis=1)
        plan_control_variation = np.sum(np.abs(np.diff(u_arr[:, :sample_count], axis=1)), axis=1)
        rows.append({
            "robot": robot_name,
            "formation_error_mean": float(np.mean(formation_error)),
            "formation_error_max": float(np.max(formation_error)),
            "formation_error_final": float(formation_error[-1]),
            "consensus_x_rms": float(np.sqrt(np.mean(consensus_error[:, 0]**2))),
            "consensus_y_rms": float(np.sqrt(np.mean(consensus_error[:, 1]**2))),
            "consensus_x_final": float(consensus_error[-1, 0]),
            "consensus_y_final": float(consensus_error[-1, 1]),
            "plan_tracking_error_mean": float(np.mean(plan_tracking_error)),
            "plan_tracking_error_max": float(np.max(plan_tracking_error)),
            "actual_min_h": float(np.nanmin(h_values)) if np.any(np.isfinite(h_values)) else np.nan,
            "actual_violate_steps": int(np.sum(h_values < -1e-6)) if np.any(np.isfinite(h_values)) else 0,
            "v_total_variation": float(np.sum(np.abs(np.diff(v_values)))) if sample_count > 1 else 0.0,
            "omega_total_variation": float(np.sum(np.abs(np.diff(omega_values)))) if sample_count > 1 else 0.0,
            "omega_abs_mean": float(np.mean(np.abs(omega_values))),
            "omega_sign_changes": int(np.sum(np.diff(np.signbit(omega_values)) != 0)) if sample_count > 1 else 0,
            "ux_total_variation": float(plan_control_variation[0]) if sample_count > 1 else 0.0,
            "uy_total_variation": float(plan_control_variation[1]) if sample_count > 1 else 0.0,
            "v_saturation_steps": int(np.sum(np.isclose(np.abs(v_values), FULL_MPC_V_MAX, atol=1e-4))),
            "omega_saturation_steps": int(np.sum(np.isclose(np.abs(omega_values), FULL_MPC_OMEGA_MAX, atol=1e-4))),
        })
    return rows


def build_full_mpc_timeseries_rows():
    """保存逐时刻诊断数据，用于定位速度/角速度跳变和具体触发障碍物。"""
    rows = []
    robot_data = [
        ("Robot 1", 0, xc1[0, :], yc1[0, :], zxeo_1, uz1),
        ("Robot 2", 1, xc2[0, :], yc2[0, :], zxeo_2, uz2),
        ("Robot 3", 2, xc3[0, :], yc3[0, :], zxeo_3, uz3),
        ("Robot 4", 3, xc4[0, :], yc4[0, :], zxeo_4, uz4),
        ("Robot 5", 4, xc5[0, :], yc5[0, :], zxeo_5, uz5),
    ]

    for robot_name, robot_index, x_arr, y_arr, zxeo_arr, uz_arr in robot_data:
        sample_count = min(len(t), len(x_arr), len(y_arr), zxeo_arr.shape[1], uz_arr.shape[1])
        previous_v = float(uz_arr[0, 0])
        previous_omega = float(uz_arr[1, 0])
        for sample_idx in range(sample_count):
            pos = np.array([x_arr[sample_idx], y_arr[sample_idx]], dtype=float)
            nominal_pos = zxeo_arr[:, sample_idx] + formation_offset(robot_index, t[sample_idx])
            leader_nominal = z0[:, sample_idx] + formation_offset(robot_index, t[sample_idx])
            obstacle_h = recorded_obstacle_barriers(pos, sample_idx)
            if obstacle_h:
                nearest_idx = int(np.argmin(obstacle_h))
                min_h = float(obstacle_h[nearest_idx])
                nearest_name = dcbf_obstacles[nearest_idx].get("name", f"obs_{nearest_idx + 1}")
            else:
                min_h = np.nan
                nearest_name = ""

            v_value = float(uz_arr[0, sample_idx])
            omega_value = float(uz_arr[1, sample_idx])
            row = {
                "time": float(t[sample_idx]),
                "robot": robot_name,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "nominal_x": float(nominal_pos[0]),
                "nominal_y": float(nominal_pos[1]),
                "formation_error": float(np.linalg.norm(pos - nominal_pos)),
                "leader_nominal_x": float(leader_nominal[0]),
                "leader_nominal_y": float(leader_nominal[1]),
                "consensus_error_x": float(pos[0] - leader_nominal[0]),
                "consensus_error_y": float(pos[1] - leader_nominal[1]),
                "v": v_value,
                "omega": omega_value,
                "delta_v": v_value - previous_v if sample_idx > 0 else 0.0,
                "delta_omega": omega_value - previous_omega if sample_idx > 0 else 0.0,
                "nearest_obstacle": nearest_name,
                "min_h": min_h,
                "mpc_active_obstacles": int(dcbf_mpc_active_obstacle_history[robot_index, min(max(sample_idx - 1, 0), dcbf_mpc_active_obstacle_history.shape[1] - 1)]),
                "mpc_success": int(dcbf_mpc_success_history[robot_index, min(max(sample_idx - 1, 0), dcbf_mpc_success_history.shape[1] - 1)]),
                "mpc_window_min_h": float(dcbf_mpc_min_h_history[robot_index, min(max(sample_idx - 1, 0), dcbf_mpc_min_h_history.shape[1] - 1)]),
            }
            for obs_idx, obs_h in enumerate(obstacle_h, 1):
                row[f"h_obs_{obs_idx}"] = float(obs_h)
            if _mpc_solver_manager is not None:
                records = _mpc_solver_manager.histories.get(f"robot{robot_index + 1}", [])
                if records:
                    detail = records[min(max(sample_idx - 1, 0), len(records) - 1)]
                    row.update(solver_attempts=detail['attempts'], solve_wall_ms=detail['wall_ms'],
                               recovery_triggered=int(detail['recovery_triggered']),
                               recovery_improved=int(detail['recovery_improved']),
                               selected_seed=detail['seed'], solver_status=detail['status'],
                               min_cbf_slack=detail['min_cbf_slack'],
                               continuous_guide_level=detail.get('guide_level', 0.),
                               continuous_guide_obstacle=detail.get('guide_obstacle'),
                               continuous_guide_side=detail.get('guide_side'),
                               guide_route_switches=detail.get('route_switches', 0),
                               guide_route_cost_ccw=detail.get('route_cost_ccw'),
                               guide_route_cost_cw=detail.get('route_cost_cw'),
                               guide_projected_follow=int(detail.get('projected_follow', False)))
            rows.append(row)
            previous_v = v_value
            previous_omega = omega_value
    return rows


def print_and_save_full_mpc_diagnostics():
    """打印并保存实验诊断数据，便于和轨迹图一起分析摆动原因。"""
    rows = build_full_mpc_diagnostic_rows()
    if _mpc_solver_manager is not None:
        summary = _mpc_solver_manager.summary()
        print(f"\n缓存NLP构建次数: {summary['cached_problem_builds']}，停滞脱困: {summary['recovery_enabled']}")
        for label, item in summary['robots'].items():
            print(f"{label}: 求解尝试={item['solver_attempts']}，脱困触发={item['recovery_triggers']}，"
                  f"脱困改进={item['recovery_improvements']}，平均控制求解={item['cycle_mean_ms']:.2f}ms")
        output_path = Path(DCBF_SOLVER_STATS_PATH)
        if not output_path.is_absolute():
            output_path = Path(__file__).resolve().parent / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print("\n=== Full MPC-D-CBF轨迹/控制诊断 ===")
    print(
        f"控制权重R={FULL_MPC_R_CONTROL}, "
        f"角速度上限={FULL_MPC_OMEGA_MAX:.3f}rad/s, "
        f"速度上限={FULL_MPC_V_MAX:.3f}m/s"
    )
    for row in rows:
        print(
            f"{row['robot']}: 相对原始观测器的队形误差均值={row['formation_error_mean']:.4f}m, "
            f"最大={row['formation_error_max']:.4f}m, 最终={row['formation_error_final']:.4f}m, "
            f"真实领导者一致误差最终值=({row['consensus_x_final']:.5f}, {row['consensus_y_final']:.5f})m, "
            f"规划-实际误差均值={row['plan_tracking_error_mean']:.4f}m, 最大={row['plan_tracking_error_max']:.4f}m, "
            f"实际最小h={row['actual_min_h']:.5f}, 违反步数={row['actual_violate_steps']}, "
            f"v总变差={row['v_total_variation']:.3f}, "
            f"omega总变差={row['omega_total_variation']:.3f}, "
            f"omega平均绝对值={row['omega_abs_mean']:.3f}, "
            f"omega换向={row['omega_sign_changes']}, "
            f"v饱和={row['v_saturation_steps']}, omega饱和={row['omega_saturation_steps']}"
        )

    if SAVE_DCBF_DIAGNOSTIC_CSV and rows:
        output_path = Path(DCBF_DIAGNOSTIC_CSV_PATH)
        if not output_path.is_absolute():
            output_path = Path(__file__).resolve().parent / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Full MPC-D-CBF诊断数据已保存: {output_path}")

    if SAVE_DCBF_TIMESERIES_CSV:
        timeseries_rows = build_full_mpc_timeseries_rows()
        output_path = Path(DCBF_TIMESERIES_CSV_PATH)
        if not output_path.is_absolute():
            output_path = Path(__file__).resolve().parent / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if timeseries_rows:
            fieldnames = list(timeseries_rows[0].keys())
            with output_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(timeseries_rows)
            print(f"Full MPC-D-CBF逐时刻诊断数据已保存: {output_path}")


def save_trajectory_animation():
    """保存机器人全过程轨迹GIF，展示实际轨迹、规划轨迹、障碍物和安全边界。"""
    if not SAVE_TRAJECTORY_GIF:
        return

    output_path = Path(TRAJECTORY_GIF_PATH).expanduser()
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    actual_x = [xc1[0, :], xc2[0, :], xc3[0, :], xc4[0, :], xc5[0, :]]
    actual_y = [yc1[0, :], yc2[0, :], yc3[0, :], yc4[0, :], yc5[0, :]]
    ref_x = [x_hat1[0, :], x_hat2[0, :], x_hat3[0, :], x_hat4[0, :], x_hat5[0, :]]
    ref_y = [y_hat1[0, :], y_hat2[0, :], y_hat3[0, :], y_hat4[0, :], y_hat5[0, :]]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:Rnum]

    all_x = np.concatenate(actual_x + ref_x + [x0[0, :], xobs])
    all_y = np.concatenate(actual_y + ref_y + [y0[0, :], yobs])
    pad = 0.35

    fig, ax = plt.subplots(figsize=(9, 7))
    plot_static_obstacles(ax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(float(np.nanmin(all_x)) - pad, float(np.nanmax(all_x)) + pad)
    ax.set_ylim(float(np.nanmin(all_y)) - pad, float(np.nanmax(all_y)) + pad)
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    ax.set_title('Robot Trajectories with MPC-D-CBF')
    ax.grid(True, alpha=0.3)

    actual_lines = []
    ref_lines = []
    actual_points = []
    ref_points = []
    for robot_idx in range(Rnum):
        actual_line, = ax.plot([], [], '-', linewidth=2, color=colors[robot_idx], label=f'Robot_{robot_idx + 1}')
        ref_line, = ax.plot([], [], '--', linewidth=1.1, color=colors[robot_idx], alpha=0.55, label=f'Ref_{robot_idx + 1}')
        actual_point, = ax.plot([], [], 'o', color=colors[robot_idx], markersize=5)
        ref_point, = ax.plot([], [], 's', color=colors[robot_idx], markersize=3, alpha=0.55)
        actual_lines.append(actual_line)
        ref_lines.append(ref_line)
        actual_points.append(actual_point)
        ref_points.append(ref_point)

    leader_line, = ax.plot([], [], ':', linewidth=1.5, color='black', label='Robot_L')
    centroid_line, = ax.plot([], [], '-', linewidth=1.5, color='tab:pink', label='Actual centroid')
    time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, va='top')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    last_index = len(t) - 1
    stride = max(1, int(TRAJECTORY_GIF_STRIDE))
    frame_indices = list(range(0, last_index + 1, stride))
    if frame_indices[-1] != last_index:
        frame_indices.append(last_index)

    def update(frame_index):
        end = frame_index + 1
        for robot_idx in range(Rnum):
            actual_lines[robot_idx].set_data(actual_x[robot_idx][:end], actual_y[robot_idx][:end])
            ref_lines[robot_idx].set_data(ref_x[robot_idx][:end], ref_y[robot_idx][:end])
            actual_points[robot_idx].set_data([actual_x[robot_idx][frame_index]], [actual_y[robot_idx][frame_index]])
            ref_points[robot_idx].set_data([ref_x[robot_idx][frame_index]], [ref_y[robot_idx][frame_index]])
        leader_line.set_data(x0[0, :end], y0[0, :end])
        centroid_line.set_data(xccc[0, :end], yccc[0, :end])
        time_text.set_text(f't = {t[frame_index]:.1f}s')
        return actual_lines + ref_lines + actual_points + ref_points + [leader_line, centroid_line, time_text]

    animation = FuncAnimation(fig, update, frames=frame_indices, interval=TRAJECTORY_GIF_INTERVAL_MS, blit=True)
    writer = PillowWriter(fps=TRAJECTORY_GIF_FPS)
    animation.save(str(output_path), writer=writer, dpi=TRAJECTORY_GIF_DPI)
    plt.close(fig)
    print(f"\n轨迹动图已保存: {output_path}")



# 基本参数设置
Nu = 2  # 保留原数组维度：当前步控制和下一步控制各一组

# D-CBF参数：参考MPC-D-CBF-main的离散CBF思想，在Full MPC中直接作为非线性约束求解。
ENABLE_DCBF = True
DCBF_GAMMA = 0.3
DCBF_SAFE_MARGIN = 0.12
dcbf_mpc_failure_counts = {}

FULL_MPC_HORIZON_STEPS = 10
FULL_MPC_FORWARD_SPEED_BOUND = False
FULL_MPC_ALLOW_REVERSE_NEAR_GOAL = False
FULL_MPC_V_MIN = 0.02
FULL_MPC_V_MAX = 0.22
FULL_MPC_OMEGA_MAX = 2.84  # 与QP基准的实际角速度限幅一致
FULL_MPC_REVERSE_ALLOW_DISTANCE = 0.10
FULL_MPC_REVERSE_ALLOW_MOTION = 0.03
FULL_MPC_Q_STATE = [35.0, 35.0, 0.6]
FULL_MPC_Q_TERMINAL = [80.0, 80.0, 1.0]
FULL_MPC_Q_PLAN = [20.0, 20.0]
FULL_MPC_Q_PLAN_TRACK = [30.0, 30.0]
# 保留此前调节的控制代价；本次仅将角速度上限对齐QP基准。
FULL_MPC_R_CONTROL = [0.15, 0.04]
FULL_MPC_R_PLAN_CONTROL = [0.04, 0.04]
# 针对完整轨迹中v/omega跳变，加入控制增量惩罚抑制相邻预测步控制抖动。
FULL_MPC_R_DELTA_CONTROL = [0.05, 0.16]
FULL_MPC_R_DELTA_PLAN_CONTROL = [0.03, 0.03]
FULL_MPC_PLAN_V_MAX = 0.22
FULL_MPC_AVOIDANCE_GUIDE_DECAY = 0.8
FULL_MPC_AVOIDANCE_GUIDE_FILTER = 0.78
FULL_MPC_AVOIDANCE_GUIDE_MAX = 0.12
# 默认关闭规划层势场引导，主算法仅由统一MPC目标和实际状态D-CBF避障。
# 对照实验可设置环境变量 MPC_ENABLE_PLAN_AVOIDANCE_GUIDE=1 开启。
ENABLE_PLAN_AVOIDANCE_GUIDE = env_flag("MPC_ENABLE_PLAN_AVOIDANCE_GUIDE", False)
# 可设 MPC_ENABLE_STALL_RECOVERY=0 单独验证缓存求解器，不启用多初值脱困。
ENABLE_STALL_RECOVERY = env_flag("MPC_ENABLE_STALL_RECOVERY", True)
ENABLE_CONTINUOUS_GUIDANCE = env_flag("MPC_ENABLE_CONTINUOUS_GUIDANCE", True)
# 保持原名义参考；在NLP内部添加可平滑启退的绕行软代价。
FULL_MPC_GUIDE_VELOCITY_WEIGHT = 600.0
FULL_MPC_GUIDE_HEADING_WEIGHT = 2.0
FULL_MPC_GUIDE_DELTA_SCALE = 8.0
FULL_MPC_ENABLE_DELTA_CONTROL_LIMIT = False
FULL_MPC_DELTA_V_MAX = 0.06
FULL_MPC_DELTA_OMEGA_MAX = 0.50
FULL_MPC_IPOPT_MAX_ITER = 1200
FULL_MPC_FALLBACK_K_V = 1.8
FULL_MPC_FALLBACK_K_W = 3.0

SAVE_TRAJECTORY_GIF = env_flag("MPC_SAVE_GIF", True)
TRAJECTORY_GIF_PATH = experiment_output_path("output/trajectory_full_mpc_dcbf.gif")
TRAJECTORY_GIF_STRIDE = 5
TRAJECTORY_GIF_FPS = 20
TRAJECTORY_GIF_INTERVAL_MS = 50
TRAJECTORY_GIF_DPI = 110
SAVE_DCBF_DIAGNOSTIC_CSV = True
DCBF_DIAGNOSTIC_CSV_PATH = experiment_output_path("output/full_mpc_dcbf_diagnostics.csv")
SAVE_DCBF_TIMESERIES_CSV = True
DCBF_TIMESERIES_CSV_PATH = experiment_output_path("output/full_mpc_dcbf_timeseries.csv")
DCBF_SOLVER_STATS_PATH = experiment_output_path("output/full_mpc_solver_stats.json")
SHOW_PLOTS = env_flag("MPC_SHOW_PLOTS", True)

# 时间设置
t1 = 0.1  # 采样时间
T = float(os.getenv("MPC_SIM_TIME", "180")) # 总时间(8字形：180s，实物轨迹：70s，直线轨迹：28s)
i = np.arange(1, T/t1 + 1)  # 时间序列
Rnum = 5  # 机器人数量

# 轨迹类型选择（新增）
trajectory_type = "" # 可选项："eight"(8字形), "real_experiment_trajectory"(实验轨迹), "circle_trajectory"(圆形轨迹), "real_experiment_trajectory_line"(实验直线轨迹)

# 设置邻接矩阵

# A = np.array([[0, 0, 1, 0, 0],
#               [1, 0, 1, 0, 0],
#               [0, 0, 0, 1, 0],
#               [1, 0, 0, 0, 0],
#               [0, 1, 0, 0, 0]])

# 现在观测器部分已经支持动态邻接矩阵，可以直接修改A矩阵来切换拓扑
# 例如：
A = np.array([[0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0]])

D = np.diag(np.sum(A, axis=1))
L = D - A
B = np.diag([1, 0, 0, 0, 0])
F = L + B
# 参考轨迹
aa = 0
bb = 1

# 初始化控制输入矩阵
ux1 = np.zeros((1, len(i)+1))
uy1 = np.zeros((1, len(i)+1))

ux2 = np.zeros((1, len(i)+1))
uy2 = np.zeros((1, len(i)+1))

ux3 = np.zeros((1, len(i)+1))
uy3 = np.zeros((1, len(i)+1))

ux4 = np.zeros((1, len(i)+1))
uy4 = np.zeros((1, len(i)+1))

ux5 = np.zeros((1, len(i)+1))
uy5 = np.zeros((1, len(i)+1))

# 初始化虚拟领导者控制输入
u0x = np.zeros((1, len(i)+1))
u0x[0, 0] = 0

u0y = np.zeros((1, len(i)+1))
u0y[0, 0] = 0

u0 = np.zeros((2, len(i)+1))
u0[:, 0] = np.array([u0x[0, 0], u0y[0, 0]])

u1 = np.zeros((2, len(i)+1))

u2 = np.zeros((2, len(i)+1))

u3 = np.zeros((2, len(i)+1))

u4 = np.zeros((2, len(i)+1))

u5 = np.zeros((2, len(i)+1))

# 初始化位置矩阵
#S形
# x_hat1 = np.zeros((1, len(i)+1))
# y_hat1 = np.zeros((1, len(i)+1))
# x_hat1[:, 0] = 0
# y_hat1[:, 0] = 0.5

# x_hat2 = np.zeros((1, len(i)+1))
# y_hat2 = np.zeros((1, len(i)+1))
# x_hat2[:, 0] = -1
# y_hat2[:, 0] = 0.5

# x_hat3 = np.zeros((1, len(i)+1))
# y_hat3 = np.zeros((1, len(i)+1))
# x_hat3[:, 0] = -0.5
# y_hat3[:, 0] = 0

# x_hat4 = np.zeros((1, len(i)+1))
# y_hat4 = np.zeros((1, len(i)+1))
# x_hat4[:, 0] = 0.5
# y_hat4[:, 0] = 0

# x_hat5 = np.zeros((1, len(i)+1))
# y_hat5 = np.zeros((1, len(i)+1))
# x_hat5[:, 0] = 1
# y_hat5[:, 0] = 0

#直线
x_hat1 = np.zeros((1, len(i)+1))
y_hat1 = np.zeros((1, len(i)+1))
x_hat1[:, 0] = 0.4
y_hat1[:, 0] = 0.0

x_hat2 = np.zeros((1, len(i)+1))
y_hat2 = np.zeros((1, len(i)+1))
x_hat2[:, 0] = 0.2
y_hat2[:, 0] = 0.4

x_hat3 = np.zeros((1, len(i)+1))
y_hat3 = np.zeros((1, len(i)+1))
x_hat3[:, 0] = -0.2
y_hat3[:, 0] = 0.2

x_hat4 = np.zeros((1, len(i)+1))
y_hat4 = np.zeros((1, len(i)+1))
x_hat4[:, 0] = -0.3
y_hat4[:, 0] = -0.3

x_hat5 = np.zeros((1, len(i)+1))
y_hat5 = np.zeros((1, len(i)+1))
x_hat5[:, 0] = 0.2
y_hat5[:, 0] = -0.3



# 初始化参考轨迹x和y方向坐标矩阵
xr1 = np.zeros((1, len(i)+1))
xr1[:, 0] = x_hat1[:, 0]
yr1 = np.zeros((1, len(i)+1))
yr1[:, 0] = y_hat1[:, 0]

xr2 = np.zeros((1, len(i)+1))
xr2[:, 0] = x_hat2[:, 0]
yr2 = np.zeros((1, len(i)+1))
yr2[:, 0] = y_hat2[:, 0]

xr3 = np.zeros((1, len(i)+1))
xr3[:, 0] = x_hat3[:, 0]
yr3 = np.zeros((1, len(i)+1))
yr3[:, 0] = y_hat3[:, 0]

xr4 = np.zeros((1, len(i)+1))
xr4[:, 0] = x_hat4[:, 0]
yr4 = np.zeros((1, len(i)+1))
yr4[:, 0] = y_hat4[:, 0]

xr5 = np.zeros((1, len(i)+1))
xr5[:, 0] = x_hat5[:, 0]
yr5 = np.zeros((1, len(i)+1))
yr5[:, 0] = y_hat5[:, 0]

# 初始化当前位置
xc1 = np.zeros((1, len(i)+1))
xc1[:, 0] = xr1[:, 0]
yc1 = np.zeros((1, len(i)+1))
yc1[:, 0] = yr1[:, 0]

xc2 = np.zeros((1, len(i)+1))
xc2[:, 0] = xr2[:, 0]
yc2 = np.zeros((1, len(i)+1))
yc2[:, 0] = yr2[:, 0]

xc3 = np.zeros((1, len(i)+1))
xc3[:, 0] = xr3[:, 0]
yc3 = np.zeros((1, len(i)+1))
yc3[:, 0] = yr3[:, 0]

xc4 = np.zeros((1, len(i)+1))
xc4[:, 0] = xr4[:, 0]
yc4 = np.zeros((1, len(i)+1))
yc4[:, 0] = yr4[:, 0]

xc5 = np.zeros((1, len(i)+1))
xc5[:, 0] = xr5[:, 0]
yc5 = np.zeros((1, len(i)+1))
yc5[:, 0] = yr5[:, 0]

# 初始化误差矩阵
xe1 = np.zeros((1, len(i)))
ye1 = np.zeros((1, len(i)))

xe2 = np.zeros((1, len(i)))
ye2 = np.zeros((1, len(i)))

xe3 = np.zeros((1, len(i)))
ye3 = np.zeros((1, len(i)))

xe4 = np.zeros((1, len(i)))
ye4 = np.zeros((1, len(i)))

xe5 = np.zeros((1, len(i)))
ye5 = np.zeros((1, len(i)))

# 初始化角度矩阵
thetar1 = np.zeros((1, len(i)+1))
thetac1 = np.zeros((1, len(i)+1))
thetae1 = np.zeros((1, len(i)+1))
thetae1[:, 0] = thetar1[:, 0] - thetac1[:, 0]
thetae_hat1 = np.zeros((1, len(i)+1))

thetar2 = np.zeros((1, len(i)+1))
thetac2 = np.zeros((1, len(i)+1))
thetae2 = np.zeros((1, len(i)+1))
thetae2[:, 0] = thetar2[:, 0] - thetac2[:, 0]
thetae_hat2 = np.zeros((1, len(i)+1))

thetar3 = np.zeros((1, len(i)+1))
thetac3 = np.zeros((1, len(i)+1))
thetae3 = np.zeros((1, len(i)+1))
thetae3[:, 0] = thetar3[:, 0] - thetac3[:, 0]
thetae_hat3 = np.zeros((1, len(i)+1))

thetar4 = np.zeros((1, len(i)+1))
thetac4 = np.zeros((1, len(i)+1))
thetae4 = np.zeros((1, len(i)+1))
thetae4[:, 0] = thetar4[:, 0] - thetac4[:, 0]
thetae_hat4 = np.zeros((1, len(i)+1))

thetar5 = np.zeros((1, len(i)+1))
thetac5 = np.zeros((1, len(i)+1))
thetae5 = np.zeros((1, len(i)+1))
thetae5[:, 0] = thetar5[:, 0] - thetac5[:, 0]
thetae_hat5 = np.zeros((1, len(i)+1))


# 初始化虚拟领导者角速度和线速度
w0 = np.zeros((1, len(i)+1))
w0[:, 0] = 0
v0 = np.zeros((1, len(i)+1))
v0[:, 0] = 0

# 初始化速度矩阵
vr1 = np.zeros((1, len(i)+1))
vc1 = np.zeros((1, len(i)+1))
ve1 = np.zeros((1, len(i)+1))
wr1 = np.zeros((1, len(i)+1))
wc1 = np.zeros((1, len(i)+1))
we1 = np.zeros((1, len(i)+1))

vr2 = np.zeros((1, len(i)+1))
vc2 = np.zeros((1, len(i)+1))
ve2 = np.zeros((1, len(i)+1))
wr2 = np.zeros((1, len(i)+1))
wc2 = np.zeros((1, len(i)+1))
we2 = np.zeros((1, len(i)+1))

vr3 = np.zeros((1, len(i)+1))
vc3 = np.zeros((1, len(i)+1))
ve3 = np.zeros((1, len(i)+1))
wr3 = np.zeros((1, len(i)+1))
wc3 = np.zeros((1, len(i)+1))
we3 = np.zeros((1, len(i)+1))

vr4 = np.zeros((1, len(i)+1))
vc4 = np.zeros((1, len(i)+1))
ve4 = np.zeros((1, len(i)+1))
wr4 = np.zeros((1, len(i)+1))
wc4 = np.zeros((1, len(i)+1))
we4 = np.zeros((1, len(i)+1))

vr5 = np.zeros((1, len(i)+1))
vc5 = np.zeros((1, len(i)+1))
ve5 = np.zeros((1, len(i)+1))
wr5 = np.zeros((1, len(i)+1))
wc5 = np.zeros((1, len(i)+1))
we5 = np.zeros((1, len(i)+1))


# 初始化控制输入
uz1 = np.zeros((2, len(i)+1))

uz2 = np.zeros((2, len(i)+1))

uz3 = np.zeros((2, len(i)+1))

uz4 = np.zeros((2, len(i)+1))

uz5 = np.zeros((2, len(i)+1))


# 初始化质心位置，即编队几何中心
xccc = np.zeros((1, len(i)+1))
xccc[0, 0] = np.sum(xc1[:, 0]+xc2[:, 0]+xc3[:, 0]+xc4[:, 0]+xc5[:, 0])/Rnum
yccc = np.zeros((1, len(i)+1))
yccc[0, 0] = np.sum(yc1[:, 0]+yc2[:, 0]+yc3[:, 0]+yc4[:, 0]+yc5[:, 0])/Rnum

# 初始化虚拟领导者位置
x0 = np.zeros((1, len(i)+1))
x0[0, 0] = 0.0
y0 = np.zeros((1, len(i)+1))
y0[0, 0] = 0.3
theta0 = np.zeros((1, len(i)+1))
theta0[0, 0] = 0.0

z0 = np.zeros((2, len(i)+1))

# 初始化误差矩阵
ex1 = np.zeros((2, len(i)))
fx1  = np.zeros((2, len(i)))
zx1 = np.zeros((2, len(i)+1))

ex2 = np.zeros((2, len(i)))
fx2  = np.zeros((2, len(i)))
zx2 = np.zeros((2, len(i)+1))

ex3 = np.zeros((2, len(i)))
fx3  = np.zeros((2, len(i)))
zx3 = np.zeros((2, len(i)+1))

ex4 = np.zeros((2, len(i)))
fx4  = np.zeros((2, len(i)))
zx4 = np.zeros((2, len(i)+1))

ex5 = np.zeros((2, len(i)))
fx5  = np.zeros((2, len(i)))
zx5 = np.zeros((2, len(i)+1))

# 初始化中间变量
L1_1=0
L2_1=0
L3_1=0
M0_1=0


L1_2=0
L2_2=0
L3_2=0
M0_2=0


L1_3=0
L2_3=0
L3_3=0
M0_3=0


L1_4=0
L2_4=0
L3_4=0
M0_4=0


L1_5=0
L2_5=0
L3_5=0
M0_5=0

# 时间序列
t = np.arange(len(i) + 1) * t1  # 与状态数组严格等长，避免短时长浮点端点多出一项

# 初始化位置偏移
px1 = np.zeros((1,  len(i)+1))
py1 = np.zeros((1,  len(i)+1))

px2 = np.zeros((1,  len(i)+1))
py2 = np.zeros((1,  len(i)+1))

px3 = np.zeros((1,  len(i)+1))
py3 = np.zeros((1,  len(i)+1))

px4 = np.zeros((1,  len(i)+1))
py4 = np.zeros((1,  len(i)+1))

px5 = np.zeros((1,  len(i)+1))
py5 = np.zeros((1,  len(i)+1))

# 初始化时间变量
t2 = 0

# 领导者分布式观测器相关初始化
# 定义领导者的观测器估计，这里的zxeo_i代表第i个机器人（agent）对leader在时间k时刻的状态估计,dzxeo_i是这些状态估计的离散时间导数
# 观测器估计变量（每个机器人一组）
zxeo_1 = np.zeros((2, len(i)+1))
dzxeo_1 = np.zeros((2, len(i)+1))

zxeo_2 = np.zeros((2, len(i)+1))
dzxeo_2 = np.zeros((2, len(i)+1))

zxeo_3 = np.zeros((2, len(i)+1))
dzxeo_3 = np.zeros((2, len(i)+1))

zxeo_4 = np.zeros((2, len(i)+1))
dzxeo_4 = np.zeros((2, len(i)+1))

zxeo_5 = np.zeros((2, len(i)+1))
dzxeo_5 = np.zeros((2, len(i)+1))

# 分布式观测器参数
ob_eta = 0.7
ob_alp = 0.8

# 🚀 Python语法改造：将单独变量组织成列表，便于避障时机器人之间距离计算索引访问
xr = [xr1, xr2, xr3, xr4, xr5]  # 索引0-4对应xr1-xr5
yr = [yr1, yr2, yr3, yr4, yr5]  # 索引0-4对应yr1-yr5

#障碍物设置
# 障碍物设置
xobs_org = np.zeros((1, 800))
yobs_org = np.zeros((1, 800))

# 设置圆形障碍物1
r_of_obs1 = 0.3
p_obs1 = np.array([1.7, 0.0]).reshape(2, 1)

for ko in range(200):
    xobs_org[0, ko] = p_obs1[0, 0] + r_of_obs1 * np.cos(ko * 2 * np.pi / 200)
    yobs_org[0, ko] = p_obs1[1, 0] + r_of_obs1 * np.sin(ko * 2 * np.pi / 200)

xyobs_org = np.zeros((2, 800))
xyobs_org[:, :200] = np.vstack((xobs_org[:, :200], yobs_org[:, :200]))

# 设置圆形障碍物2（与结果图及 consensus_obstacle_1.world 一致）
r_of_obs2_a = 0.4
r_of_obs2_b = 0.4
p_obs2 = np.array([-3.0, 2.0]).reshape(2, 1)
rot_1 = np.pi / 4

for ko in range(200, 400):
    xobs_org[0, ko] = r_of_obs2_a * np.cos((ko - 1) * 2 * np.pi / 200)
    yobs_org[0, ko] = r_of_obs2_b * np.sin((ko - 1) * 2 * np.pi / 200)

# 旋转并平移
pts2 = np.vstack((xobs_org[:, 200:400], yobs_org[:, 200:400]))  # shape (2,200) → stack to (4,200)?
# 注意：MATLAB 原来是 [x; y] 变成 2×200，再右乘 2×2 旋转矩阵，这里：
pts2_2d = pts2.reshape(2, 200)    # 重新组合成 (2,200)
R1 = np.array([[ np.cos(rot_1),  np.sin(rot_1)],
               [-np.sin(rot_1),  np.cos(rot_1)]])
xyobs_rot2 = R1 @ pts2_2d + p_obs2 @ np.ones((1, 200))
xyobs_org[:, 200:400] = xyobs_rot2

# 设置矩形障碍物3  
la, lb = 1.0, 0.8
p_obs3 = np.array([1.0, -1.5]).reshape(2, 1)
rot_3 = -np.pi / 8  # 下方 R3 使用相反符号，实际逆时针旋转 22.5 度

# 先在 xobs_org/yobs_org 中填 400~600 列
for ko in range(1, 51):
    idx1 = 400 + ko - 1
    idx2 = 400 + ko + 100 - 1
    xobs_org[0, idx1] = -la/2 + (ko-1) * la/50
    yobs_org[0, idx1] = -lb/2
    xobs_org[0, idx2] = la/2 - (ko-1) * la/50
    yobs_org[0, idx2] = lb/2

for ko in range(51, 101):
    idx1 = 400 + ko - 1
    idx2 = 400 + ko + 100 - 1
    xobs_org[0, idx1] = la/2
    yobs_org[0, idx1] = -lb/2 + (ko-51) * lb/50
    xobs_org[0, idx2] = -la/2
    yobs_org[0, idx2] = lb/2 - (ko-51) * lb/50

# 旋转并平移
pts3 = np.vstack((xobs_org[:, 400:600], yobs_org[:, 400:600]))
pts3_2d = pts3.reshape(2, 200)
R3 = np.array([[ np.cos(rot_3),  np.sin(rot_3)],
               [-np.sin(rot_3),  np.cos(rot_3)]])
xyobs_rot3 = R3 @ pts3_2d + p_obs3 @ np.ones((1, 200))
xyobs_org[:, 400:600] = xyobs_rot3

# 设置圆形障碍物4
r_of_obs4 = 0.15
p_obs4 = np.array([0.5, 2.6]).reshape(2, 1)

for ko in range(600, 800):
    xobs_org[0, ko] = p_obs4[0, 0] + r_of_obs4 * np.cos((ko - 1) * 2 * np.pi / 200)
    yobs_org[0, ko] = p_obs4[1, 0] + r_of_obs4 * np.sin((ko - 1) * 2 * np.pi / 200)

xyobs_org[:, 600:800] = np.vstack((xobs_org[:, 600:800], yobs_org[:, 600:800]))

xyobs_rot = xyobs_org.copy()

n_o = xyobs_org.shape[1]  # 障碍物总数800

xobs = xyobs_rot[0, 0:n_o]#所有障碍物的x坐标
yobs = xyobs_rot[1, 0:n_o]#所有障碍物的y坐标

# 对每组完整边界点仅拟合一次最小包络椭圆，五个机器人共享。
# 不估计/预测障碍物运动；仅保留参考项目的协方差递推与半轴膨胀。
static_obstacle_model = SharedStaticObstacleModel(
    [xyobs_org[:, start:start+200].T for start in range(0, 800, 200)],
    ["circle_1", "circle_2", "rectangle_3", "circle_4"],
    FULL_MPC_HORIZON_STEPS, t1,
)
dcbf_obstacles = static_obstacle_model.advance()
dcbf_obstacle_envelope_history = []

# 初始化机器人与障碍物距离矩阵
dobsmin_1 = np.zeros((1, len(i)))

dobsmin_2 = np.zeros((1, len(i)))

dobsmin_3 = np.zeros((1, len(i)))

dobsmin_4 = np.zeros((1, len(i)))

dobsmin_5 = np.zeros((1, len(i)))

ord_avoid = 0.35 #用于计算避障时的距离参数
ord_safe = 0.3 #判断是否为危险障碍物的距离阈值

ord_avoid_robot = 0.35 #用于计算避障时的距离参数
ord_safe_robot = 0.3 #判断是否为危险障碍物的距离阈值

dobs_1 = np.zeros((1, n_o+1)) # 机器人参考轨迹与障碍物距离

dobs_2 = np.zeros((1, n_o+1)) # 机器人参考轨迹与障碍物距离

dobs_3 = np.zeros((1, n_o+1)) # 机器人参考轨迹与障碍物距离

dobs_4 = np.zeros((1, n_o+1)) # 机器人参考轨迹与障碍物距离

dobs_5 = np.zeros((1, n_o+1)) # 机器人参考轨迹与障碍物距离

dij_avoid = 0.35 #用于计算机器人间避障时的距离参数

min_obs_num_1 = np.zeros((1,), dtype=int) #每个参考位置机器人最近的障碍物编号

min_obs_num_2 = np.zeros((1,), dtype=int) #每个参考位置机器人最近的障碍物编号

min_obs_num_3 = np.zeros((1,), dtype=int) #每个参考位置机器人最近的障碍物编号

min_obs_num_4 = np.zeros((1,), dtype=int) #每个参考位置机器人最近的障碍物编号

min_obs_num_5 = np.zeros((1,), dtype=int) #每个参考位置机器人最近的障碍物编号


pxy_avoid_1 = np.zeros((2, 1)) #避障参考轨迹轨迹偏移向量

pxy_avoid_2 = np.zeros((2, 1)) #避障参考轨迹轨迹偏移向量

pxy_avoid_3 = np.zeros((2, 1)) #避障参考轨迹轨迹偏移向量

pxy_avoid_4 = np.zeros((2, 1)) #避障参考轨迹轨迹偏移向量

pxy_avoid_5 = np.zeros((2, 1)) #避障参考轨迹轨迹偏移向量

# 避障力大小历史记录数组
avoidance_force_mag_1 = np.zeros(len(i))  # 机器人1避障力大小历史
avoidance_force_mag_2 = np.zeros(len(i))  # 机器人2避障力大小历史
avoidance_force_mag_3 = np.zeros(len(i))  # 机器人3避障力大小历史
avoidance_force_mag_4 = np.zeros(len(i))  # 机器人4避障力大小历史
avoidance_force_mag_5 = np.zeros(len(i))  # 机器人5避障力大小历史

# 静态障碍物避障力历史记录数组（分离记录）
static_avoidance_force_mag_1 = np.zeros(len(i))
static_avoidance_force_mag_2 = np.zeros(len(i))
static_avoidance_force_mag_3 = np.zeros(len(i))
static_avoidance_force_mag_4 = np.zeros(len(i))
static_avoidance_force_mag_5 = np.zeros(len(i))

# 机器人间避障力历史记录数组（分离记录）
robot_avoidance_force_mag_1 = np.zeros(len(i))
robot_avoidance_force_mag_2 = np.zeros(len(i))
robot_avoidance_force_mag_3 = np.zeros(len(i))
robot_avoidance_force_mag_4 = np.zeros(len(i))
robot_avoidance_force_mag_5 = np.zeros(len(i))

dcbf_mpc_active_obstacle_history = np.zeros((Rnum, len(i)), dtype=int)
dcbf_mpc_success_history = np.zeros((Rnum, len(i)), dtype=int)
dcbf_mpc_min_h_history = np.full((Rnum, len(i)), np.nan)
dcbf_mpc_last_solutions = [None] * Rnum
filtered_static_avoidance = np.zeros((Rnum, 2))

d_R2R_1    = np.zeros((1, Rnum)) #机器人与所有机器人的距离矩阵

d_R2R_2    = np.zeros((1, Rnum)) #机器人与所有机器人的距离矩阵

d_R2R_3    = np.zeros((1, Rnum)) #机器人与所有机器人的距离矩阵

d_R2R_4    = np.zeros((1, Rnum)) #机器人与所有机器人的距离矩阵

d_R2R_5    = np.zeros((1, Rnum)) #机器人与所有机器人的距离矩阵


# 主控制循环
for k in range(len(i)):
    # 每个采样时刻更新一次，不能在五次机器人求解中分别推进协方差。
    if k > 0:
        dcbf_obstacles = static_obstacle_model.advance()
    dcbf_obstacle_envelope_history.append(dcbf_obstacles)

    px1[:, k+1] = bb * 0.4 * np.cos(aa * t2 + (0 * 2 * np.pi / Rnum))
    py1[:, k+1] = bb * 0.4 * np.sin(aa * t2 + (0 * 2 * np.pi / Rnum)) 

    px2[:, k+1] = bb * 0.4 * np.cos(aa * t2 + (1 * 2 * np.pi / Rnum))
    py2[:, k+1] = bb * 0.4 * np.sin(aa * t2 + (1 * 2 * np.pi / Rnum)) 

    px3[:, k+1] = bb * 0.4 * np.cos(aa * t2 + (2 * 2 * np.pi / Rnum))
    py3[:, k+1] = bb * 0.4 * np.sin(aa * t2 + (2 * 2 * np.pi / Rnum)) 

    px4[:, k+1] = bb * 0.4 * np.cos(aa * t2 + (3 * 2 * np.pi / Rnum))
    py4[:, k+1] = bb * 0.4 * np.sin(aa * t2 + (3 * 2 * np.pi / Rnum)) 

    px5[:, k+1] = bb * 0.4 * np.cos(aa * t2 + (4 * 2 * np.pi / Rnum))
    py5[:, k+1] = bb * 0.4 * np.sin(aa * t2 + (4 * 2 * np.pi / Rnum)) 
    

    #机器人1避障
    dobsmin_1[0, k] = 10000
    min_obs_num_1 = 0
    pxy_avoid_1[:, 0] = 0  # 重置避障力向量
    pxy_static_1 = np.zeros(2)  # 静态障碍物避障力
    pxy_robot_1 = np.zeros(2)   # 机器人间避障力
    
    for jj in range(n_o):
        dobs_1[0, jj] = np.sqrt((xobs[jj] - xr1[0, k])**2 + 
                                (yobs[jj] - yr1[0, k])**2)

        if dobs_1[0, jj] <= dobsmin_1[0, k]:
            min_obs_num_1 = jj
            dobsmin_1[0, k] = dobs_1[0, jj]
    # 找出最近的障碍物


    # 避障控制
    if dobsmin_1[0, k] <= ord_safe:
        jj_min = min_obs_num_1
        angle_o2r = np.arctan2(
            yobs[jj_min] - yr1[0, k],
            xobs[jj_min] - xr1[0, k]
        )
        force = (ord_avoid / dobs_1[0, jj_min]) * (ord_avoid - dobs_1[0, jj_min])
        pxy_static_1 = force * np.array([np.cos(angle_o2r), np.sin(angle_o2r)])
        pxy_avoid_1[:, 0] = pxy_static_1
    else:
        pxy_avoid_1[:, 0] = 0
    

    # if 460 <= k <= 510:
    #     print(f"下一步期望轨迹的最近障碍物全局坐标为:{xobs[jj_min]}, {yobs[jj_min]}，与该点距离: {dobsmin_1[0, k]}")
        
    #     print(f"角度: {angle_o2r}")
    #     print(f"力: {force}")
    #     print(f"避障偏移向量: {pxy_avoid_1[:, 0]}")

    # 🚀 定义当前机器人编号（这里是机器人1，索引为0）
    ii = 0  # 机器人1对应索引0
    
    for jj in range(Rnum):
        if jj != ii:
            # 🚀 Python语法改造：使用列表索引访问对应的xr和yr变量
            # xr[0]=xr1, xr[1]=xr2, xr[2]=xr3, xr[3]=xr4, xr[4]=xr5
            # yr[0]=yr1, yr[1]=yr2, yr[2]=yr3, yr[3]=yr4, yr[4]=yr5
            d_R2R_1[0, jj] = np.sqrt((xr[jj][0, k] - xr[ii][0, k])**2 + 
                            (yr[jj][0, k] - yr[ii][0, k])**2)

            if d_R2R_1[0, jj] <= 0.3:
                angle_r2i = np.arctan2(yr[jj][0, k] - yr[ii][0, k],
                                        xr[jj][0, k] - xr[ii][0, k])
                force_rr = (dij_avoid / d_R2R_1[0, jj]) * (dij_avoid - d_R2R_1[0, jj])
                robot_force = force_rr * np.array([np.cos(angle_r2i), np.sin(angle_r2i)])
                pxy_robot_1 += robot_force
                pxy_avoid_1[:, 0] += robot_force
    # px1[:, k+1] -= pxy_avoid_1[0, 0]
    # py1[:, k+1] -= pxy_avoid_1[1, 0]
    
    # 记录机器人1避障力大小（总避障力、静态障碍物避障力、机器人间避障力）
    avoidance_force_mag_1[k] = np.linalg.norm(pxy_avoid_1[:, 0])
    static_avoidance_force_mag_1[k] = np.linalg.norm(pxy_static_1)
    robot_avoidance_force_mag_1[k] = np.linalg.norm(pxy_robot_1)


    #机器人2避障
    dobsmin_2[0, k] = 10000
    min_obs_num_2 = 0
    pxy_avoid_2[:, 0] = 0  # 重置避障力向量
    pxy_static_2 = np.zeros(2)  # 静态障碍物避障力
    pxy_robot_2 = np.zeros(2)   # 机器人间避障力

    for jj in range(n_o):
        dobs_2[0, jj] = np.sqrt((xobs[jj] - xr2[0, k])**2 + 
                                (yobs[jj] - yr2[0, k])**2)

        if dobs_2[0, jj] <= dobsmin_2[0, k]:
            min_obs_num_2 = jj
            dobsmin_2[0, k] = dobs_2[0, jj]

    # 避障控制
    if dobsmin_2[0, k] <= ord_safe:
        jj_min = min_obs_num_2
        angle_o2r = np.arctan2(
            yobs[jj_min] - yr2[0, k],
            xobs[jj_min] - xr2[0, k]
        )
        force = (ord_avoid / dobs_2[0, jj_min]) * (ord_avoid - dobs_2[0, jj_min])
        pxy_static_2 = force * np.array([np.cos(angle_o2r), np.sin(angle_o2r)])
        pxy_avoid_2[:, 0] = pxy_static_2
    else:
        pxy_avoid_2[:, 0] = 0

    # 🚀 定义当前机器人编号（这里是机器人2，索引为1）
    ii = 1  # 机器人2对应索引1
    for jj in range(Rnum):
        if jj != ii:
            # 🚀 Python语法改造：使用列表索引访问对应的xr和yr变量
            # xr[0]=xr1, xr[1]=xr2, xr[2]=xr3, xr[3]=xr4, xr[4]=xr5
            # yr[0]=yr1, yr[1]=yr2, yr[2]=yr3, yr[3]=yr4, yr[4]=yr5
            d_R2R_2[0, jj] = np.sqrt((xr[jj][0, k] - xr[ii][0, k])**2 + 
                            (yr[jj][0, k] - yr[ii][0, k])**2)

            if d_R2R_2[0, jj] <= 0.3:
                angle_r2i = np.arctan2(yr[jj][0, k] - yr[ii][0, k],
                                        xr[jj][0, k] - xr[ii][0, k])
                force_rr = (dij_avoid / d_R2R_2[0, jj]) * (dij_avoid - d_R2R_2[0, jj])
                robot_force = force_rr * np.array([np.cos(angle_r2i), np.sin(angle_r2i)])
                pxy_robot_2 += robot_force
                pxy_avoid_2[:, 0] += robot_force
    # px2[:, k+1] -= pxy_avoid_2[0, 0]
    # py2[:, k+1] -= pxy_avoid_2[1, 0]
    
    # 记录机器人2避障力大小
    avoidance_force_mag_2[k] = np.linalg.norm(pxy_avoid_2[:, 0])
    static_avoidance_force_mag_2[k] = np.linalg.norm(pxy_static_2)
    robot_avoidance_force_mag_2[k] = np.linalg.norm(pxy_robot_2)
    
    #机器人3避障
    dobsmin_3[0, k] = 10000
    min_obs_num_3 = 0
    pxy_avoid_3[:, 0] = 0  # 重置避障力向量
    pxy_static_3 = np.zeros(2)
    pxy_robot_3 = np.zeros(2)

    for jj in range(n_o):
        dobs_3[0, jj] = np.sqrt((xobs[jj] - xr3[0, k])**2 + 
                                (yobs[jj] - yr3[0, k])**2)

        if dobs_3[0, jj] <= dobsmin_3[0, k]:
            min_obs_num_3 = jj
            dobsmin_3[0, k] = dobs_3[0, jj]

    # 避障控制
    if dobsmin_3[0, k] <= ord_safe:
        jj_min = min_obs_num_3
        angle_o2r = np.arctan2(
            yobs[jj_min] - yr3[0, k],
            xobs[jj_min] - xr3[0, k]
        )
        force = (ord_avoid / dobs_3[0, jj_min]) * (ord_avoid - dobs_3[0, jj_min])
        pxy_static_3 = force * np.array([np.cos(angle_o2r), np.sin(angle_o2r)])
        pxy_avoid_3[:, 0] = pxy_static_3
    else:
        pxy_avoid_3[:, 0] = 0

    # 🚀 定义当前机器人编号（这里是机器人3，索引为2）
    ii = 2  # 机器人3对应索引2
    for jj in range(Rnum):
        if jj != ii:
            # 🚀 Python语法改造：使用列表索引访问对应的xr和yr变量
            # xr[0]=xr1, xr[1]=xr2, xr[2]=xr3, xr[3]=xr4, xr[4]=xr5
            # yr[0]=yr1, yr[1]=yr2, yr[2]=yr3, yr[3]=yr4, yr[4]=yr5
            d_R2R_3[0, jj] = np.sqrt((xr[jj][0, k] - xr[ii][0, k])**2 + 
                            (yr[jj][0, k] - yr[ii][0, k])**2)

            if d_R2R_3[0, jj] <= 0.3:
                angle_r2i = np.arctan2(yr[jj][0, k] - yr[ii][0, k],
                                        xr[jj][0, k] - xr[ii][0, k])
                force_rr = (dij_avoid / d_R2R_3[0, jj]) * (dij_avoid - d_R2R_3[0, jj])
                robot_force = force_rr * np.array([np.cos(angle_r2i), np.sin(angle_r2i)])
                pxy_robot_3 += robot_force
                pxy_avoid_3[:, 0] += robot_force
    # px3[:, k+1] -= pxy_avoid_3[0, 0]
    # py3[:, k+1] -= pxy_avoid_3[1, 0]
    
    # 记录机器人3避障力大小
    avoidance_force_mag_3[k] = np.linalg.norm(pxy_avoid_3[:, 0])
    static_avoidance_force_mag_3[k] = np.linalg.norm(pxy_static_3)
    robot_avoidance_force_mag_3[k] = np.linalg.norm(pxy_robot_3)

    #机器人4避障
    dobsmin_4[0, k] = 10000
    min_obs_num_4 = 0
    pxy_avoid_4[:, 0] = 0  # 重置避障力向量
    pxy_static_4 = np.zeros(2)
    pxy_robot_4 = np.zeros(2)

    for jj in range(n_o):
        dobs_4[0, jj] = np.sqrt((xobs[jj] - xr4[0, k])**2 + 
                                (yobs[jj] - yr4[0, k])**2)

        if dobs_4[0, jj] <= dobsmin_4[0, k]:
            min_obs_num_4 = jj
            dobsmin_4[0, k] = dobs_4[0, jj]

    # 避障控制
    if dobsmin_4[0, k] <= ord_safe:
        jj_min = min_obs_num_4
        angle_o2r = np.arctan2(
            yobs[jj_min] - yr4[0, k],
            xobs[jj_min] - xr4[0, k]
        )
        force = (ord_avoid / dobs_4[0, jj_min]) * (ord_avoid - dobs_4[0, jj_min])
        pxy_static_4 = force * np.array([np.cos(angle_o2r), np.sin(angle_o2r)])
        pxy_avoid_4[:, 0] = pxy_static_4
    else:
        pxy_avoid_4[:, 0] = 0

    # 🚀 定义当前机器人编号（这里是机器人4，索引为3）
    ii = 3  # 机器人4对应索引3
    for jj in range(Rnum):
        if jj != ii:
            # 🚀 Python语法改造：使用列表索引访问对应的xr和yr变量
            # xr[0]=xr1, xr[1]=xr2, xr[2]=xr3, xr[3]=xr4, xr[4]=xr5
            # yr[0]=yr1, yr[1]=yr2, yr[2]=yr3, yr[3]=yr4, yr[4]=yr5
            d_R2R_4[0, jj] = np.sqrt((xr[jj][0, k] - xr[ii][0, k])**2 + 
                            (yr[jj][0, k] - yr[ii][0, k])**2)

            if d_R2R_4[0, jj] <= 0.3:
                angle_r2i = np.arctan2(yr[jj][0, k] - yr[ii][0, k],
                                        xr[jj][0, k] - xr[ii][0, k])
                force_rr = (dij_avoid / d_R2R_4[0, jj]) * (dij_avoid - d_R2R_4[0, jj])
                robot_force = force_rr * np.array([np.cos(angle_r2i), np.sin(angle_r2i)])
                pxy_robot_4 += robot_force
                pxy_avoid_4[:, 0] += robot_force
    # px4[:, k+1] -= pxy_avoid_4[0, 0]
    # py4[:, k+1] -= pxy_avoid_4[1, 0]
    
    # 记录机器人4避障力大小
    avoidance_force_mag_4[k] = np.linalg.norm(pxy_avoid_4[:, 0])
    static_avoidance_force_mag_4[k] = np.linalg.norm(pxy_static_4)
    robot_avoidance_force_mag_4[k] = np.linalg.norm(pxy_robot_4)

    #机器人5避障
    dobsmin_5[0, k] = 10000
    min_obs_num_5 = 0
    pxy_avoid_5[:, 0] = 0  # 重置避障力向量
    pxy_static_5 = np.zeros(2)
    pxy_robot_5 = np.zeros(2)

    for jj in range(n_o):
        dobs_5[0, jj] = np.sqrt((xobs[jj] - xr5[0, k])**2 + 
                                (yobs[jj] - yr5[0, k])**2)

        if dobs_5[0, jj] <= dobsmin_5[0, k]:
            min_obs_num_5 = jj
            dobsmin_5[0, k] = dobs_5[0, jj]

    # 避障控制
    if dobsmin_5[0, k] <= ord_safe:
        jj_min = min_obs_num_5
        angle_o2r = np.arctan2(
            yobs[jj_min] - yr5[0, k],
            xobs[jj_min] - xr5[0, k]
        )
        force = (ord_avoid / dobs_5[0, jj_min]) * (ord_avoid - dobs_5[0, jj_min])
        pxy_static_5 = force * np.array([np.cos(angle_o2r), np.sin(angle_o2r)])
        pxy_avoid_5[:, 0] = pxy_static_5
    else:
        pxy_avoid_5[:, 0] = 0

    # 🚀 定义当前机器人编号（这里是机器人5，索引为4）
    ii = 4  # 机器人5对应索引4
    for jj in range(Rnum):
        if jj != ii:
            # 🚀 Python语法改造：使用列表索引访问对应的xr和yr变量
            # xr[0]=xr1, xr[1]=xr2, xr[2]=xr3, xr[3]=xr4, xr[4]=xr5
            # yr[0]=yr1, yr[1]=yr2, yr[2]=yr3, yr[3]=yr4, yr[4]=yr5
            d_R2R_5[0, jj] = np.sqrt((xr[jj][0, k] - xr[ii][0, k])**2 + 
                            (yr[jj][0, k] - yr[ii][0, k])**2)

            if d_R2R_5[0, jj] <= 0.3:
                angle_r2i = np.arctan2(yr[jj][0, k] - yr[ii][0, k],
                                        xr[jj][0, k] - xr[ii][0, k])
                force_rr = (dij_avoid / d_R2R_5[0, jj]) * (dij_avoid - d_R2R_5[0, jj])
                robot_force = force_rr * np.array([np.cos(angle_r2i), np.sin(angle_r2i)])
                pxy_robot_5 += robot_force
                pxy_avoid_5[:, 0] += robot_force
    # px5[:, k+1] -= pxy_avoid_5[0, 0]
    # py5[:, k+1] -= pxy_avoid_5[1, 0]
    
    # 记录机器人5避障力大小
    avoidance_force_mag_5[k] = np.linalg.norm(pxy_avoid_5[:, 0])
    static_avoidance_force_mag_5[k] = np.linalg.norm(pxy_static_5)
    robot_avoidance_force_mag_5[k] = np.linalg.norm(pxy_robot_5)

    # if k==0:
    #     print(dobsmin_1[0, k])
    #     print(d_R2R_1[0, :])
    #     print(dobsmin_2[0, k])
    #     print(d_R2R_2[0, :])
    #     print(dobsmin_3[0, k])
    #     print(d_R2R_3[0, :])
    #     print(dobsmin_4[0, k])
    #     print(d_R2R_4[0, :])
    #     print(dobsmin_5[0, k])
    #     print(d_R2R_5[0, :])

    # 更新参考轨迹（支持多种轨迹类型）
    if trajectory_type == "real_experiment_trajectory_line":
        # 实验直线轨迹 - 从(0, 0.3)到(3.5, 3.2)
        start_x, start_y = 0.0, 0.0
        end_x, end_y = 3.5, 3.5
        
        # 计算方向向量和距离
        dx = end_x - start_x  # 3.2
        dy = end_y - start_y  # 3.2
        total_distance = np.sqrt(dx**2 + dy**2)  # ≈ 4.52
        
        # 直线速度(m/s)
        linear_velocity = 0.15
        
        # 单位方向向量
        dir_x = dx / total_distance  # ≈ 0.707
        dir_y = dy / total_distance  # ≈ 0.707
        
        # 计算当前时刻k的位置
        traveled_distance_k = linear_velocity * t2
        if traveled_distance_k <= total_distance:
            x0[0, k] = start_x + dir_x * traveled_distance_k
            y0[0, k] = start_y + dir_y * traveled_distance_k
        else:
            # 到达终点后保持不动
            x0[0, k] = end_x
            y0[0, k] = end_y
        
        # 计算下一时刻k+1的位置
        traveled_distance_k1 = linear_velocity * (t2 + t1)
        if traveled_distance_k1 <= total_distance:
            x0[0, k+1] = start_x + dir_x * traveled_distance_k1
            y0[0, k+1] = start_y + dir_y * traveled_distance_k1
        else:
            x0[0, k+1] = end_x
            y0[0, k+1] = end_y
        
        # 通过位置差分计算速度
        u0x[0, k+1] = (x0[0, k+1] - x0[0, k])/t1
        u0y[0, k+1] = (y0[0, k+1] - y0[0, k])/t1
        
        # 更新控制输入和角度
        u0[:, k+1] = np.array([u0x[0, k+1], u0y[0, k+1]])
        theta0[0, k+1] = np.arctan2(u0y[0, k+1], u0x[0, k+1])
        
        # 角度wrap处理
        theta0_sub = 0
        flagplne0 = 0
        
        if theta0[0, k] >= np.pi/2 and theta0[0, k] <= np.pi and \
           theta0[0, k+1] <= -np.pi/2 and theta0[0, k+1] >= -np.pi:
            flagplne0 = -1
        elif theta0[0, k] <= -np.pi/2 and theta0[0, k] >= -np.pi and \
             theta0[0, k+1] >= np.pi/2 and theta0[0, k+1] <= np.pi:
            flagplne0 = 1
        
        theta0_sub = theta0[0, k] + 2 * np.pi * flagplne0
        
        # 通过角度差分计算角速度和线速度
        w0[0, k+1] = (theta0[0, k+1] - theta0_sub)/t1
        v0[0, k+1] = np.sqrt((x0[0, k+1] - x0[0, k])**2 + 
                            (y0[0, k+1] - y0[0, k])**2)/t1
        
    elif trajectory_type == "real_experiment_trajectory":
        # 实验轨迹 - 平均速度0.15，半圆正弦变化
        leader_x0 = 0.0   # 固定起点x坐标
        leader_y0 = 0.3   # 固定起点y坐标
        radius = 0.75  # 半圆半径0.75m
        
        # 速度设计：v = 0.15 + A*sin(2πp)
        # 边界v=0.15，平均v=0.15
        # 顶部半圆：A=-0.03，最小v=0.12
        # 底部半圆：A=0.03，最大v=0.18
        
        # 各段距离
        dist1 = 2.55 - 0.3  # 从起点(0, 0.3)到顶部半圆左端(0, 2.55)
        arc_length = np.pi * radius  # 半圆
        
        # 第三段：垂直线，从(1.5, 2.55)到(1.5, 1.25)
        dist3 = 2.55 - 1.25  # 1.3m垂直距离
        dist5 = 3.3 - 1.25  # 从底部半圆右端(3.0, 1.25)到终点(3.0, 3.3)
        
        v_avg = 0.15
        t1_duration = dist1 / v_avg
        t2_duration = arc_length / v_avg
        t3_duration = dist3 / v_avg
        t4_duration = arc_length / v_avg
        t5_duration = dist5 / v_avg
        
        T1 = t1_duration
        T2 = T1 + t2_duration
        T3 = T2 + t3_duration
        T4 = T3 + t4_duration
        T5 = T4 + t5_duration
        
        # 计算当前时刻k的位置
        if t2 <= T1:
            # 段1：垂直上升，从(0.0, 0.3)开始
            progress = t2 / T1
            x0[0, k] = leader_x0
            y0[0, k] = leader_y0 + dist1 * progress
        elif t2 <= T2:
            # 段2：顶部半圆，圆心(0.75, 2.55)，v = 0.15 - 0.03*sin(2πp)
            t_seg = t2 - T1
            duration = T2 - T1
            progress = t_seg / duration
            
            s = 0.15 * t_seg + 0.03 * duration * (np.cos(2 * np.pi * progress) - 1) / (2 * np.pi)
            angle = np.pi - s / radius
            x0[0, k] = 0.75 + radius * np.cos(angle)
            y0[0, k] = 2.55 + radius * np.sin(angle)
        elif t2 <= T3:
            # 段3：垂直线，从顶部半圆右端(1.5, 2.55)到底部半圆左端(1.5, 1.25)
            progress = (t2 - T2) / (T3 - T2)
            x0[0, k] = 1.5
            y0[0, k] = 2.55 - 1.3 * progress
        elif t2 <= T4:
            # 段4：底部半圆，圆心(2.25, 1.25)，v = 0.15 + 0.03*sin(2πp)
            t_seg = t2 - T3
            duration = T4 - T3
            progress = t_seg / duration
            
            s = 0.15 * t_seg - 0.03 * duration * (np.cos(2 * np.pi * progress) - 1) / (2 * np.pi)
            angle = np.pi + s / radius
            x0[0, k] = 2.25 + radius * np.cos(angle)
            y0[0, k] = 1.25 + radius * np.sin(angle)
        else:
            # 段5：垂直上升，从底部半圆右端(3.0, 1.25)到终点(3.0, 3.3)
            progress = (t2 - T4) / (T5 - T4)
            x0[0, k] = 3.0
            y0[0, k] = 1.25 + dist5 * progress
        
        # 计算下一时刻k+1的位置
        calc_time = t2 + t1
        if calc_time <= T1:
            progress = calc_time / T1
            x0[0, k+1] = leader_x0
            y0[0, k+1] = leader_y0 + dist1 * progress
        elif calc_time <= T2:
            t_seg = calc_time - T1
            duration = T2 - T1
            progress = t_seg / duration
            
            s = 0.15 * t_seg + 0.03 * duration * (np.cos(2 * np.pi * progress) - 1) / (2 * np.pi)
            angle = np.pi - s / radius
            x0[0, k+1] = 0.75 + radius * np.cos(angle)
            y0[0, k+1] = 2.55 + radius * np.sin(angle)
        elif calc_time <= T3:
            progress = (calc_time - T2) / (T3 - T2)
            x0[0, k+1] = 1.5
            y0[0, k+1] = 2.55 - 1.3 * progress
        elif calc_time <= T4:
            t_seg = calc_time - T3
            duration = T4 - T3
            progress = t_seg / duration
            
            s = 0.15 * t_seg - 0.03 * duration * (np.cos(2 * np.pi * progress) - 1) / (2 * np.pi)
            angle = np.pi + s / radius
            x0[0, k+1] = 2.25 + radius * np.cos(angle)
            y0[0, k+1] = 1.25 + radius * np.sin(angle)
        else:
            progress = (calc_time - T4) / (T5 - T4)
            x0[0, k+1] = 3.0
            y0[0, k+1] = 1.25 + dist5 * progress
        
        # 通过位置差分计算速度（与8字形完全一致）
        u0x[0, k+1] = (x0[0, k+1] - x0[0, k])/t1
        u0y[0, k+1] = (y0[0, k+1] - y0[0, k])/t1
        
        # 更新控制输入和角度
        u0[:, k+1] = np.array([u0x[0, k+1], u0y[0, k+1]])
        theta0[0, k+1] = np.arctan2(u0y[0, k+1], u0x[0, k+1])
        
        # 角度wrap处理（与8字形完全一致）
        theta0_sub = 0
        flagplne0 = 0
        
        if theta0[0, k] >= np.pi/2 and theta0[0, k] <= np.pi and \
           theta0[0, k+1] <= -np.pi/2 and theta0[0, k+1] >= -np.pi:
            flagplne0 = -1
        elif theta0[0, k] <= -np.pi/2 and theta0[0, k] >= -np.pi and \
             theta0[0, k+1] >= np.pi/2 and theta0[0, k+1] <= np.pi:
            flagplne0 = 1
        
        theta0_sub = theta0[0, k] + 2 * np.pi * flagplne0
        
        # 通过角度差分计算角速度和线速度（与8字形完全一致）
        w0[0, k+1] = (theta0[0, k+1] - theta0_sub)/t1
        v0[0, k+1] = np.sqrt((x0[0, k+1] - x0[0, k])**2 + 
                            (y0[0, k+1] - y0[0, k])**2)/t1
        
    elif trajectory_type == "circle_trajectory":
        # 圆形轨迹：先从起点移动到圆轨迹起点，然后沿圆运动
        circle_center_x = 0.0  # 圆心x坐标
        circle_center_y = 0.0  # 圆心y坐标
        radius = 0.75          # 半径
        omega = 0.15           # 角速度 rad/s
        initial_angle = np.pi  # 初始角度（π表示从圆心正西侧开始）
        
        # 计算圆轨迹起点
        start_x = circle_center_x + radius * np.cos(initial_angle)  # -0.75
        start_y = circle_center_y + radius * np.sin(initial_angle)  # 0.0
        
        # 从初始位置(0,0)到圆轨迹起点的距离
        transition_dist = np.sqrt((start_x - x0[0,0])**2 + (start_y - y0[0,0])**2)
        v_transition = 0.15  # 过渡段速度
        transition_time = transition_dist / v_transition
        
        if t2 <= transition_time:
            # 过渡段：从初始位置匀速移动到圆轨迹起点
            progress = t2 / transition_time
            x0[0, k] = x0[0, 0] + (start_x - x0[0, 0]) * progress
            y0[0, k] = y0[0, 0] + (start_y - y0[0, 0]) * progress
            
            progress_next = min((t2 + t1) / transition_time, 1.0)
            x0[0, k+1] = x0[0, 0] + (start_x - x0[0, 0]) * progress_next
            y0[0, k+1] = y0[0, 0] + (start_y - y0[0, 0]) * progress_next
        else:
            # 圆轨迹段：从过渡时间后开始计算
            t_circle = t2 - transition_time
            angle_k = omega * t_circle + initial_angle
            x0[0, k] = circle_center_x + radius * np.cos(angle_k)
            y0[0, k] = circle_center_y + radius * np.sin(angle_k)
            
            t_circle_next = t2 + t1 - transition_time
            angle_k1 = omega * t_circle_next + initial_angle
            x0[0, k+1] = circle_center_x + radius * np.cos(angle_k1)
            y0[0, k+1] = circle_center_y + radius * np.sin(angle_k1)
        
        # 通过位置差分计算速度
        u0x[0, k+1] = (x0[0, k+1] - x0[0, k])/t1
        u0y[0, k+1] = (y0[0, k+1] - y0[0, k])/t1
        
        # 更新控制输入和角度
        u0[:, k+1] = np.array([u0x[0, k+1], u0y[0, k+1]])
        theta0[0, k+1] = np.arctan2(u0y[0, k+1], u0x[0, k+1])
        
        # 角度wrap处理
        theta0_sub = 0
        flagplne0 = 0
        
        if theta0[0, k] >= np.pi/2 and theta0[0, k] <= np.pi and \
           theta0[0, k+1] <= -np.pi/2 and theta0[0, k+1] >= -np.pi:
            flagplne0 = -1
        elif theta0[0, k] <= -np.pi/2 and theta0[0, k] >= -np.pi and \
             theta0[0, k+1] >= np.pi/2 and theta0[0, k+1] <= np.pi:
            flagplne0 = 1
        
        theta0_sub = theta0[0, k] + 2 * np.pi * flagplne0
        
        # 计算角速度和线速度
        w0[0, k+1] = (theta0[0, k+1] - theta0_sub)/t1
        v0[0, k+1] = np.sqrt((x0[0, k+1] - x0[0, k])**2 + 
                            (y0[0, k+1] - y0[0, k])**2)/t1
        
    else:
        # 默认8字形轨迹
        x0[0, k] = x0[0, 0] + 2.5 * np.sin(t2/15)
        y0[0, k] = y0[0, 0] + 2.5 * np.sin(t2/30)
        
        x0[0, k+1] = x0[0, 0] + 2.5 * np.sin((t2+t1)/15)
        y0[0, k+1] = y0[0, 0] + 2.5 * np.sin((t2+t1)/30)

        # 计算参考速度
        u0x[0, k+1] = (x0[0, k+1] - x0[0, k])/t1
        u0y[0, k+1] = (y0[0, k+1] - y0[0, k])/t1
        
        # 更新控制输入和角度（8字形轨迹）
        u0[:, k+1] = np.array([u0x[0, k+1], u0y[0, k+1]])
        theta0[0, k+1] = np.arctan2(u0y[0, k+1], u0x[0, k+1])
        
        # 角度处理
        theta0_sub = 0
        flagplne0 = 0
            
        if theta0[0, k] >= np.pi/2 and theta0[0, k] <= np.pi and \
        theta0[0, k+1] <= -np.pi/2 and theta0[0, k+1] >= -np.pi:
            flagplne0 = -1
        elif theta0[0, k] <= -np.pi/2 and theta0[0, k] >= -np.pi and \
            theta0[0, k+1] >= np.pi/2 and theta0[0, k+1] <= np.pi:
            flagplne0 = 1
        
        theta0_sub = theta0[0, k] + 2 * np.pi * flagplne0
            
        # 计算角速度和线速度
        w0[0, k+1] = (theta0[0, k+1] - theta0_sub)/t1
        v0[0, k+1] = np.sqrt((x0[0, k+1] - x0[0, k])**2 + 
                            (y0[0, k+1] - y0[0, k])**2)/t1
        
    # 更新机器人编队误差状态
    zx1[:, k] = np.array([x_hat1[0, k] - px1[0, k+1], y_hat1[0, k] - py1[0, k+1]])
    
    zx2[:, k] = np.array([x_hat2[0, k] - px2[0, k+1], y_hat2[0, k] - py2[0, k+1]])
    
    zx3[:, k] = np.array([x_hat3[0, k] - px3[0, k+1], y_hat3[0, k] - py3[0, k+1]])
    
    zx4[:, k] = np.array([x_hat4[0, k] - px4[0, k+1], y_hat4[0, k] - py4[0, k+1]])
    
    zx5[:, k] = np.array([x_hat5[0, k] - px5[0, k+1], y_hat5[0, k] - py5[0, k+1]])


    z0[:, k+1] = np.array([x0[0, k+1], y0[0, k+1]])

    # print(f"时间步={k+1}, 领导者位置=({x0[0, k+1]:.2f}, {y0[0, k+1]:.2f}), 速度=({u0x[0, k+1]:.2f}, {u0y[0, k+1]:.2f})")
          
    # 观测器状态存储列表，便于动态访问
    zxeo_all = [zxeo_1, zxeo_2, zxeo_3, zxeo_4, zxeo_5]
    dzxeo_all = [dzxeo_1, dzxeo_2, dzxeo_3, dzxeo_4, dzxeo_5]
    
    # 动态计算每个机器人的观测器状态更新
    for robot_id in range(Rnum):
        sum_dzxeo_j = np.zeros((2, 1))
        sum_consensus_error_zxeo = np.zeros((2, 1))
        
        # 根据邻接矩阵动态获取邻居
        for neighbor_id in range(Rnum):
            if A[robot_id, neighbor_id] != 0:  # 如果存在连接
                weight = A[robot_id, neighbor_id]
                sum_dzxeo_j += weight * dzxeo_all[neighbor_id][:, k].reshape(2, 1)
                sum_consensus_error_zxeo += weight * (zxeo_all[robot_id][:, k].reshape(2, 1) - zxeo_all[neighbor_id][:, k].reshape(2, 1))
                
                # # 🔍 打印前3个时间步的邻居观测器数据
                # if robot_id == 0:
                #     if k < 3:
                #         print(f"[时间步{k}] Robot {robot_id} 接收到邻居 {neighbor_id} 的观测器数据: "
                #             f"zxeo=[{zxeo_all[neighbor_id][0, k]:.10f}, {zxeo_all[neighbor_id][1, k]:.10f}], "
                #             f"dzxeo=[{dzxeo_all[neighbor_id][0, k]:.10f}, {dzxeo_all[neighbor_id][1, k]:.10f}]")
        
        # 计算SS项
        SS = sum_consensus_error_zxeo + B[robot_id, robot_id] * (zxeo_all[robot_id][:, k].reshape(2, 1) - z0[:, k+1].reshape(2, 1))
        
        # 观测器状态增量
        dzxeo_all[robot_id][:, k+1] = (
            (1.0 / (D[robot_id, robot_id] + B[robot_id, robot_id])) * (sum_dzxeo_j + B[robot_id, robot_id] * u0[:, k+1].reshape(2, 1))
            - ob_eta * (1.0 / (D[robot_id, robot_id] + B[robot_id, robot_id]))
            * np.sign(SS) * np.power(np.abs(SS), ob_alp)
        ).flatten()

        # 观测器状态
        zxeo_all[robot_id][:, k+1] = dzxeo_all[robot_id][:, k+1] * t1 + zxeo_all[robot_id][:, k]

        # if robot_id == 2:
        #     if k<3:
        #         print(f"[时间步{k}] Robot {robot_id} 计算结果: sum_dzxeo_j=[{sum_dzxeo_j[0,0]:.10f}, {sum_dzxeo_j[1,0]:.10f}], "
        #             f"SS=[{SS[0,0]:.10f}, {SS[1,0]:.10f}], "
        #             f"dzxeo=[{dzxeo_all[robot_id][0, k+1]:.10f}, {dzxeo_all[robot_id][1, k+1]:.10f}],"
        #             f"zxeo=[{zxeo_all[robot_id][0, k+1]:.10f}, {zxeo_all[robot_id][1, k+1]:.10f}]")
        
    # 计算误差和动态反馈项
    zx_all = [zx1, zx2, zx3, zx4, zx5]
    ex_all = [ex1, ex2, ex3, ex4, ex5]
    fx_all = [fx1, fx2, fx3, fx4, fx5]



    for robot_id in range(Rnum):
        ex_all[robot_id][:, k] = zx_all[robot_id][:, k] - zxeo_all[robot_id][:, k]
        fx_all[robot_id][:, k] = ex_all[robot_id][:, k] - t1 * dzxeo_all[robot_id][:, k+1]

    # if k<3:
    #     print(f"[时间步{k}] Robot {0} zx=[{zx_all[0][0, k]:.10f}, {zx_all[0][1, k]:.10f}], "
    #             f"ex_all=[{ex_all[0][0, k]:.10f}, {ex_all[0][1, k]:.10f}], "
    #             f"fx_all=[{fx_all[0][0, k]:.10f}, {fx_all[0][1, k]:.10f}]")

    # # 计算动态系统状态
    # ex1[:, k] = Fx_1 @ zx1[:, k] - zx3[:, k]-z0[:, k+1]
    # fx1[:, k] = ex1[:, k] - t1*u0[:, k+1]

    # ex2[:, k] = Fx_2 @ zx2[:, k] - zx1[:, k] - zx3[:, k]
    # fx2[:, k] = ex2[:, k] 

    # ex3[:, k] = Fx_3 @ zx3[:, k] - zx4[:, k]
    # fx3[:, k] = ex3[:, k] 

    # ex4[:, k] = Fx_4 @ zx4[:, k] - zx1[:, k]
    # fx4[:, k] = ex4[:, k] 

    # ex5[:, k] = Fx_5 @ zx5[:, k] - zx2[:, k]
    # fx5[:, k] = ex5[:, k] 
   
    # print(ex1[:, 1], ex2[:, 1], ex3[:, 1], ex4[:, 1], ex5[:, 1])
    # print(fx1[:, 1], fx2[:, 1], fx3[:, 1], fx4[:, 1], fx5[:, 1])

    raw_static_avoidance = [pxy_static_1, pxy_static_2, pxy_static_3, pxy_static_4, pxy_static_5]
    if not ENABLE_PLAN_AVOIDANCE_GUIDE:
        raw_static_avoidance = [np.zeros(2) for _ in range(Rnum)]
        filtered_static_avoidance.fill(0.0)
    for robot_id, raw_guide in enumerate(raw_static_avoidance):
        raw_guide = np.asarray(raw_guide, dtype=float).reshape(2)
        raw_norm = np.linalg.norm(raw_guide)
        if raw_norm > FULL_MPC_AVOIDANCE_GUIDE_MAX:
            raw_guide = raw_guide * (FULL_MPC_AVOIDANCE_GUIDE_MAX / raw_norm)
            raw_norm = FULL_MPC_AVOIDANCE_GUIDE_MAX
        filtered_static_avoidance[robot_id] = (
            FULL_MPC_AVOIDANCE_GUIDE_FILTER * filtered_static_avoidance[robot_id]
            + (1.0 - FULL_MPC_AVOIDANCE_GUIDE_FILTER) * raw_guide
        )

    goal_state_1 = build_full_mpc_goal_states(0, zxeo_1[:, k+1], dzxeo_1[:, k+1], t2, thetac1[0, k], filtered_static_avoidance[0])
    goal_state_2 = build_full_mpc_goal_states(1, zxeo_2[:, k+1], dzxeo_2[:, k+1], t2, thetac2[0, k], filtered_static_avoidance[1])
    goal_state_3 = build_full_mpc_goal_states(2, zxeo_3[:, k+1], dzxeo_3[:, k+1], t2, thetac3[0, k], filtered_static_avoidance[2])
    goal_state_4 = build_full_mpc_goal_states(3, zxeo_4[:, k+1], dzxeo_4[:, k+1], t2, thetac4[0, k], filtered_static_avoidance[3])
    goal_state_5 = build_full_mpc_goal_states(4, zxeo_5[:, k+1], dzxeo_5[:, k+1], t2, thetac5[0, k], filtered_static_avoidance[4])

    control_1, plan_control_1, states_1, controls_1, plan_controls_1, ok_mpc_1, active_mpc_1, min_h_mpc_1 = solve_full_mpc_dcbf(
        np.array([xc1[0, k], yc1[0, k], thetac1[0, k]]), goal_state_1, dcbf_obstacles,
        dcbf_mpc_last_solutions[0], "robot1"
    )
    control_2, plan_control_2, states_2, controls_2, plan_controls_2, ok_mpc_2, active_mpc_2, min_h_mpc_2 = solve_full_mpc_dcbf(
        np.array([xc2[0, k], yc2[0, k], thetac2[0, k]]), goal_state_2, dcbf_obstacles,
        dcbf_mpc_last_solutions[1], "robot2"
    )
    control_3, plan_control_3, states_3, controls_3, plan_controls_3, ok_mpc_3, active_mpc_3, min_h_mpc_3 = solve_full_mpc_dcbf(
        np.array([xc3[0, k], yc3[0, k], thetac3[0, k]]), goal_state_3, dcbf_obstacles,
        dcbf_mpc_last_solutions[2], "robot3"
    )
    control_4, plan_control_4, states_4, controls_4, plan_controls_4, ok_mpc_4, active_mpc_4, min_h_mpc_4 = solve_full_mpc_dcbf(
        np.array([xc4[0, k], yc4[0, k], thetac4[0, k]]), goal_state_4, dcbf_obstacles,
        dcbf_mpc_last_solutions[3], "robot4"
    )
    control_5, plan_control_5, states_5, controls_5, plan_controls_5, ok_mpc_5, active_mpc_5, min_h_mpc_5 = solve_full_mpc_dcbf(
        np.array([xc5[0, k], yc5[0, k], thetac5[0, k]]), goal_state_5, dcbf_obstacles,
        dcbf_mpc_last_solutions[4], "robot5"
    )

    dcbf_mpc_last_solutions[0] = {"states": states_1, "controls": controls_1, "plan_controls": plan_controls_1}
    dcbf_mpc_last_solutions[1] = {"states": states_2, "controls": controls_2, "plan_controls": plan_controls_2}
    dcbf_mpc_last_solutions[2] = {"states": states_3, "controls": controls_3, "plan_controls": plan_controls_3}
    dcbf_mpc_last_solutions[3] = {"states": states_4, "controls": controls_4, "plan_controls": plan_controls_4}
    dcbf_mpc_last_solutions[4] = {"states": states_5, "controls": controls_5, "plan_controls": plan_controls_5}

    store_full_mpc_solution(control_1, plan_control_1, states_1, k, x_hat1, y_hat1, u1, uz1)
    store_full_mpc_solution(control_2, plan_control_2, states_2, k, x_hat2, y_hat2, u2, uz2)
    store_full_mpc_solution(control_3, plan_control_3, states_3, k, x_hat3, y_hat3, u3, uz3)
    store_full_mpc_solution(control_4, plan_control_4, states_4, k, x_hat4, y_hat4, u4, uz4)
    store_full_mpc_solution(control_5, plan_control_5, states_5, k, x_hat5, y_hat5, u5, uz5)

    dcbf_mpc_success_history[:, k] = np.array([ok_mpc_1, ok_mpc_2, ok_mpc_3, ok_mpc_4, ok_mpc_5], dtype=int)
    dcbf_mpc_active_obstacle_history[:, k] = np.array([active_mpc_1, active_mpc_2, active_mpc_3, active_mpc_4, active_mpc_5])
    dcbf_mpc_min_h_history[:, k] = np.array([min_h_mpc_1, min_h_mpc_2, min_h_mpc_3, min_h_mpc_4, min_h_mpc_5])

    # 控制输入限制和轨迹更新
    # 限制x方向输入
    if u1[0, k+1] >= 0.22:
        u1[0, k+1] = 0.22
    elif u1[0, k+1] < -0.22:
        u1[0, k+1] = -0.22
    
    # 限制y方向输入
    if u1[1, k+1] >= 0.22:
        u1[1, k+1] = 0.22
    elif u1[1, k+1] < -0.22:
        u1[1, k+1] = -0.22

    # 提取控制输入
    ux1[0, k+1] = u1[0, k+1]
    uy1[0, k+1] = u1[1, k+1]

    # 计算期望航向角
    thetar1[0, k+1] = np.arctan2(uy1[0, k+1], ux1[0, k+1])
    
    # 航向角跨象限处理
    thetar_sub = 0
    flagplne = 0
    if thetar1[0, k] >= np.pi/2 and thetar1[0, k] <= np.pi and \
    thetar1[0, k+1] <= -np.pi/2 and thetar1[0, k+1] >= -np.pi:
        flagplne = -1
    elif thetar1[0, k] <= -np.pi/2 and thetar1[0, k] >= -np.pi and \
        thetar1[0, k+1] >= np.pi/2 and thetar1[0, k+1] <= np.pi:
        flagplne = 1
        
    thetar1_sub = thetar1[0, k] + 2 * np.pi * flagplne
    wr1[0, k+1] = (thetar1[0, k+1] - thetar1_sub) / t1
    vr1[0, k+1] = np.sqrt(ux1[0, k+1]**2 + uy1[0, k+1]**2)


# 限制x方向输入
    if u2[0, k+1] >= 0.22:
        u2[0, k+1] = 0.22
    elif u2[0, k+1] < -0.22:
        u2[0, k+1] = -0.22

    # 限制y方向输入
    if u2[1, k+1] >= 0.22:
        u2[1, k+1] = 0.22
    elif u2[1, k+1] < -0.22:
        u2[1, k+1] = -0.22
        
    # 提取控制输入
    ux2[0, k+1] = u2[0, k+1]
    uy2[0, k+1] = u2[1, k+1]
    
    # 计算期望航向角
    thetar2[0, k+1] = np.arctan2(uy2[0, k+1], ux2[0, k+1])
    
    # 航向角跨象限处理
    thetar_sub = 0
    flagplne = 0
    if thetar2[0, k] >= np.pi/2 and thetar2[0, k] <= np.pi and \
    thetar2[0, k+1] <= -np.pi/2 and thetar2[0, k+1] >= -np.pi:
        flagplne = -1
    elif thetar2[0, k] <= -np.pi/2 and thetar2[0, k] >= -np.pi and \
        thetar2[0, k+1] >= np.pi/2 and thetar2[0, k+1] <= np.pi:
        flagplne = 1
        
    thetar2_sub = thetar2[0, k] + 2 * np.pi * flagplne
    wr2[0, k+1] = (thetar2[0, k+1] - thetar2_sub) / t1
    vr2[0, k+1] = np.sqrt(ux2[0, k+1]**2 + uy2[0, k+1]**2)


# 限制x方向输入
    if u3[0, k+1] >= 0.22:
        u3[0, k+1] = 0.22
    elif u3[0, k+1] < -0.22:
        u3[0, k+1] = -0.22
    
    # 限制y方向输入
    if u3[1, k+1] >= 0.22:
        u3[1, k+1] = 0.22
    elif u3[1, k+1] < -0.22:
        u3[1, k+1] = -0.22
        
    # 提取控制输入
    ux3[0, k+1] = u3[0, k+1]
    uy3[0, k+1] = u3[1, k+1]
    
    # 计算期望航向角
    thetar3[0, k+1] = np.arctan2(uy3[0, k+1], ux3[0, k+1])
    
    # 航向角跨象限处理
    thetar_sub = 0
    flagplne = 0
    if thetar3[0, k] >= np.pi/2 and thetar3[0, k] <= np.pi and \
    thetar3[0, k+1] <= -np.pi/2 and thetar3[0, k+1] >= -np.pi:
        flagplne = -1
    elif thetar3[0, k] <= -np.pi/2 and thetar3[0, k] >= -np.pi and \
        thetar3[0, k+1] >= np.pi/2 and thetar3[0, k+1] <= np.pi:
        flagplne = 1
        
    thetar3_sub = thetar3[0, k] + 2 * np.pi * flagplne
    wr3[0, k+1] = (thetar3[0, k+1] - thetar3_sub) / t1
    vr3[0, k+1] = np.sqrt(ux3[0, k+1]**2 + uy3[0, k+1]**2)


# 限制x方向输入
    if u4[0, k+1] >= 0.22:
        u4[0, k+1] = 0.22
    elif u4[0, k+1] < -0.22:
        u4[0, k+1] = -0.22
    
    # 限制y方向输入
    if u4[1, k+1] >= 0.22:
        u4[1, k+1] = 0.22
    elif u4[1, k+1] < -0.22:
        u4[1, k+1] = -0.22
        
    # 提取控制输入
    ux4[0, k+1] = u4[0, k+1]
    uy4[0, k+1] = u4[1, k+1]
    
    # 计算期望航向角
    thetar4[0, k+1] = np.arctan2(uy4[0, k+1], ux4[0, k+1])
    
    # 航向角跨象限处理
    thetar_sub = 0
    flagplne = 0
    if thetar4[0, k] >= np.pi/2 and thetar4[0, k] <= np.pi and \
    thetar4[0, k+1] <= -np.pi/2 and thetar4[0, k+1] >= -np.pi:
        flagplne = -1
    elif thetar4[0, k] <= -np.pi/2 and thetar4[0, k] >= -np.pi and \
        thetar4[0, k+1] >= np.pi/2 and thetar4[0, k+1] <= np.pi:
        flagplne = 1
        
    thetar4_sub = thetar4[0, k] + 2 * np.pi * flagplne
    wr4[0, k+1] = (thetar4[0, k+1] - thetar4_sub) / t1
    vr4[0, k+1] = np.sqrt(ux4[0, k+1]**2 + uy4[0, k+1]**2)


# 限制x方向输入
    if u5[0, k+1] >= 0.22:
        u5[0, k+1] = 0.22
    elif u5[0, k+1] < -0.22:
        u5[0, k+1] = -0.22
    
    # 限制y方向输入
    if u5[1, k+1] >= 0.22:
        u5[1, k+1] = 0.22
    elif u5[1, k+1] < -0.22:
        u5[1, k+1] = -0.22
        
    # 提取控制输入
    ux5[0, k+1] = u5[0, k+1]
    uy5[0, k+1] = u5[1, k+1]
    
    # 计算期望航向角
    thetar5[0, k+1] = np.arctan2(uy5[0, k+1], ux5[0, k+1])
    
    # 航向角跨象限处理
    thetar_sub = 0
    flagplne = 0
    if thetar5[0, k] >= np.pi/2 and thetar5[0, k] <= np.pi and \
    thetar5[0, k+1] <= -np.pi/2 and thetar5[0, k+1] >= -np.pi:
        flagplne = -1
    elif thetar5[0, k] <= -np.pi/2 and thetar5[0, k] >= -np.pi and \
        thetar5[0, k+1] >= np.pi/2 and thetar5[0, k+1] <= np.pi:
        flagplne = 1
        
    thetar5_sub = thetar5[0, k] + 2 * np.pi * flagplne
    wr5[0, k+1] = (thetar5[0, k+1] - thetar5_sub) / t1
    vr5[0, k+1] = np.sqrt(ux5[0, k+1]**2 + uy5[0, k+1]**2)
    
    # x_hat/y_hat已由Full MPC预测状态写入。
    xr1[:, k+1] = x_hat1[:, k+1]
    yr1[:, k+1] = y_hat1[:, k+1]
    xr2[:, k+1] = x_hat2[:, k+1]
    yr2[:, k+1] = y_hat2[:, k+1]
    xr3[:, k+1] = x_hat3[:, k+1]
    yr3[:, k+1] = y_hat3[:, k+1]
    xr4[:, k+1] = x_hat4[:, k+1]
    yr4[:, k+1] = y_hat4[:, k+1]
    xr5[:, k+1] = x_hat5[:, k+1]
    yr5[:, k+1] = y_hat5[:, k+1]


# merge_plan_track_modification_5: 旧跟踪QP和误差计算已全部删除

    # 实际控制输入限制和更新
    # 限制线速度
    if uz1[0, k+1] >= FULL_MPC_V_MAX:
        uz1[0, k+1] = FULL_MPC_V_MAX
    elif uz1[0, k+1] < -FULL_MPC_V_MAX:
        uz1[0, k+1] = -FULL_MPC_V_MAX
    
    # 限制角速度
    if uz1[1, k+1] >= FULL_MPC_OMEGA_MAX:
        uz1[1, k+1] = FULL_MPC_OMEGA_MAX
    elif uz1[1, k+1] < -FULL_MPC_OMEGA_MAX:
        uz1[1, k+1] = -FULL_MPC_OMEGA_MAX
    
    vc1[0, k+1] = uz1[0, k+1]
    wc1[0, k+1] = uz1[1, k+1]
    
    rrr = 0  # 模拟扰动
    # 更新机器人位姿
    thetac1[0, k+1] = thetac1[0, k] + t1 * wc1[0, k+1]
    
    # 位姿跨象限调整
    if thetac1[0, k+1] >= np.pi:
        thetac1[0, k+1] = thetac1[0, k+1] - 2 * np.pi
    elif thetac1[0, k+1] <= -np.pi:
        thetac1[0, k+1] = thetac1[0, k+1] + 2 * np.pi
    


    if uz2[0, k+1] >= FULL_MPC_V_MAX:
        uz2[0, k+1] = FULL_MPC_V_MAX
    elif uz2[0, k+1] < -FULL_MPC_V_MAX:
        uz2[0, k+1] = -FULL_MPC_V_MAX
    
    # 限制角速度
    if uz2[1, k+1] >= FULL_MPC_OMEGA_MAX:
        uz2[1, k+1] = FULL_MPC_OMEGA_MAX
    elif uz2[1, k+1] < -FULL_MPC_OMEGA_MAX:
        uz2[1, k+1] = -FULL_MPC_OMEGA_MAX
    
    vc2[0, k+1] = uz2[0, k+1]
    wc2[0, k+1] = uz2[1, k+1]
    
    rrr = 0  # 模拟扰动
    # 更新机器人位姿
    thetac2[0, k+1] = thetac2[0, k] + t1 * wc2[0, k+1]
    
    # 位姿跨象限调整
    if thetac2[0, k+1] >= np.pi:
        thetac2[0, k+1] = thetac2[0, k+1] - 2 * np.pi
    elif thetac2[0, k+1] <= -np.pi:
        thetac2[0, k+1] = thetac2[0, k+1] + 2 * np.pi



    if uz3[0, k+1] >= FULL_MPC_V_MAX:
        uz3[0, k+1] = FULL_MPC_V_MAX
    elif uz3[0, k+1] < -FULL_MPC_V_MAX:
        uz3[0, k+1] = -FULL_MPC_V_MAX
    
    # 限制角速度
    if uz3[1, k+1] >= FULL_MPC_OMEGA_MAX:
        uz3[1, k+1] = FULL_MPC_OMEGA_MAX
    elif uz3[1, k+1] < -FULL_MPC_OMEGA_MAX:
        uz3[1, k+1] = -FULL_MPC_OMEGA_MAX
    
    vc3[0, k+1] = uz3[0, k+1]
    wc3[0, k+1] = uz3[1, k+1]
    
    rrr = 0  # 模拟扰动
    # 更新机器人位姿
    thetac3[0, k+1] = thetac3[0, k] + t1 * wc3[0, k+1]
    
    # 位姿跨象限调整
    if thetac3[0, k+1] >= np.pi:
        thetac3[0, k+1] = thetac3[0, k+1] - 2 * np.pi
    elif thetac3[0, k+1] <= -np.pi:
        thetac3[0, k+1] = thetac3[0, k+1] + 2 * np.pi



    if uz4[0, k+1] >= FULL_MPC_V_MAX:
        uz4[0, k+1] = FULL_MPC_V_MAX
    elif uz4[0, k+1] < -FULL_MPC_V_MAX:
        uz4[0, k+1] = -FULL_MPC_V_MAX
    
    # 限制角速度
    if uz4[1, k+1] >= FULL_MPC_OMEGA_MAX:
        uz4[1, k+1] = FULL_MPC_OMEGA_MAX
    elif uz4[1, k+1] < -FULL_MPC_OMEGA_MAX:
        uz4[1, k+1] = -FULL_MPC_OMEGA_MAX
    
    vc4[0, k+1] = uz4[0, k+1]
    wc4[0, k+1] = uz4[1, k+1]
    
    rrr = 0  # 模拟扰动
    # 更新机器人位姿
    thetac4[0, k+1] = thetac4[0, k] + t1 * wc4[0, k+1]
    
    # 位姿跨象限调整
    if thetac4[0, k+1] >= np.pi:
        thetac4[0, k+1] = thetac4[0, k+1] - 2 * np.pi
    elif thetac4[0, k+1] <= -np.pi:
        thetac4[0, k+1] = thetac4[0, k+1] + 2 * np.pi



    if uz5[0, k+1] >= FULL_MPC_V_MAX:
        uz5[0, k+1] = FULL_MPC_V_MAX
    elif uz5[0, k+1] < -FULL_MPC_V_MAX:
        uz5[0, k+1] = -FULL_MPC_V_MAX
    
    # 限制角速度
    if uz5[1, k+1] >= FULL_MPC_OMEGA_MAX:
        uz5[1, k+1] = FULL_MPC_OMEGA_MAX
    elif uz5[1, k+1] < -FULL_MPC_OMEGA_MAX:
        uz5[1, k+1] = -FULL_MPC_OMEGA_MAX
    
    vc5[0, k+1] = uz5[0, k+1]
    wc5[0, k+1] = uz5[1, k+1]
    
    # print(f"时间步={k+1}, "
    #       f"机器人1：速度={vc1[0, k+1]:.3f}, 角速度={wc1[0, k+1]:.3f} | "
    #       f"机器人2：速度={vc2[0, k+1]:.3f}, 角速度={wc2[0, k+1]:.3f} | "
    #       f"机器人3：速度={vc3[0, k+1]:.3f}, 角速度={wc3[0, k+1]:.3f} | "
    #       f"机器人4：速度={vc4[0, k+1]:.3f}, 角速度={wc4[0, k+1]:.3f} | "
    #       f"机器人5：速度={vc5[0, k+1]:.3f}, 角速度={wc5[0, k+1]:.3f})")


    rrr = 0  # 模拟扰动
    # 更新机器人位姿
    thetac5[0, k+1] = thetac5[0, k] + t1 * wc5[0, k+1]
    
    # 位姿跨象限调整
    if thetac5[0, k+1] >= np.pi:
        thetac5[0, k+1] = thetac5[0, k+1] - 2 * np.pi
    elif thetac5[0, k+1] <= -np.pi:
        thetac5[0, k+1] = thetac5[0, k+1] + 2 * np.pi


    # 更新机器人位置
    
    xc1[0, k+1] = xc1[0, k] + vc1[0, k+1] * t1 * np.cos(thetac1[0, k+1]) + \
                (np.random.random() - 0.5) * rrr
    yc1[0, k+1] = yc1[0, k] + vc1[0, k+1] * t1 * np.sin(thetac1[0, k+1]) + \
                (np.random.random() - 0.5) * rrr

    xc2[0, k+1] = xc2[0, k] + vc2[0, k+1] * t1 * np.cos(thetac2[0, k+1]) + \
                (np.random.random() - 0.5) * rrr
    yc2[0, k+1] = yc2[0, k] + vc2[0, k+1] * t1 * np.sin(thetac2[0, k+1]) + \
                (np.random.random() - 0.5) * rrr
    
    xc3[0, k+1] = xc3[0, k] + vc3[0, k+1] * t1 * np.cos(thetac3[0, k+1]) + \
                (np.random.random() - 0.5) * rrr
    yc3[0, k+1] = yc3[0, k] + vc3[0, k+1] * t1 * np.sin(thetac3[0, k+1]) + \
                (np.random.random() - 0.5) * rrr
    
    xc4[0, k+1] = xc4[0, k] + vc4[0, k+1] * t1 * np.cos(thetac4[0, k+1]) + \
                (np.random.random() - 0.5) * rrr
    yc4[0, k+1] = yc4[0, k] + vc4[0, k+1] * t1 * np.sin(thetac4[0, k+1]) + \
                (np.random.random() - 0.5) * rrr

    xc5[0, k+1] = xc5[0, k] + vc5[0, k+1] * t1 * np.cos(thetac5[0, k+1]) + \
                (np.random.random() - 0.5) * rrr
    yc5[0, k+1] = yc5[0, k] + vc5[0, k+1] * t1 * np.sin(thetac5[0, k+1]) + \
                (np.random.random() - 0.5) * rrr




    # 更新质心位置
    xccc[0, k+1] = np.sum(xc1[:, k+1]+xc2[:, k+1]+xc3[:, k+1]+xc4[:, k+1]+xc5[:, k+1])/Rnum
    yccc[0, k+1] = np.sum(yc1[:, k+1]+yc2[:, k+1]+yc3[:, k+1]+yc4[:, k+1]+yc5[:, k+1])/Rnum

    # 存储位置误差
    
    # zx1[:, k+1] = np.array([
    #     x_hat1[0, k+1] - px1[0, k+1],
    #     y_hat1[0, k+1] - py1[0, k+1]
    # ])

    # zx2[:, k+1] = np.array([
    #     x_hat2[0, k+1] - px2[0, k+1],
    #     y_hat2[0, k+1] - py2[0, k+1]
    # ])

    # zx3[:, k+1] = np.array([
    #     x_hat3[0, k+1] - px3[0, k+1],
    #     y_hat3[0, k+1] - py3[0, k+1]
    # ])


    # zx4[:, k+1] = np.array([
    #     x_hat4[0, k+1] - px4[0, k+1],
    #     y_hat4[0, k+1] - py4[0, k+1]
    # ])


    # zx5[:, k+1] = np.array([
    #     x_hat5[0, k+1] - px5[0, k+1],
    #     y_hat5[0, k+1] - py5[0, k+1]
    # ])


    # z0[:, k+1] = np.array([x0[0, k+1], y0[0, k+1]])

    t2 = t2 + t1


# 可视化
tclc = np.array([1, int(k/2), k+1])



# 图1：机器人轨迹（含障碍物）
plt.figure(1, figsize=(10, 8))

# 绘制机器人轨迹
plt.plot(xc1[0, :], yc1[0, :], linewidth=2, label='Robot_1')
plt.plot(xc2[0, :], yc2[0, :], linewidth=2, label='Robot_2')
plt.plot(xc3[0, :], yc3[0, :], linewidth=2, label='Robot_3')
plt.plot(xc4[0, :], yc4[0, :], linewidth=2, label='Robot_4')
plt.plot(xc5[0, :], yc5[0, :], linewidth=2, label='Robot_5')

plt.plot(x_hat1[0, :], y_hat1[0, :], '--', linewidth=1.2, alpha=0.6, label='Ref_1')
plt.plot(x_hat2[0, :], y_hat2[0, :], '--', linewidth=1.2, alpha=0.6, label='Ref_2')
plt.plot(x_hat3[0, :], y_hat3[0, :], '--', linewidth=1.2, alpha=0.6, label='Ref_3')
plt.plot(x_hat4[0, :], y_hat4[0, :], '--', linewidth=1.2, alpha=0.6, label='Ref_4')
plt.plot(x_hat5[0, :], y_hat5[0, :], '--', linewidth=1.2, alpha=0.6, label='Ref_5')

plt.plot(x0[0, :], y0[0, :], '--', linewidth=2, label='Robot_L')
plt.plot(xccc[0, :], yccc[0, :], linewidth=2, label='Actual centroid')

plot_static_obstacles(plt.gca())

# 绘制编队连线（在几个关键时刻）
for j in range(3):
    plt.plot([xc1[0, tclc[j]], xc2[0, tclc[j]]],
            [yc1[0, tclc[j]], yc2[0, tclc[j]]],
            color=[0.7, 0.3, 0.9], linewidth=3, alpha=0.7)

    plt.plot([xc2[0, tclc[j]], xc3[0, tclc[j]]],
            [yc2[0, tclc[j]], yc3[0, tclc[j]]],
            color=[0.7, 0.3, 0.9], linewidth=3, alpha=0.7)

    plt.plot([xc3[0, tclc[j]], xc4[0, tclc[j]]],
            [yc3[0, tclc[j]], yc4[0, tclc[j]]],
            color=[0.7, 0.3, 0.9], linewidth=3, alpha=0.7)

    plt.plot([xc4[0, tclc[j]], xc5[0, tclc[j]]],
            [yc4[0, tclc[j]], yc5[0, tclc[j]]],
            color=[0.7, 0.3, 0.9], linewidth=3, alpha=0.7)

    plt.plot([xc5[0, tclc[j]], xc1[0, tclc[j]]],
            [yc5[0, tclc[j]], yc1[0, tclc[j]]],
            color=[0.7, 0.3, 0.9], linewidth=3, alpha=0.7)

plt.axis('equal')
plt.xlabel('X coordinate (m)')
plt.ylabel('Y coordinate (m)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.title('Robot Trajectories with Obstacles')


# # 图2：机器人与障碍物最小距离
# plt.figure(2, figsize=(10, 6))

# plt.plot(t[:-1], dobsmin_1[0, :], linewidth=2, label='dobsmin_1')
# plt.plot(t[:-1], dobsmin_2[0, :], linewidth=2, label='dobsmin_2')
# plt.plot(t[:-1], dobsmin_3[0, :], linewidth=2, label='dobsmin_3')
# plt.plot(t[:-1], dobsmin_4[0, :], linewidth=2, label='dobsmin_4')
# plt.plot(t[:-1], dobsmin_5[0, :], linewidth=2, label='dobsmin_5')

# # 添加安全距离阈值线
# plt.axhline(y=ord_safe, color='red', linestyle='--', linewidth=2, 
#            label=f'Safety threshold ({ord_safe}m)', alpha=0.8)
# plt.axhline(y=ord_avoid, color='orange', linestyle=':', linewidth=2, 
#            label=f'Avoidance threshold ({ord_avoid}m)', alpha=0.8)

# plt.xlabel('Time (s)')
# plt.ylabel('Min distance of obstacle')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.title('Minimum Distance between Robots and Obstacles')
# plt.xlim([0, k*t1])
# plt.ylim([0, 4.0])  # 根据图像调整y轴范围

# 图3：线速度
plt.figure(3)

plt.plot(t, vc1[0, :], linewidth=2)
plt.plot(t, vc2[0, :], linewidth=2)
plt.plot(t, vc3[0, :], linewidth=2)
plt.plot(t, vc4[0, :], linewidth=2)
plt.plot(t, vc5[0, :], linewidth=2)


plt.plot(t, v0[0, :], '--', linewidth=2)
plt.axis([0, k*0.1, -0.51, 0.51])
plt.grid(True)
plt.xlabel('Time(s)')
plt.ylabel('Linear velocities')
plt.legend(['v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_L'])

# 图4：角速度
plt.figure(4)

plt.plot(t, wc1[0, :], linewidth=2)
plt.plot(t, wc2[0, :], linewidth=2)
plt.plot(t, wc3[0, :], linewidth=2)
plt.plot(t, wc4[0, :], linewidth=2)
plt.plot(t, wc5[0, :], linewidth=2)


plt.plot(t, w0[0, :], '--', linewidth=2)
plt.axis([0, k*0.1, -2.6, 2.6])
plt.grid(True)
plt.xlabel('Time(s)')
plt.ylabel('Angular velocities')
plt.legend(['ω_1', 'ω_2', 'ω_3', 'ω_4', 'ω_5', 'ω_L'])

# 图5：X方向误差
plt.figure(5)

plt.plot(t, xc1[0, :] - z0[0, :] - px1[0, :], linewidth=2)
plt.plot(t, xc2[0, :] - z0[0, :] - px2[0, :], linewidth=2)
plt.plot(t, xc3[0, :] - z0[0, :] - px3[0, :], linewidth=2)
plt.plot(t, xc4[0, :] - z0[0, :] - px4[0, :], linewidth=2)
plt.plot(t, xc5[0, :] - z0[0, :] - px5[0, :], linewidth=2)


plt.grid(True)
plt.xlabel('Time(s)')
plt.ylabel('Errors of x_i')
plt.ylim([-1,1])
plt.legend(['x_1-x_L-p_1x', 'x_2-x_L-p_2x', 'x_3-x_L-p_3x', 
           'x_4-x_L-p_4x', 'x_5-x_L-p_5x'])





# 图6的Y方向误差部分
plt.figure(6)

plt.plot(t, yc1[0, :] - z0[1, :] - py1[0, :], linewidth=2)
plt.plot(t, yc2[0, :] - z0[1, :] - py2[0, :], linewidth=2)
plt.plot(t, yc3[0, :] - z0[1, :] - py3[0, :], linewidth=2)
plt.plot(t, yc4[0, :] - z0[1, :] - py4[0, :], linewidth=2)
plt.plot(t, yc5[0, :] - z0[1, :] - py5[0, :], linewidth=2)



plt.grid(True)
plt.xlabel('Time(s)')
plt.ylabel('Errors of y_i')
plt.ylim([-1,1])
plt.legend(['y_1-y_L-p_1y', 'y_2-y_L-p_2y', 'y_3-y_L-p_3y', 
           'y_4-y_L-p_4y', 'y_5-y_L-p_5y'])




# 图7：航向角误差
plt.figure(7)

plt.plot(t, thetac1[0, :] - theta0[0, :], linewidth=2)
plt.plot(t, thetac2[0, :] - theta0[0, :], linewidth=2)
plt.plot(t, thetac3[0, :] - theta0[0, :], linewidth=2)
plt.plot(t, thetac4[0, :] - theta0[0, :], linewidth=2)
plt.plot(t, thetac5[0, :] - theta0[0, :], linewidth=2)

plt.grid(True)
plt.xlabel('Time(s)')
plt.ylabel('Errors of θ_i (rad)')
plt.ylim([-4,4])
plt.legend(['θ_1-θ_L', 'θ_2-θ_L', 'θ_3-θ_L', 'θ_4-θ_L', 'θ_5-θ_L'])


# # 图7：实际朝向角
# plt.figure(7)

# plt.plot(t, thetac1[0, :], linewidth=2)
# plt.plot(t, thetac2[0, :], linewidth=2)
# plt.plot(t, thetac3[0, :], linewidth=2)
# plt.plot(t, thetac4[0, :], linewidth=2)
# plt.plot(t, thetac5[0, :], linewidth=2)

# plt.grid(True)
# plt.xlabel('Time(s)')
# plt.ylabel('θ_i')
# plt.legend(['θ_1', 'θ_2', 'θ_3', 'θ_4', 'θ_5'])

# # 图8：领导者朝向角
# plt.figure(8)
# plt.plot(t, theta0[0, :], linewidth=2)
# plt.grid(True)
# plt.xlabel('Time(s)')
# plt.ylabel('θ_L')

# # 图9：实际X位置
# plt.figure(9)

# plt.plot(t, xc1[0, :], linewidth=2)
# plt.plot(t, xc2[0, :], linewidth=2)
# plt.plot(t, xc3[0, :], linewidth=2)
# plt.plot(t, xc4[0, :], linewidth=2)
# plt.plot(t, xc5[0, :], linewidth=2)


# plt.grid(True)
# plt.xlabel('Time(s)')
# plt.ylabel(' x_i')
# plt.legend(['x_1', 'x_2', 'x_3', 'x_4', 'x_5'])


# # 图10：实际领导者位置
# plt.figure(10)

# plt.plot(t,  z0[0, :] , linewidth=2)
# plt.grid(True)
# plt.xlabel('Time(s)')
# plt.ylabel('x_L')
# plt.legend(['x_L'])


# # 图11：x方向偏移
# plt.figure(11)

# plt.plot(t, px1[0, :], linewidth=2)
# plt.plot(t, px2[0, :], linewidth=2)
# plt.plot(t, px3[0, :], linewidth=2)
# plt.plot(t, px4[0, :], linewidth=2)
# plt.plot(t, px5[0, :], linewidth=2)

# plt.grid(True)
# plt.xlabel('Time(s)')
# plt.ylabel('p_xi')
# plt.legend(['p_1x', 'p_2x', 'p_3x', 'p_4x', 'p_5x'])


# # 图12：Y方向实际位置
# plt.figure(12)

# plt.plot(t, yc1[0, :] ,linewidth=2)
# plt.plot(t, yc2[0, :], linewidth=2)
# plt.plot(t, yc3[0, :], linewidth=2)
# plt.plot(t, yc4[0, :], linewidth=2)
# plt.plot(t, yc5[0, :], linewidth=2)



# plt.grid(True)
# plt.xlabel('Time(s)')
# plt.ylabel(' y_i')
# plt.legend(['y_1', 'y_2', 'y_3', 'y_4', 'y_5'])


# # 图13：领导者Y方向位置
# plt.figure(13)

# plt.plot(t,  z0[1, :], linewidth=2)
# plt.grid(True)
# plt.xlabel('Time(s)')
# plt.ylabel('y_L')
# plt.legend(['y_L'])



# # 图14：Y方向偏移
# plt.figure(14)

# plt.plot(t,  py1[0, :], linewidth=2)
# plt.plot(t,  py2[0, :], linewidth=2)
# plt.plot(t,  py3[0, :], linewidth=2)
# plt.plot(t,  py4[0, :], linewidth=2)
# plt.plot(t,  py5[0, :], linewidth=2)

# plt.grid(True)
# plt.xlabel('Time(s)')
# plt.ylabel('py_i')
# plt.legend(['p_1y', 'p_2y', 'p_3y', 'p_4y', 'p_5y'])


# # 图15：线速度
# plt.figure(15)
# plt.plot(t, vc1[0, :], linewidth=2)
# plt.plot(t, vc2[0, :], linewidth=2)
# plt.plot(t, vc3[0, :], linewidth=2)
# plt.plot(t, vc4[0, :], linewidth=2)
# plt.plot(t, vc5[0, :], linewidth=2)
# plt.grid(True)
# plt.xlabel('Time(s)')
# plt.ylabel(' v_i')
# plt.legend(['v_1', 'v_2', 'v_3', 'v_4', 'v_5'])

# # 图16：角速度
# plt.figure(16)
# plt.plot(t, wc1[0, :], linewidth=2)
# plt.plot(t, wc2[0, :], linewidth=2)
# plt.plot(t, wc3[0, :], linewidth=2)
# plt.plot(t, wc4[0, :], linewidth=2)
# plt.plot(t, wc5[0, :], linewidth=2)
# plt.grid(True)
# plt.xlabel('Time(s)')
# plt.ylabel(' ω_i')
# plt.legend(['ω_1', 'ω_2', 'ω_3', 'ω_4', 'ω_5'])


# # 图17：轨迹规划ux
# plt.figure(17)
# plt.plot(t, ux1[0, :], linewidth=2)
# plt.plot(t, ux2[0, :], linewidth=2)
# plt.plot(t, ux3[0, :], linewidth=2)
# plt.plot(t, ux4[0, :], linewidth=2)
# plt.plot(t, ux5[0, :], linewidth=2)
# plt.grid(True)
# plt.xlabel('Time(s)')
# plt.ylabel(' u_x')
# plt.legend(['u_x1', 'u_x2', 'u_x3', 'u_x4', 'u_x5'])

# # 图18：轨迹规划uy
# plt.figure(18)
# plt.plot(t, uy1[0, :], linewidth=2)
# plt.plot(t, uy2[0, :], linewidth=2)
# plt.plot(t, uy3[0, :], linewidth=2)
# plt.plot(t, uy4[0, :], linewidth=2)
# plt.plot(t, uy5[0, :], linewidth=2)
# plt.grid(True)
# plt.xlabel('Time(s)')
# plt.ylabel(' u_y')
# plt.legend(['u_y1', 'u_y2', 'u_y3', 'u_y4', 'u_y5'])

# # 图19：期望x:x_hat
# plt.figure(19)
# plt.plot(t, x_hat1[0, :], linewidth=2)
# plt.plot(t, x_hat2[0, :], linewidth=2)
# plt.plot(t, x_hat3[0, :], linewidth=2)
# plt.plot(t, x_hat4[0, :], linewidth=2)
# plt.plot(t, x_hat5[0, :], linewidth=2)
# plt.grid(True)
# plt.xlabel('Time(s)')
# plt.ylabel(' x_hat')
# plt.legend(['x_hat1', 'x_hat2', 'x_hat3', 'x_hat4', 'x_hat5'])

# # 图20：期望y:y_hat
# plt.figure(20)
# plt.plot(t, y_hat1[0, :], linewidth=2)
# plt.plot(t, y_hat2[0, :], linewidth=2)
# plt.plot(t, y_hat3[0, :], linewidth=2)
# plt.plot(t, y_hat4[0, :], linewidth=2)
# plt.plot(t, y_hat5[0, :], linewidth=2)
# plt.grid(True)
# plt.xlabel('Time(s)')
# plt.ylabel(' y_hat')
# plt.legend(['y_hat1', 'y_hat2', 'y_hat3', 'y_hat4', 'y_hat5'])

# # 图21：ex
# plt.figure(21)
# plt.plot(t[:-1], ex1[0, :], linewidth=2)
# plt.plot(t[:-1], ex1[1, :], linewidth=2)
# plt.plot(t[:-1], ex2[0, :], linewidth=2)
# plt.plot(t[:-1], ex2[1, :], linewidth=2)
# plt.plot(t[:-1], ex3[0, :], linewidth=2)
# plt.plot(t[:-1], ex3[1, :], linewidth=2)
# plt.plot(t[:-1], ex4[0, :], linewidth=2)
# plt.plot(t[:-1], ex4[1, :], linewidth=2)
# plt.plot(t[:-1], ex5[0, :], linewidth=2)
# plt.plot(t[:-1], ex5[1, :], linewidth=2)

# plt.grid(True)
# plt.xlabel('Time(s)')
# plt.ylabel(' ex1')
# plt.legend(['ex1_1', 'ex1_2', 'ex1_3', 'ex1_4', 'ex1_5', 'ex1_6', 'ex1_7', 'ex1_8', 'ex1_9', 'ex1_10'])

# # 图22：fx
# plt.figure(22)
# plt.plot(t[:-1], fx1[0, :], linewidth=2)
# plt.plot(t[:-1], fx1[1, :], linewidth=2)
# plt.plot(t[:-1], fx2[0, :], linewidth=2)
# plt.plot(t[:-1], fx2[1, :], linewidth=2)
# plt.plot(t[:-1], fx3[0, :], linewidth=2)
# plt.plot(t[:-1], fx3[1, :], linewidth=2)
# plt.plot(t[:-1], fx4[0, :], linewidth=2)
# plt.plot(t[:-1], fx4[1, :], linewidth=2)
# plt.plot(t[:-1], fx5[0, :], linewidth=2)
# plt.plot(t[:-1], fx5[1, :], linewidth=2)
# plt.grid(True)
# plt.xlabel('Time(s)')
# plt.ylabel(' fx')
# plt.legend(['fx_1', 'fx_2', 'fx_3', 'fx_4', 'fx_5', 'fx_6', 'fx_7', 'fx_8', 'fx_9', 'fx_10'])

# # 避障力大小图表
# plt.figure(22, figsize=(12, 8))
# plt.plot(t[:-1], avoidance_force_mag_1, linewidth=2, label='Robot 1')
# plt.plot(t[:-1], avoidance_force_mag_2, linewidth=2, label='Robot 2')
# plt.plot(t[:-1], avoidance_force_mag_3, linewidth=2, label='Robot 3')
# plt.plot(t[:-1], avoidance_force_mag_4, linewidth=2, label='Robot 4')
# plt.plot(t[:-1], avoidance_force_mag_5, linewidth=2, label='Robot 5')

# plt.grid(True)
# plt.xlabel('Time (s)', fontsize=12)
# plt.ylabel('Avoidance Force Magnitude', fontsize=12)
# plt.title('Pure Simulation: Avoidance Force Magnitude Over Time', fontsize=14)
# plt.legend()

# # 计算并显示统计信息
# print("\n=== 纯仿真避障力大小统计 ===")
# for i, (robot_name, force_data) in enumerate([
#     ('Robot 1', avoidance_force_mag_1),
#     ('Robot 2', avoidance_force_mag_2), 
#     ('Robot 3', avoidance_force_mag_3),
#     ('Robot 4', avoidance_force_mag_4),
#     ('Robot 5', avoidance_force_mag_5)
# ], 1):
#     max_force = np.max(force_data)
#     avg_force = np.mean(force_data)
#     std_force = np.std(force_data)
#     print(f"{robot_name}: 最大={max_force:.4f}, 平均={avg_force:.4f}, 标准差={std_force:.4f}")

# # 图23：静态障碍物避障力
# plt.figure(23, figsize=(12, 8))
# plt.plot(t[:-1], static_avoidance_force_mag_1, linewidth=2, label='Robot 1', color='#1f77b4')
# plt.plot(t[:-1], static_avoidance_force_mag_2, linewidth=2, label='Robot 2', color='#ff7f0e')
# plt.plot(t[:-1], static_avoidance_force_mag_3, linewidth=2, label='Robot 3', color='#2ca02c')
# plt.plot(t[:-1], static_avoidance_force_mag_4, linewidth=2, label='Robot 4', color='#d62728')
# plt.plot(t[:-1], static_avoidance_force_mag_5, linewidth=2, label='Robot 5', color='#9467bd')

# plt.grid(True)
# plt.xlabel('Time (s)', fontsize=12)
# plt.ylabel('Static Obstacle Avoidance Force Magnitude', fontsize=12)
# plt.title('Pure Simulation: Static Obstacle Avoidance Force Over Time', fontsize=14)
# plt.legend()

# # 计算并显示静态障碍物避障力统计信息
# print("\n=== 纯仿真静态障碍物避障力统计 ===")
# for i, (robot_name, force_data) in enumerate([
#     ('Robot 1', static_avoidance_force_mag_1),
#     ('Robot 2', static_avoidance_force_mag_2), 
#     ('Robot 3', static_avoidance_force_mag_3),
#     ('Robot 4', static_avoidance_force_mag_4),
#     ('Robot 5', static_avoidance_force_mag_5)
# ], 1):
#     max_force = np.max(force_data)
#     avg_force = np.mean(force_data)
#     nonzero_count = np.count_nonzero(force_data)
#     activation_rate = (nonzero_count / len(force_data)) * 100
#     print(f"{robot_name}: 最大={max_force:.4f}, 平均={avg_force:.4f}, 激活率={activation_rate:.2f}%")

# # 图24：机器人间避障力
# plt.figure(24, figsize=(12, 8))
# plt.plot(t[:-1], robot_avoidance_force_mag_1, linewidth=2, label='Robot 1', color='#1f77b4')
# plt.plot(t[:-1], robot_avoidance_force_mag_2, linewidth=2, label='Robot 2', color='#ff7f0e')
# plt.plot(t[:-1], robot_avoidance_force_mag_3, linewidth=2, label='Robot 3', color='#2ca02c')
# plt.plot(t[:-1], robot_avoidance_force_mag_4, linewidth=2, label='Robot 4', color='#d62728')
# plt.plot(t[:-1], robot_avoidance_force_mag_5, linewidth=2, label='Robot 5', color='#9467bd')

# plt.grid(True)
# plt.xlabel('Time (s)', fontsize=12)
# plt.ylabel('Robot-Robot Avoidance Force Magnitude', fontsize=12)
# plt.title('Pure Simulation: Robot-Robot Avoidance Force Over Time', fontsize=14)
# plt.legend()

# # 计算并显示机器人间避障力统计信息
# print("\n=== 纯仿真机器人间避障力统计 ===")
# for i, (robot_name, force_data) in enumerate([
#     ('Robot 1', robot_avoidance_force_mag_1),
#     ('Robot 2', robot_avoidance_force_mag_2), 
#     ('Robot 3', robot_avoidance_force_mag_3),
#     ('Robot 4', robot_avoidance_force_mag_4),
#     ('Robot 5', robot_avoidance_force_mag_5)
# ], 1):
#     max_force = np.max(force_data)
#     avg_force = np.mean(force_data)
#     nonzero_count = np.count_nonzero(force_data)
#     activation_rate = (nonzero_count / len(force_data)) * 100
#     print(f"{robot_name}: 最大={max_force:.4f}, 平均={avg_force:.4f}, 激活率={activation_rate:.2f}%")

if ENABLE_DCBF:
    print("\n=== Full MPC-D-CBF求解统计 ===")
    print(
        f"预测步数={FULL_MPC_HORIZON_STEPS}, "
        f"共享静态障碍物数={len(dcbf_obstacles)}, "
        f"车体安全裕量={DCBF_SAFE_MARGIN:.3f}m, "
        f"参考协方差膨胀=开启（运动跟踪=关闭）"
    )
    for robot_idx in range(Rnum):
        active_obs_data = dcbf_mpc_active_obstacle_history[robot_idx, :]
        success_data = dcbf_mpc_success_history[robot_idx, :]
        min_h_data = dcbf_mpc_min_h_history[robot_idx, :]
        active_steps = np.count_nonzero(active_obs_data)
        success_steps = np.count_nonzero(success_data)
        success_rate = success_steps / len(i) * 100
        finite_h = min_h_data[np.isfinite(min_h_data)]
        min_h = np.min(finite_h) if finite_h.size > 0 else np.nan
        print(
            f"Robot {robot_idx + 1}: MPC激活步数={active_steps}, "
            f"成功率={success_rate:.2f}%, MPC窗口最小h={min_h:.4f}"
        )

# 图25：观测器对领导者X坐标的跟踪效果
plt.figure(25, figsize=(12, 8))
# 绘制领导者真实X坐标
plt.plot(t, x0[0, :], 'k--', linewidth=3, label='Leader X (True)', zorder=10)
# 绘制各机器人观测器估计的领导者X坐标
plt.plot(t, zxeo_1[0, :], linewidth=2, label='Robot 1 Observer', alpha=0.8)
plt.plot(t, zxeo_2[0, :], linewidth=2, label='Robot 2 Observer', alpha=0.8)
plt.plot(t, zxeo_3[0, :], linewidth=2, label='Robot 3 Observer', alpha=0.8)
plt.plot(t, zxeo_4[0, :], linewidth=2, label='Robot 4 Observer', alpha=0.8)
plt.plot(t, zxeo_5[0, :], linewidth=2, label='Robot 5 Observer', alpha=0.8)

plt.grid(True, alpha=0.3)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('X Coordinate (m)', fontsize=12)
plt.title('Observer Tracking of Leader X Coordinate', fontsize=14)
plt.legend(loc='best')
plt.xlim([0, k*t1])

# 计算并显示X坐标观测器跟踪误差统计
print("\n=== 观测器X坐标跟踪误差统计 ===")
for i, (robot_name, zxeo) in enumerate([
    ('Robot 1', zxeo_1),
    ('Robot 2', zxeo_2), 
    ('Robot 3', zxeo_3),
    ('Robot 4', zxeo_4),
    ('Robot 5', zxeo_5)
], 1):
    error = zxeo[0, :] - x0[0, :]
    max_error = np.max(np.abs(error))
    final_error = np.abs(error[-1])
    mean_error = np.mean(np.abs(error))
    print(f"{robot_name}: 最大误差={max_error:.4f}m, 最终误差={final_error:.4f}m, 平均误差={mean_error:.4f}m")

# 图26：观测器对领导者Y坐标的跟踪效果
plt.figure(26, figsize=(12, 8))
# 绘制领导者真实Y坐标
plt.plot(t, y0[0, :], 'k--', linewidth=3, label='Leader Y (True)', zorder=10)
# 绘制各机器人观测器估计的领导者Y坐标
plt.plot(t, zxeo_1[1, :], linewidth=2, label='Robot 1 Observer', alpha=0.8)
plt.plot(t, zxeo_2[1, :], linewidth=2, label='Robot 2 Observer', alpha=0.8)
plt.plot(t, zxeo_3[1, :], linewidth=2, label='Robot 3 Observer', alpha=0.8)
plt.plot(t, zxeo_4[1, :], linewidth=2, label='Robot 4 Observer', alpha=0.8)
plt.plot(t, zxeo_5[1, :], linewidth=2, label='Robot 5 Observer', alpha=0.8)

plt.grid(True, alpha=0.3)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Y Coordinate (m)', fontsize=12)
plt.title('Observer Tracking of Leader Y Coordinate', fontsize=14)
plt.legend(loc='best')
plt.xlim([0, k*t1])

# 计算并显示Y坐标观测器跟踪误差统计
print("\n=== 观测器Y坐标跟踪误差统计 ===")
for i, (robot_name, zxeo) in enumerate([
    ('Robot 1', zxeo_1),
    ('Robot 2', zxeo_2), 
    ('Robot 3', zxeo_3),
    ('Robot 4', zxeo_4),
    ('Robot 5', zxeo_5)
], 1):
    error = zxeo[1, :] - y0[0, :]
    max_error = np.max(np.abs(error))
    final_error = np.abs(error[-1])
    mean_error = np.mean(np.abs(error))
    print(f"{robot_name}: 最大误差={max_error:.4f}m, 最终误差={final_error:.4f}m, 平均误差={mean_error:.4f}m")

print_and_save_full_mpc_diagnostics()
save_trajectory_animation()

if SHOW_PLOTS:
    plt.show()
else:
    plt.close('all')
