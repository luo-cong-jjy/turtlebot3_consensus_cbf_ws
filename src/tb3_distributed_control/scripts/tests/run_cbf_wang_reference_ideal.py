#!/usr/bin/env python3
"""Offline ideal-state diagnostic, NOT a realtime/perception/full-paper claim."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import yaml

PACKAGE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PACKAGE / 'scripts'), str(PACKAGE / '纯仿真代码')]
from cbf_mpc_core import default_config, nearest_line_reference
from evaluate_cbf_active_motion import evaluate_motion
from mpc_dcbf_cached_solver import MPCSolverManager, CachedMPCProblem, rollout


def run(method, output):
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    sources = out / 'sources'; sources.mkdir()
    relatives = ['scripts/cbf_mpc_core.py', '纯仿真代码/mpc_dcbf_cached_solver.py',
                 'scripts/evaluate_cbf_active_motion.py', 'scripts/tests/run_cbf_wang_reference_ideal.py',
                 'config/cbf_active_avoidance_scenarios.yaml', 'config/cbf_active_avoidance_protocol.yaml']
    manifest = {}
    for relative in relatives:
        path = PACKAGE / relative; shutil.copy2(path, sources / path.name)
        manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = yaml.safe_load((PACKAGE / relatives[-1]).read_text())
    spec = yaml.safe_load((PACKAGE / relatives[-2]).read_text())['active_overtaking'][0]
    cfg = default_config()
    cfg.update(realtime=True, nlp_wall_time=.4, manager_wall_time=.48, execution_backup=True)
    if method != 'legacy':
        cfg.update(horizon=25, suppress_guidance=True, prediction_risk=True,
                   constraint_mode='ellipse', risk_penalty='wang_inverse', risk_weight=.3)
    nearest = method == 'wang_nearest'
    (out / 'run_metadata.json').write_text(json.dumps(dict(
        method=method, simulator='ideal_discrete_unicycle', forecast='exact_prescribed_constant_velocity',
        realtime_claim=False, offline_budget_seconds=.5, configuration=cfg, protocol=protocol,
        nearest_line_reference=nearest, source_sha256=manifest,
        scope='Wang mechanism adaptation; original nonlinear joint MPC, not complete paper reproduction'), indent=2))
    dt = cfg['dt']; n = cfg['horizon']; duration = protocol['duration']
    state = np.array([.4, 0., 0.]); previous = np.array([.15, 0.]); last = None
    velocity = np.array(spec['velocity']); start = np.array(spec['start'])
    radius = spec['radius'] + .02 + .35 * np.linalg.norm(velocity)

    def obstacle(t):
        centers = start + (t + dt * np.arange(n + 1))[:, None] * velocity
        return dict(name='ideal_actor', center=centers[0], a=radius, b=radius, theta=0.,
                    dynamic=True, guidance=False, prediction_model='cv',
                    prediction_centers=centers, prediction_axes=np.full((n + 1, 2), radius))

    manager = MPCSolverManager(cfg)
    problem = CachedMPCProblem(cfg, 1, False)
    zero = np.zeros((n, 2))
    problem.solve(state, np.tile(state, (n + 1, 1)), [obstacle(0)], [-.22, .22], None, zero, zero)
    manager.problems[(1, False)] = problem
    rows = []; truth = []; faults = 0
    with (out / 'solver_trace.jsonl').open('x') as traces:
        for k in range(round(duration / dt)):
            t = k * dt
            nominal = np.zeros((n + 1, 3)); nominal[:, 0] = .4 + .15 * (t + dt * np.arange(n + 1))
            goals = nearest_line_reference(state, nominal, dt) if nearest else nominal.copy()
            begin = time.perf_counter()
            try:
                answer = manager.solve(state, goals, [obstacle(t)], [-.22, .22], last, 'tb3_0',
                                       deadline=begin + .5, applied_previous=previous, cycle_token=k)
                control = answer['controls'][0].copy(); last = answer
                status = 'ok' if answer['success'] else 'feasible_iterate' if answer['validated_steps'] == n else 'validated_one_step'
            except RuntimeError:
                control = np.zeros(2); status = 'solver_failed_stop'
                if protocol['motion_window'][0] <= t <= protocol['motion_window'][1]: faults += 1
            rows.append(dict(time=t, x=state[0], y=state[1], theta=state[2], v=control[0], omega=control[1],
                             nominal_goal_x=nominal[0, 0], nominal_goal_y=nominal[0, 1],
                             local_goal_x=goals[0, 0], local_goal_y=goals[0, 1],
                             consensus_error_x=state[0] - nominal[0, 0], consensus_error_y=state[1] - nominal[0, 1],
                             solve_ms=1000 * (time.perf_counter() - begin), status=status))
            truth.append(state[:2].copy())
            traces.write(json.dumps(dict(time=t, manager=manager.last_records.get('tb3_0', {}))) + '\n')
            state = rollout(state, control[None, :], dt)[1]; previous = control
            if k % 100 == 0: print(method, 'task_time', round(t, 1), flush=True)
    truth.append(state[:2].copy()); times = np.arange(len(truth)) * dt
    actors = start + times[:, None] * velocity
    report = evaluate_motion(times, truth, actors, spec['radius'], protocol, faults)
    report.update(method=method, position_rmse=float(np.sqrt(np.mean([
        r['consensus_error_x'] ** 2 + r['consensus_error_y'] ** 2 for r in rows]))),
        omega_total_variation=float(np.abs(np.diff([r['omega'] for r in rows])).sum()),
        statuses={s: sum(r['status'] == s for r in rows) for s in set(r['status'] for r in rows)},
        diagnostic='offline ideal state/forecast, not Gazebo or realtime', source_sha256=manifest)
    with (out / 'tb3_0.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out / 'ideal_truth.csv').open('x', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['time', 'ego_x', 'ego_y', 'actor_x', 'actor_y'])
        writer.writerows(np.column_stack((times, truth, actors)))
    (out / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ['method', 'passed', 'failed_checks', 'minimum_rolling_speed',
                                            'longest_standstill_seconds', 'final_position_error']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['legacy', 'wang_timed', 'wang_nearest'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.method, args.output)
