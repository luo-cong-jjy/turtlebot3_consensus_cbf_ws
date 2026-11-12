#!/usr/bin/env python3
"""Post-hoc same-input symmetry/initialization diagnostic; no live commands."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import yaml

PACKAGE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PACKAGE / 'scripts'), str(PACKAGE / '纯仿真代码')]
from mpc_dcbf_cached_solver import CachedMPCProblem, rollout


def analyze(directory, output):
    out = Path(output)
    if out.exists(): raise FileExistsError(out)
    directory = Path(directory)
    metadata = json.loads((directory / 'run_metadata.json').read_text())
    kernel = PACKAGE / '纯仿真代码/mpc_dcbf_cached_solver.py'
    assert hashlib.sha256(kernel.read_bytes()).hexdigest() == metadata['source_sha256']['纯仿真代码/mpc_dcbf_cached_solver.py']
    cfg = dict(metadata['configuration'], nlp_wall_time=2., manager_wall_time=2.2)
    n = cfg['horizon']; dt = cfg['dt']
    spec = yaml.safe_load((directory / 'sources/cbf_active_avoidance_scenarios.yaml').read_text())['active_overtaking'][0]
    rows = list(csv.DictReader((directory / 'tb3_0.csv').open())); records = []
    for stamp in (4., 20.):
        index = min(range(len(rows)), key=lambda i: abs(float(rows[i]['time']) - stamp))
        row = rows[index]; t = float(row['time'])
        state = np.array([float(row[k]) for k in ('x', 'y', 'theta')])
        previous = np.array([float(rows[index - 1][k]) for k in ('v', 'omega')])
        goals = np.zeros((n + 1, 3)); goals[:, 0] = .4 + .15 * (t + dt * np.arange(n + 1))
        velocity = np.array(spec['velocity']); radius = spec['radius'] + .02 + .35 * np.linalg.norm(velocity)
        centers = np.array(spec['start']) + (t + dt * np.arange(n + 1))[:, None] * velocity
        obs = dict(name='ideal_actor', center=centers[0], a=radius, b=radius, theta=0., dynamic=True,
                   guidance=False, prediction_centers=centers, prediction_axes=np.full((n + 1, 2), radius))
        frame = dict(time=t, state=state.tolist(), previous=previous.tolist(), goals=goals.tolist(),
                     obstacle_centers=centers.tolist(), obstacle_radius=radius, attempts=[])
        problem = CachedMPCProblem(cfg, 1, False)
        for side in (0, 1, -1):
            controls = np.tile([.15 if side == 0 else .22, 0.], (n, 1))
            if side:
                controls[:10, 1] = side * .8; controls[10:20, 1] = -side * .8
            positions = rollout(state, controls, dt)
            plan = controls[:, :1] * np.column_stack((np.cos(positions[1:, 2]), np.sin(positions[1:, 2])))
            begin = time.perf_counter()
            try:
                result = problem.solve(state, goals, [obs], [-.22, .22], previous, controls, plan)
                valid = problem.validate(result, state, [obs], [-.22, .22], previous)
                record = dict(seed=side, valid=bool(valid), cost=float(result['cost']),
                              first_control=result['controls'][0].tolist(), states=result['states'].tolist(),
                              max_lateral=float(np.abs(result['states'][:, 1]).max()),
                              solver_status=result.get('status'))
            except RuntimeError as exc:
                record = dict(seed=side, valid=False, error=str(exc).splitlines()[-1])
            record['wall_ms'] = 1000 * (time.perf_counter() - begin); frame['attempts'].append(record)
        records.append(frame)
    out.write_text(json.dumps(dict(source_directory=str(directory.resolve()),
        analysis_type='post-hoc ideal-state same-input local-solver diagnostic, NOT closed-loop or realtime',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), configuration=cfg, frames=records), indent=2))
    for frame in records:
        print(frame['time'], [{k: r.get(k) for k in ('seed', 'valid', 'cost', 'first_control', 'max_lateral')} for r in frame['attempts']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory'); parser.add_argument('--output', required=True)
    args = parser.parse_args(); analyze(args.directory, args.output)
