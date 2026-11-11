#!/usr/bin/env python3
"""Plot actual Gazebo truth separately from offline feasibility witnesses."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw(directories, output):
    prefix = Path(output)
    if any(prefix.with_suffix(s).exists() for s in ('.png', '.pdf')):
        raise FileExistsError(prefix)
    fig, axes = plt.subplots(len(directories), 2, figsize=(12, 3 * len(directories)), squeeze=False)
    for (path_ax, speed_ax), directory in zip(axes, directories):
        directory = Path(directory)
        rows = list(csv.DictReader((directory / 'gazebo_truth.csv').open()))
        meta = json.loads((directory / 'run_metadata.json').read_text())
        protocol = meta['active_protocol']
        spec = json.loads((directory / 'dynamic_scenario.json').read_text())['specifications'][0]
        t = np.array([float(r['time']) for r in rows])
        ego = np.array([[float(r['tb3_0_x']), float(r['tb3_0_y'])] for r in rows])
        actor = np.array([[float(r[spec['name'] + '_x']), float(r[spec['name'] + '_y'])] for r in rows])
        path_ax.plot(ego[:, 0], ego[:, 1], lw=1.2, label='Measured robot')
        shown = np.abs(actor[:, 1]) <= 1.2
        path_ax.plot(actor[shown, 0], actor[shown, 1], '--', lw=1, label='Measured obstacle')
        path_ax.plot([.4, 6.4], [0, 0], 'k:', lw=1, label='Nominal reference')
        path_ax.set(xlabel='x (m)', ylabel='y (m)', ylim=(-1.2, 1.2))
        path_ax.set_aspect('equal', adjustable='datalim')
        movement = np.linalg.norm(np.diff(ego, axis=0), axis=1)
        cumulative = np.r_[0., np.cumsum(movement)]
        lo, hi = protocol['motion_window']
        window = protocol['rolling_window']
        grid = np.arange(lo + window, hi + 1e-8, .1)
        rolling = (np.interp(grid, t, cumulative) - np.interp(grid - window, t, cumulative)) / window
        speed_ax.plot((t[:-1] + t[1:]) / 2, movement / np.diff(t), color='0.7', lw=.6, label='Pose-difference speed')
        speed_ax.plot(grid, rolling, lw=1.2, label='1 s travelled speed')
        speed_ax.axhline(protocol['minimum_rolling_speed'], color='C3', ls='--', lw=1, label='Acceptance minimum')
        speed_ax.axhline(protocol['stop_speed'], color='0.3', ls=':', lw=1, label='Standstill threshold')
        speed_ax.set(xlim=(lo, hi), ylim=(0, .28), xlabel='Task time (s)', ylabel='Speed (m/s)')
        report_path = directory / 'active_evaluation.json'
        if not report_path.exists():
            report_path = directory / 'active_evaluation_replayed.json'
        report = json.loads(report_path.read_text())
        title = meta['dynamic_scenario'] + ': active acceptance ' + ('PASS' if report['passed'] else 'FAIL')
        path_ax.set_title(title, fontsize=10)
        speed_ax.set_title('Minimum 1 s speed = %.4f m/s' % report['minimum_rolling_speed'], fontsize=10)
        for ax in (path_ax, speed_ax):
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(fontsize=7)
    fig.suptitle('Actual Gazebo measurements (existing CV controller; NOT feasibility witness)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    for suffix in ('.png', '.pdf'):
        fig.savefig(prefix.with_suffix(suffix), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    draw(args.directories, args.output)
