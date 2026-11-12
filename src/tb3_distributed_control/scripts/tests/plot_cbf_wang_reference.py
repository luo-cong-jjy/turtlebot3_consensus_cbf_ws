#!/usr/bin/env python3
"""Keep ideal diagnostics and actual Gazebo trajectories visibly separate."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw(root, output):
    root = Path(root); prefix = Path(output)
    if any(prefix.with_suffix(s).exists() for s in ('.png', '.pdf')): raise FileExistsError(prefix)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    groups = [[('ideal_legacy', 'Original guided MPC', 'C0'),
               ('ideal_wang_timed', 'Wang core / timed', 'C1'),
               ('ideal_wang_nearest', 'Wang core / nearest', 'C2')],
              [('gazebo_wang_timed', 'Wang core / timed', 'C1'),
               ('gazebo_wang_nearest_line', 'Wang core / nearest with abstention', 'C2')]]
    for index, group in enumerate(groups):
        trajectory_ax, speed_ax = axes[index]
        for directory, label, color in group:
            path = root / directory
            meta = json.loads((path / 'run_metadata.json').read_text())
            protocol = meta['protocol'] if index == 0 else meta['active_protocol']
            file = 'ideal_truth.csv' if index == 0 else 'gazebo_truth.csv'
            rows = list(csv.DictReader((path / file).open()))
            t = np.array([float(r['time']) for r in rows])
            xkey, ykey = ('ego_x', 'ego_y') if index == 0 else ('tb3_0_x', 'tb3_0_y')
            ego = np.array([[float(r[xkey]), float(r[ykey])] for r in rows])
            trajectory_ax.plot(ego[:, 0], ego[:, 1], lw=1.2, color=color, label=label)
            movement = np.linalg.norm(np.diff(ego, axis=0), axis=1)
            cumulative = np.r_[0., np.cumsum(movement)]
            lo, hi = protocol['motion_window']; window = protocol['rolling_window']
            grid = np.arange(lo + window, hi + 1e-8, .1)
            speed = (np.interp(grid, t, cumulative) - np.interp(grid - window, t, cumulative)) / window
            speed_ax.plot(grid, speed, color=color, lw=1.2, label=label)
        trajectory_ax.plot([.4, 6.4], [0, 0], 'k:', lw=1, label='Original timed path')
        trajectory_ax.set(xlabel='x (m)', ylabel='y (m)', ylim=(-1., 1.))
        trajectory_ax.set_aspect('equal', adjustable='datalim')
        speed_ax.axhline(protocol['minimum_rolling_speed'], color='C3', ls='--', lw=1, label='Acceptance minimum')
        speed_ax.set(xlabel='Task time (s)', ylabel='1 s travelled speed (m/s)', xlim=(lo, hi), ylim=(0, .25))
        title = 'Ideal state / exact CV / offline budget' if index == 0 else 'Actual Gazebo / own scan + CV / 100 ms budget'
        trajectory_ax.set_title(title, fontsize=10); speed_ax.set_title(title, fontsize=10)
        for ax in (trajectory_ax, speed_ax):
            ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Wang reference mechanism adaptation: all five runs FAIL active acceptance', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    for suffix in ('.png', '.pdf'): fig.savefig(prefix.with_suffix(suffix), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory'); parser.add_argument('--output', required=True)
    args = parser.parse_args(); draw(args.directory, args.output)
