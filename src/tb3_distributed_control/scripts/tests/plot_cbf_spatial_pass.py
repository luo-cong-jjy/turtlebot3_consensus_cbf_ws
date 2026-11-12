#!/usr/bin/env python3
"""Measured trajectory/speed checks and a paired flag-off/on comparison."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def load(directory):
    rows = list(csv.DictReader((directory/'gazebo_truth.csv').open()))
    t = np.array([float(r['time']) for r in rows])
    ego = np.array([[float(r['tb3_0_x']),float(r['tb3_0_y'])] for r in rows])
    actor = np.array([[float(r['cbf_dynamic_0_x']),float(r['cbf_dynamic_0_y'])] for r in rows])
    distance = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(ego,axis=0),axis=1))]
    grid = np.arange(5.,39.+1e-8,.1)
    speed = np.interp(grid,t,distance)-np.interp(grid-1.,t,distance)
    report = json.loads((directory/'active_evaluation.json').read_text())
    return t,ego,actor,grid,speed,report


def save(fig,prefix):
    if any(prefix.with_suffix(s).exists() for s in ('.png','.pdf')):raise FileExistsError(prefix)
    fig.tight_layout(rect=(0,0,1,.96))
    for suffix in ('.png','.pdf'):fig.savefig(prefix.with_suffix(suffix),dpi=180)
    plt.close(fig)


def draw(root):
    root = Path(root)
    fig,axes = plt.subplots(3,2,figsize=(12,9))
    for (path_ax,speed_ax),(directory,title) in zip(axes,[('gazebo_overtaking_v2','Slow-target overtaking'),
            ('gazebo_head_on_v2','Head-on'),('gazebo_oblique_v2','Oblique crossing')]):
        t,ego,actor,grid,speed,report = load(root/directory)
        path_ax.plot(ego[:,0],ego[:,1],lw=1.2,label='Measured robot')
        shown = np.abs(actor[:,1])<1.
        path_ax.plot(actor[shown,0],actor[shown,1],'--',lw=1,color='C3',label='Measured obstacle')
        path_ax.plot([.4,6.4],[0,0],'k:',lw=1,label='Original timed path')
        event = report['first_forward_pass'];stamp = event['time']
        for positions,radius,color in [(ego,.13,'C0'),(actor,.18,'C3')]:
            point = [np.interp(stamp,t,positions[:,j]) for j in (0,1)]
            path_ax.add_patch(Circle(point,radius,color=color,alpha=.2))
        path_ax.set(xlabel='x (m)',ylabel='y (m)',ylim=(-.9,.9))
        path_ax.set_aspect('equal',adjustable='datalim');path_ax.set_title(title+' / active acceptance PASS',fontsize=10)
        speed_ax.plot(grid,speed,lw=1.2,label='1 s travelled speed')
        speed_ax.axhline(.075,color='C3',ls='--',lw=1,label='Frozen acceptance minimum')
        speed_ax.set(xlabel='Task time (s)',ylabel='Speed (m/s)',xlim=(4,39),ylim=(0,.26))
        speed_ax.set_title('Minimum = %.4f m/s; longest standstill = %.3f s' % (
            report['minimum_rolling_speed'],report['longest_standstill_seconds']),fontsize=10)
        for ax in (path_ax,speed_ax):ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);axes[0,1].legend(fontsize=7)
    fig.suptitle('Actual Gazebo / CV / original timed goals; circles are body proxies at passing',fontsize=12)
    save(fig,root/'three_scenarios_v2')

    fig,axes = plt.subplots(2,2,figsize=(12,7))
    for name,label,color in [('gazebo_overtaking_paired_baseline','Flag off / baseline','C1'),
                              ('gazebo_overtaking_v2','Flag on / spatial pass','C0')]:
        directory=root/name;t,ego,actor,grid,speed,report=load(directory)
        axes[0,0].plot(ego[:,0],ego[:,1],lw=1.2,label=label,color=color)
        axes[0,1].plot(grid,speed,lw=1.2,label=label,color=color)
        rows=list(csv.DictReader((directory/'tb3_0.csv').open()))
        stamp=np.array([float(r['time']) for r in rows]);omega=[float(r['omega']) for r in rows]
        error=[np.hypot(float(r['consensus_error_x']),float(r['consensus_error_y'])) for r in rows]
        axes[1,0].plot(stamp,omega,lw=.8,label=label,color=color)
        axes[1,1].plot(stamp,error,lw=1.2,label=label,color=color)
    axes[0,0].set(xlabel='x (m)',ylabel='y (m)',title='Actual trajectories')
    axes[0,0].set_aspect('equal',adjustable='datalim')
    axes[0,1].axhline(.075,color='C3',ls='--',lw=1)
    axes[0,1].set(xlabel='Task time (s)',ylabel='1 s travelled speed (m/s)',xlim=(4,39),ylim=(0,.26))
    axes[1,0].set(xlabel='Task time (s)',ylabel='Applied omega (rad/s)',xlim=(0,40))
    axes[1,1].set(xlabel='Task time (s)',ylabel='Original timed-leader error (m)',xlim=(0,40))
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Paired overtake: same scene/seed/bounds; original metrics retained',fontsize=12)
    save(fig,root/'overtaking_paired_v2')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory')
    args=parser.parse_args();draw(args.directory)
