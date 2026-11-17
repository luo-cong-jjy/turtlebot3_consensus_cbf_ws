#!/usr/bin/env python3
"""Measured runtime-profile comparison and all preregistered final trials."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    truth=list(csv.DictReader((path/'gazebo_truth.csv').open()));t=np.array([float(r['time']) for r in truth])
    xy=np.array([[float(r['tb3_0_x']),float(r['tb3_0_y'])] for r in truth])
    d=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))];g=np.arange(5.,39.+1e-8,.1)
    velocity=np.interp(g,t,d)-np.interp(g-1,t,d)
    rows=list(csv.DictReader((path/'tb3_0.csv').open()))
    return t,xy,g,velocity,rows,json.loads((path/'evaluation.json').read_text())


def save(fig,path):
    if any(path.with_suffix(s).exists() for s in ('.png','.pdf')):raise FileExistsError(path)
    fig.tight_layout(rect=(0,0,1,.95))
    for suffix in ('.png','.pdf'):fig.savefig(path.with_suffix(suffix),dpi=180)
    plt.close(fig)


def draw(directory):
    root=Path(directory);fig,axes=plt.subplots(2,2,figsize=(12,7))
    for name,label,color in [('legacy_v5_current_211','Original V5 profile','C1'),
                              ('final_overtaking_211_r1','Native preview + checked continuity + reserve','C0')]:
        t,xy,g,v,rows,result=load(root/name)
        axes[0,0].plot(xy[:,0],xy[:,1],lw=1.2,label=label,color=color)
        axes[0,1].plot(g,v,lw=1.2,label=label,color=color)
        stamp=np.array([float(r['time']) for r in rows])
        axes[1,0].plot(stamp,[float(r['command_wall_ms']) for r in rows],lw=.8,label=label,color=color)
        axes[1,1].plot(stamp,[np.hypot(float(r['consensus_error_x']),float(r['consensus_error_y'])) for r in rows],lw=1.2,label=label,color=color)
    axes[0,0].set(xlabel='x (m)',ylabel='y (m)',title='Actual Gazebo trajectories');axes[0,0].set_aspect('equal',adjustable='datalim')
    axes[0,1].axhline(.075,color='C3',ls='--',lw=1);axes[0,1].set(xlim=(4,39),ylim=(0,.26),xlabel='Task time (s)',ylabel='1 s travelled speed (m/s)')
    axes[1,0].axhline(100,color='C3',ls='--',lw=1);axes[1,0].set(xlim=(0,40),xlabel='Task time (s)',ylabel='Measured time to command (ms)')
    axes[1,1].set(xlim=(0,40),xlabel='Task time (s)',ylabel='Original timed-leader error (m)')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Same scene/seed: complete runtime profiles, not a single-factor performance claim',fontsize=12)
    save(fig,root/'runtime_profile_comparison')

    cases=['final_overtaking_211_r1','final_overtaking_211_r2','final_overtaking_211_r3','final_overtaking_239',
           'final_overtaking_307','final_head_on_211','final_oblique_211']
    labels=['Overtake211 #1','Overtake211 #2','Overtake211 #3','Overtake239','Overtake307','Head-on211','Oblique211']
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for i,name in enumerate(cases):
        result=json.loads((root/name/'evaluation.json').read_text());a=result['active_motion']
        axes[0].scatter(i,a['minimum_rolling_speed'],s=35,color='C0' if result['passed'] else 'C3')
        axes[1].scatter(i,result['robot_statistics']['0']['timing_max_ms'],s=35,color='C0' if result['passed'] else 'C3')
    axes[0].axhline(.075,color='C3',ls='--',lw=1);axes[0].set(ylabel='Minimum 1 s speed (m/s)',ylim=(0,.2))
    axes[1].axhline(100,color='C3',ls='--',lw=1);axes[1].set(ylabel='Maximum recorded cycle (ms)',ylim=(0,120))
    for ax in axes:ax.set_xticks(range(len(labels)));ax.set_xticklabels(labels,rotation=35,ha='right',fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('All seven scheduled final trials; strict acceptance 7/7 (finite headless simulation only)',fontsize=11)
    save(fig,root/'final_trials_overview')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');a=p.parse_args();draw(a.directory)
