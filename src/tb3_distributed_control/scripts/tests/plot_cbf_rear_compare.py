#!/usr/bin/env python3
"""Single-robot actual Gazebo paths plus unchanged nominal-error/control plots."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    prefix=Path(a.output)
    if any(prefix.with_suffix(s).exists() for s in ('.png','.pdf')):raise FileExistsError(prefix)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for directory,label,color in [(a.baseline,'Existing guide','C0'),(a.candidate,'Rear-pass candidate','C1')]:
        d=Path(directory);truth=list(csv.DictReader((d/'gazebo_truth.csv').open()));rows=list(csv.DictReader((d/'tb3_0.csv').open()))
        t=np.array([float(r['time']) for r in truth]);xy=np.array([[float(r['tb3_0_x']),float(r['tb3_0_y'])] for r in truth])
        axes[0].plot(xy[:,0],xy[:,1],label=label,color=color,lw=1.3)
        for stamp in (8,12,16):
            point=np.array([np.interp(stamp,t,xy[:,j]) for j in (0,1)]);axes[0].plot(*point,'o',color=color,ms=3)
        measure=np.array([float(r['evaluation_time']) for r in rows]);err=np.array([[float(r['consensus_error_x']),float(r['consensus_error_y'])] for r in rows])
        axes[1].plot(measure,np.linalg.norm(err,axis=1),color=color,label=label,lw=1.3)
        axes[2].plot(measure,[float(r['omega']) for r in rows],color=color,lw=1.1)
        if directory==a.baseline:
            spec=json.loads((d/'dynamic_scenario.json').read_text())['specifications'][0];name=spec['name']
            actor=np.array([[float(r[name+'_x']),float(r[name+'_y'])] for r in truth]);keep=(t>=4)&(t<=20)
            axes[0].plot(actor[keep,0],actor[keep,1],'--',color='.5',lw=.8,label='Actor (baseline run)')
            for stamp in (8,12,16):
                point=[np.interp(stamp,t,actor[:,j]) for j in (0,1)];axes[0].add_patch(Circle(point,spec['radius'],color='.6',alpha=.2))
                axes[0].annotate(str(stamp)+'s',point,fontsize=7)
    axes[0].plot([.4,6.4],[0,0],'k:',lw=.8,label='Nominal reference');axes[0].set_aspect('equal',adjustable='box');axes[0].set_xlim(.3,3.8);axes[0].set_ylim(-.9,1.5)
    axes[0].set_xlabel('x (m)');axes[0].set_ylabel('y (m)');axes[0].set_title('Actual paths: interaction region');axes[0].legend(fontsize=7)
    axes[1].set_ylabel('Nominal position error (m)');axes[2].set_ylabel('omega (rad/s)')
    for ax in axes[1:]:ax.set_xlabel('Measurement-aligned time (s)');ax.set_xlim(0,40)
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout();prefix.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.pdf'):fig.savefig(prefix.with_suffix(suffix),dpi=180)
    plt.close(fig)
