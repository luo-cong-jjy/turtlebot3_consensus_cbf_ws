#!/usr/bin/env python3
"""Actual two-robot trajectories, continuous motion and formation recovery."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw(directory,output):
    path=Path(directory);prefix=Path(output)
    if any(prefix.with_suffix(s).exists() for s in ('.png','.pdf')):raise FileExistsError(prefix)
    rows=list(csv.DictReader((path/'gazebo_truth.csv').open()))
    t=np.array([float(r['time']) for r in rows])
    positions=np.array([[[float(r['tb3_'+str(i)+'_x']),float(r['tb3_'+str(i)+'_y'])] for r in rows] for i in range(2)])
    actor=np.array([[float(r['cbf_dynamic_0_x']),float(r['cbf_dynamic_0_y'])] for r in rows])
    report=json.loads((path/'active_evaluation.json').read_text())
    meta=json.loads((path/'run_metadata.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    grid=np.arange(5.,39.+1e-8,.1)
    for i,color in enumerate(('C0','C1')):
        p=positions[i];label='Robot '+str(i)
        axes[0,0].plot(p[:,0],p[:,1],color=color,lw=1.2,label=label)
        axes[0,0].scatter(*p[0],color=color,s=16)
        distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        speed=np.interp(grid,t,distance)-np.interp(grid-1.,t,distance)
        axes[0,1].plot(grid,speed,color=color,lw=1.2,label=label)
        control=list(csv.DictReader((path/('tb3_'+str(i)+'.csv')).open()))
        time=[float(r.get('evaluation_time',r['time'])) for r in control]
        error=[np.hypot(float(r['consensus_error_x']),float(r['consensus_error_y'])) for r in control]
        axes[1,1].plot(time,error,color=color,lw=1.2,label=label)
    for stamp in (0,10,20,30,40):
        points=np.array([[np.interp(stamp,t,positions[i,:,j]) for j in range(2)] for i in range(2)])
        axes[0,0].plot(points[:,0],points[:,1],color='.5',lw=.7,alpha=.5)
    shown=abs(actor[:,1])<1.2
    axes[0,0].plot(actor[shown,0],actor[shown,1],'--',color='C3',lw=1,label='Measured obstacle')
    axes[0,0].plot([-.5,6.5],[0,0],'k:',lw=.8,label='Original formation centreline')
    axes[0,0].set(xlabel='x (m)',ylabel='y (m)',title='Actual paths; formation segments every 10 s')
    axes[0,0].set_aspect('equal',adjustable='datalim')
    axes[0,1].axhline(.075,color='C3',ls='--',lw=1,label='Frozen minimum')
    axes[0,1].set(xlim=(4,39),ylim=(0,.26),xlabel='Task time (s)',ylabel='1 s travelled speed (m/s)')
    peer=np.linalg.norm(positions[0]-positions[1],axis=1)-.26
    axes[1,0].plot(t,peer,lw=1.2,label='Circle proxy clearance')
    axes[1,0].axhline(0,color='C3',ls='--',lw=1)
    axes[1,0].set(xlim=(0,40),xlabel='Task time (s)',ylabel='Robot-pair proxy clearance (m)',
        title='Final relative formation error = %.4f m'%report['final_relative_formation_error'])
    axes[1,1].set(xlim=(0,40),xlabel='Task time (s)',ylabel='Original timed formation error (m)')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle(meta['dynamic_scenario']+' / independent team acceptance '+('PASS' if report['passed'] else 'FAIL'),fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95))
    for suffix in ('.png','.pdf'):fig.savefig(prefix.with_suffix(suffix),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('--output',required=True)
    a=p.parse_args();draw(a.directory,a.output)
