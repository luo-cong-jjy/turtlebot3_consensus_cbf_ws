#!/usr/bin/env python3
"""Plot measured nominal formation and retained rejected compactness trials."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def truth(path,count):
    rows=list(csv.DictReader((path/'gazebo_truth.csv').open()));t=np.array([float(r['time']) for r in rows])
    points=np.array([[[float(r['tb3_%d_x'%i]),float(r['tb3_%d_y'%i])] for r in rows] for i in range(count)])
    return t,points


def save(fig,prefix):
    if any(prefix.with_suffix(s).exists() for s in ('.png','.pdf')):raise FileExistsError(prefix)
    fig.tight_layout(rect=(0,0,1,.95))
    for s in ('.png','.pdf'):fig.savefig(prefix.with_suffix(s),dpi=180)
    plt.close(fig)


def draw(root,suffix=''):
    root=Path(root);t,p=truth(root/'pentagon_nominal_v4',5)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for i in range(5):axes[0].plot(p[i,:,0],p[i,:,1],lw=1,label='Robot '+str(i))
    for stamp in (0,10,20,30):
        q=np.array([[np.interp(stamp,t,p[i,:,j]) for j in range(2)] for i in range(5)])
        q=np.vstack((q,q[0]));axes[0].plot(q[:,0],q[:,1],color='.3',lw=.7,alpha=.65)
    axes[0].set_aspect('equal',adjustable='datalim');axes[0].set(xlabel='x (m)',ylabel='y (m)',title='Measured formation, R=0.4 m; polygons every 10 s')
    offsets=.4*np.column_stack((np.cos(2*np.pi*np.arange(5)/5),np.sin(2*np.pi*np.arange(5)/5)))
    errors=np.array([np.linalg.norm(p[i]-p[j]-offsets[i]+offsets[j],axis=1) for i in range(5) for j in range(i)])
    selected=(t>=5)&(t<=29)
    axes[1].plot(t[selected],1000*errors.max(axis=0)[selected],lw=1.1,label='Maximum pair-vector error')
    axes[1].set(xlim=(5,29),xlabel='Task time (s)',ylabel='Relative shape error (mm)',title='Post-startup shape maintenance')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Five robots: nominal empty-world tracking PASS; not dynamic five-robot avoidance',fontsize=11)
    save(fig,root/('pentagon_nominal_result'+suffix))
    fig,axes=plt.subplots(3,2,figsize=(12,9))
    for (xy_ax,v_ax),scene in zip(axes,['overtaking','head_on','oblique']):
        for kind,label,color in [('legacy','Original reference extras','C1'),('moderate','Smaller extras EXPERIMENT','C0')]:
            path=root/(kind+'_'+scene+'_211');t,q=truth(path,1);point=q[0]
            report=json.loads((path/'evaluation.json').read_text());xy_ax.plot(point[:,0],point[:,1],color=color,lw=1.1,label=label)
            distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(point,axis=0),axis=1))];g=np.arange(5,39.001,.1)
            velocity=np.interp(g,t,distance)-np.interp(g-1,t,distance)
            v_ax.plot(g,velocity,color=color,lw=1.1,label=label+' '+('PASS' if report['passed'] else 'FAIL'))
        xy_ax.set_aspect('equal',adjustable='datalim');xy_ax.set(xlabel='x (m)',ylabel='y (m)',title=scene)
        v_ax.axhline(.075,color='C3',ls='--',lw=1);v_ax.set(xlim=(4,39),ylim=(0,.26),xlabel='Task time (s)',ylabel='1 s travelled speed (m/s)')
        for ax in (xy_ax,v_ax):ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Smaller planning extras: REJECTED, continuity/compactness acceptance not met',fontsize=12)
    save(fig,root/('compactness_negative_comparison'+suffix))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('--suffix',default='');a=p.parse_args();draw(a.directory,a.suffix)
