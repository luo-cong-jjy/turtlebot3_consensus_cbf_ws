#!/usr/bin/env python3
"""Offline five-robot truth plots; never publishes to a controller."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot(root,suffix=""):
    root=Path(root);cases=['V1_overtaking_211','V1_head_on_211','V1_oblique_crossing_211']
    cases=[case+suffix for case in cases]
    labels=['Overtaking','Head-on','Oblique crossing'];colors=plt.get_cmap('tab10').colors[:5]
    fig,axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
    for row,(case,label) in enumerate(zip(cases,labels)):
        path=root/case;result=json.loads((path/'evaluation.json').read_text());meta=json.loads((path/'run_metadata.json').read_text())
        spec=json.loads((path/'dynamic_scenario.json').read_text())['specifications'][0]
        rows=list(csv.DictReader((path/'gazebo_truth.csv').open()));t=np.array([float(r['time']) for r in rows])
        positions=np.array([[[float(r[f'tb3_{i}_x']),float(r[f'tb3_{i}_y'])] for r in rows] for i in range(5)])
        actor=np.array([[float(r[spec['name']+'_x']),float(r[spec['name']+'_y'])] for r in rows])
        proto=meta['active_protocol'];offsets=np.asarray(proto['robot_offsets']);leader=np.array(proto['reference_start'])+t[:,None]*proto['reference_velocity']
        ax,sp,er=axes[row]
        for i in range(5):
            xy=positions[i];nominal=leader+offsets[i]
            ax.plot(xy[:,0],xy[:,1],color=colors[i],lw=1.5,label=f'Robot {i}')
            ax.plot(nominal[:,0],nominal[:,1],color=colors[i],lw=.65,ls='--',alpha=.35)
            cumulative=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
            grid=np.arange(5,39.001,.1);speed=np.interp(grid,t,cumulative)-np.interp(grid-1,t,cumulative)
            sp.plot(grid,speed,color=colors[i],lw=1.1)
            er.plot(t,np.linalg.norm(xy-nominal,axis=1),color=colors[i],lw=1.1)
        for stamp in (0,12,25,39):
            k=int(np.argmin(abs(t-stamp)));pent=positions[:,k];pent=np.vstack([pent,pent[0]])
            ax.plot(pent[:,0],pent[:,1],color='.35',lw=.7,alpha=.6)
            ax.text(pent[0,0],pent[0,1]+.07,f'{stamp}s',fontsize=7,color='.3')
        ax.plot(actor[:,0],actor[:,1],color='black',ls='-.',lw=1,label='Actor centre')
        for stamp in (5,10,15):
            k=int(np.argmin(abs(t-stamp)));ax.add_patch(Circle(actor[k],spec['radius'],color='gray',alpha=.22))
        pair_error=np.max(np.array([np.linalg.norm(positions[i]-positions[j]-(offsets[i]-offsets[j]),axis=1) for i in range(5) for j in range(i)]),axis=0)
        er.plot(t,pair_error,color='black',ls='--',lw=1,label='Max pair-shape error')
        state='PASS' if result['passed'] else 'FAIL'
        ax.set_title(f'{label}: {state}, R = 0.4 m');ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
        sp.axhline(proto['minimum_rolling_speed'],color='black',ls='--',lw=.8);sp.set(xlabel='Time (s)',ylabel='1 s path speed (m/s)',xlim=(4,39),ylim=(0,.25))
        er.set(xlabel='Time (s)',ylabel='Position / pair-shape error (m)',xlim=(0,40))
        for a in (ax,sp,er):a.grid(True,alpha=.2)
    axes[0,0].legend(fontsize=7,ncol=3,loc='upper right');axes[0,2].legend(fontsize=7)
    fig.suptitle('Five independent controllers: fixed safety settings; pentagon lines are actual snapshots',fontsize=12)
    for suffix in ('png','pdf'):
        out=root/('five_robot_dynamic_overview.'+suffix)
        if out.exists():raise FileExistsError(out)
        fig.savefig(out,dpi=170)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--suffix',default='');args=parser.parse_args();plot(args.root,args.suffix)
