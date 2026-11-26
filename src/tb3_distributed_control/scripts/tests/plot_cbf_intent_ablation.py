#!/usr/bin/env python3
"""Offline head-on exchange/local-only comparison; truth is evaluation only."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot(root):
    root=Path(root);output=root/'figures';output.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(15,7.2),constrained_layout=True)
    cases=[('V1_head_on_211','Intent exchange'),('V1_head_on_local_only_211','Local references only')]
    colors=plt.get_cmap('tab10').colors[:5]
    for row,(name,label) in enumerate(cases):
        case=root/name;meta=json.loads((case/'run_metadata.json').read_text())
        result=json.loads((case/'evaluation.json').read_text());proto=meta['active_protocol']
        spec=json.loads((case/'dynamic_scenario.json').read_text())['specifications'][0]
        records=list(csv.DictReader((case/'gazebo_truth.csv').open()));t=np.array([float(r['time']) for r in records])
        xy=np.array([[[float(r[f'tb3_{i}_x']),float(r[f'tb3_{i}_y'])] for r in records] for i in range(5)])
        offsets=np.asarray(proto['robot_offsets']);leader=np.array(proto['reference_start'])+t[:,None]*proto['reference_velocity']
        for i in range(5):
            nominal=leader+offsets[i];axes[row,0].plot(*xy[i].T,color=colors[i],lw=1.5,label=f'Robot {i}')
            axes[row,0].plot(*nominal.T,color=colors[i],lw=.65,ls='--',alpha=.35)
            distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy[i],axis=0),axis=1))]
            grid=np.arange(5,39.001,.1);speed=np.interp(grid,t,distance)-np.interp(grid-1,t,distance)
            axes[row,1].plot(grid,speed,color=colors[i],lw=1.1)
            axes[row,2].plot(t,np.linalg.norm(xy[i]-nominal,axis=1),color=colors[i],lw=1.1)
        for stamp in (0,12,25,39):
            k=int(np.argmin(abs(t-stamp)));pent=np.vstack((xy[:,k],xy[0,k]))
            axes[row,0].plot(*pent.T,color='.35',lw=.7,alpha=.6)
        actor=np.array([[float(r[spec['name']+'_x']),float(r[spec['name']+'_y'])] for r in records])
        axes[row,0].plot(*actor.T,color='black',ls='-.',lw=1.)
        for stamp in (5,10,15):
            k=int(np.argmin(abs(t-stamp)));axes[row,0].add_patch(Circle(actor[k],spec['radius'],color='gray',alpha=.22))
        pair=np.max(np.array([np.linalg.norm(xy[i]-xy[j]-(offsets[i]-offsets[j]),axis=1) for i in range(5) for j in range(i)]),axis=0)
        axes[row,2].plot(t,pair,color='black',ls='--',lw=1,label='Max pair-shape error')
        axes[row,0].set_title(label+': '+('PASS' if result['passed'] else 'FAIL'))
        axes[row,0].set(xlabel='x (m)',ylabel='y (m)');axes[row,0].set_aspect('equal',adjustable='datalim')
        axes[row,1].axhline(proto['minimum_rolling_speed'],color='black',ls='--',lw=.8)
        axes[row,1].set(xlabel='Time (s)',ylabel='1 s path speed (m/s)',xlim=(4,39),ylim=(0,.25))
        axes[row,2].set(xlabel='Time (s)',ylabel='Position / pair error (m)',xlim=(0,40))
        for ax in axes[row]:ax.grid(True,alpha=.2)
    axes[0,0].legend(fontsize=7,ncol=3);axes[0,2].legend(fontsize=7)
    fig.suptitle('Same local reference family, exchange on/off; one trial per setting',fontsize=12)
    for extension in ('png','pdf'):
        path=output/('head_on_intent_ablation.'+extension)
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=170)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args();plot(args.root)
