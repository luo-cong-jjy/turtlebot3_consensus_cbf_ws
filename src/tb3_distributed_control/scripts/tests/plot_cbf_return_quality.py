#!/usr/bin/env python3
"""Measured return trajectories and errors, no smoothing or offline witness."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw(root,output):
    out=Path(output)
    if any(out.with_suffix(s).exists() for s in ('.png','.pdf')):raise FileExistsError(out)
    fig,axes=plt.subplots(3,3,figsize=(15,9))
    for row,(name,title) in enumerate([('head_on','Head-on'),('oblique','Oblique crossing'),('overtaking','Overtaking')]):
        maximum=0.
        for suffix,color,label,style in [('baseline','0.5','Original return','--'),('new','C0','New return','-')]:
            p=root/(name+'_'+suffix);r=list(csv.DictReader((p/'gazebo_truth.csv').open()))
            t=np.array([float(s['time']) for s in r]);xy=np.array([[float(s['tb3_0_x']),float(s['tb3_0_y'])] for s in r])
            meta=json.loads((p/'run_metadata.json').read_text());proto=meta['active_protocol']
            target=proto['reference_start']+t[:,None]*np.asarray(proto['reference_velocity'])
            error=xy-target;norm=np.linalg.norm(error,axis=1);maximum=max(maximum,float(abs(error[:,1]).max()))
            q=json.loads((p/'return_quality.json').read_text())
            axes[row,0].plot(xy[:,0],xy[:,1],style,color=color,lw=1.2,label=label)
            axes[row,1].plot(t,abs(error[:,1]),style,color=color,lw=1.2,label=label)
            axes[row,2].plot(t,norm,style,color=color,lw=1.2,label=label)
            if q['first_return']:
                stamp=q['first_return']['time'];point=np.array([np.interp(stamp,t,xy[:,j]) for j in (0,1)])
                axes[row,0].plot(*point,'*',color=color,ms=8)
            if q['settled_xy_time'] is not None:
                axes[row,2].axvline(q['settled_xy_time'],color=color,ls=':',lw=1)
            if suffix=='baseline':
                actor_name=json.loads((p/'dynamic_scenario.json').read_text())['specifications'][0]['name']
                actor=np.array([[float(s[actor_name+'_x']),float(s[actor_name+'_y'])] for s in r])
                visible=abs(actor[:,1])<=max(1.,maximum+.1)
                axes[row,0].plot(actor[visible,0],actor[visible,1],':',color='C2',lw=.9,label='Measured obstacle')
        axes[row,0].plot([.4,6.4],[0,0],'k:',lw=.8,label='Timed nominal path')
        axes[row,0].set(title=title+' (stars: return starts)',xlabel='x (m)',ylabel='y (m)',ylim=(-maximum-.1,maximum+.1))
        axes[row,0].set_aspect('equal',adjustable='datalim')
        for j,ylabel in [(1,'Absolute lateral error (m)'),(2,'Timed-target position error (m)')]:
            axes[row,j].axhspan(0,.05,color='C2',alpha=.1)
            axes[row,j].axhline(.05,color='C2',lw=.8,ls=':')
            axes[row,j].set(xlim=(0,30),ylim=(0,None),xlabel='Task time (s)',ylabel=ylabel)
        for ax in axes[row]:ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);axes[0,1].legend(fontsize=8)
    fig.suptitle('Actual Gazebo measurements: return to the unchanged timed target',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96))
    for s in ('.png','.pdf'):fig.savefig(out.with_suffix(s),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',required=True)
    args=p.parse_args();draw(args.root,args.output)
