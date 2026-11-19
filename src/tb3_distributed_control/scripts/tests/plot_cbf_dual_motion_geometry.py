#!/usr/bin/env python3
"""Two independently controlled measured paths, formation error and peer gap."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):
    rows=list(csv.DictReader((path/'gazebo_truth.csv').open()));t=np.array([float(r['time']) for r in rows])
    positions=np.array([[[float(r[name+'_x']),float(r[name+'_y'])] for r in rows] for name in ('tb3_0','tb3_1')])
    spec=json.loads((path/'dynamic_scenario.json').read_text())['specifications'][0]
    actor=np.array([[float(r[spec['name']+'_x']),float(r[spec['name']+'_y'])] for r in rows])
    return t,positions,actor


def draw(root,output,paired=False):
    out=Path(output)
    if any(out.with_suffix(s).exists() for s in ('.png','.pdf')):raise FileExistsError(out)
    cases=[('legacy_overtaking_211','Old two-robot profile'),('new_overtaking_211','Current bundle')] if paired else [
        ('new_overtaking_211','Overtaking'),('new_head_on_211','Head-on'),('new_oblique_211','Oblique crossing')]
    fig,axes=plt.subplots(len(cases),3,figsize=(15,3.2*len(cases)),squeeze=False)
    for row,(name,title) in enumerate(cases):
        path=root/name;t,p,actor=load(path);meta=json.loads((path/'run_metadata.json').read_text());proto=meta['active_protocol']
        reference=np.asarray(proto['reference_start'])+t[:,None]*np.asarray(proto['reference_velocity'])
        for i,color in [(0,'C0'),(1,'C1')]:
            axes[row,0].plot(p[i,:,0],p[i,:,1],color=color,lw=1.2,label='Measured tb3_'+str(i))
            error=np.linalg.norm(p[i]-reference-np.asarray(proto['robot_offsets'][i]),axis=1)
            if paired:
                movement=np.linalg.norm(np.diff(p[i],axis=0),axis=1);cumulative=np.r_[0.,np.cumsum(movement)]
                grid=np.arange(5.,39.,.1);speed=np.interp(grid,t,cumulative)-np.interp(grid-1,t,cumulative)
                axes[row,1].plot(grid,speed,color=color,lw=1.1,label='tb3_'+str(i))
            else:axes[row,1].plot(t,error,color=color,lw=1.1,label='tb3_'+str(i))
        for stamp in [0.,8.,12.,16.,24.,39.9]:
            sample=np.array([[np.interp(stamp,t,p[i,:,j]) for j in (0,1)] for i in (0,1)])
            axes[row,0].plot(sample[:,0],sample[:,1],color='0.7',lw=.65,marker='.',ms=3)
        shown=abs(actor[:,1])<=1.2
        axes[row,0].plot(actor[shown,0],actor[shown,1],':',color='C2',lw=1,label='Measured actor')
        axes[row,0].axhline(0,color='0.4',ls=':',lw=.7)
        axes[row,0].set(title=title,xlabel='x (m)',ylabel='y (m)');axes[row,0].set_aspect('equal',adjustable='datalim')
        delta=p[0]-p[1];formation=np.linalg.norm(delta-np.array([1.,0.]),axis=1);gap=np.linalg.norm(delta,axis=1)-.26
        axes[row,2].plot(t,formation,color='C3',lw=1.1,label='Relative formation error')
        axes[row,2].plot(t,gap,color='C2',lw=1.1,label='Peer circular-proxy gap')
        axes[row,2].axhline(0,color='k',lw=.8,ls=':');axes[row,2].set(xlim=(0,40),xlabel='Task time (s)',ylabel='Distance (m)')
        if paired:
            axes[row,1].axhline(.075,color='C3',lw=.9,ls=':',label='Frozen speed minimum')
            axes[row,1].set(xlim=(4,39),ylim=(0,.26),xlabel='Task time (s)',ylabel='1 s travelled speed (m/s)')
        else:axes[row,1].set(xlim=(0,40),xlabel='Task time (s)',ylabel='Timed-target position error (m)')
        report=json.loads((path/'active_evaluation.json').read_text())
        axes[row,2].set_title('Original team acceptance: '+('PASS' if report['passed'] else 'FAIL'),fontsize=10)
        for ax in axes[row]:ax.grid(alpha=.2)
    for ax in axes[0]:ax.legend(fontsize=7)
    fig.suptitle('Actual Gazebo truth: two own-scan controllers, 1 m initial tandem spacing',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96))
    for suffix in ('.png','.pdf'):fig.savefig(out.with_suffix(suffix),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',required=True);p.add_argument('--paired',action='store_true');a=p.parse_args();draw(a.root,a.output,a.paired)
