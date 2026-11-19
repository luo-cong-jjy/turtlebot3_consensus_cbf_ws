#!/usr/bin/env python3
"""Measured geometry and identical-input motion replay, with truth labelled."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle


def plot(root,output):
    out=Path(output)
    if any(out.with_suffix(s).exists() for s in ('.png','.pdf')):raise FileExistsError(out)
    p=root/'capture_oblique';r=json.loads((p/'observer_replay_v2.json').read_text())
    truth=list(csv.DictReader((p/'gazebo_truth.csv').open()));begin=np.median([float(s['ros_time'])-float(s['time']) for s in truth])
    frames=[f for f in map(json.loads,(p/'obstacle_predictions.jsonl').open()) if len(f['measurements'])==1]
    frame=min(frames,key=lambda f:abs(f['stamp']-begin-8.));e=frame['measurements'][0];points=np.array(e['points'])
    # Production geometric fit, used only on recorded points here.
    from test_cbf_inline_geometry import inline
    original=inline.Ellipse(*e['center'],e['a'],e['b'],e['theta'],e['point_count'])
    circle=inline.circle_observation(points,original,frame['sensor_origin']);assert circle['accepted']
    t=np.array([float(s['time']) for s in truth]);actual=np.array([[float(s['cbf_dynamic_0_x']),float(s['cbf_dynamic_0_y'])] for s in truth])
    stamp=frame['stamp']-begin;center=np.array([np.interp(stamp,t,actual[:,j]) for j in (0,1)])
    radius=json.loads((p/'dynamic_scenario.json').read_text())['specifications'][0]['radius']
    fig,axes=plt.subplots(2,2,figsize=(12,8));ax=axes[0,0]
    ax.scatter(points[:,0],points[:,1],s=14,c='k',label='Recorded scan cluster')
    ax.add_patch(Ellipse(e['center'],2*e['a'],2*e['b'],angle=np.degrees(e['theta']),fill=False,color='C0',label='Original padded MVEE'))
    ax.add_patch(Circle(circle['center'],circle['enclosure_radius'],fill=False,color='C1',label='New enclosing circle'))
    ax.add_patch(Circle(center,radius,fill=False,color='0.5',ls='--',label='Physical cylinder (offline truth)'))
    ax.plot(*e['center'],'o',c='C0');ax.plot(*circle['center'],'x',c='C1',ms=9);ax.plot(*center,'+',c='0.3',ms=10)
    ax.set(aspect='equal',xlabel='World x (m)',ylabel='World y (m)',title='Visible-contour centre and reconstructed centre')
    ax.legend(fontsize=7)
    records=r['records'];times=np.array([s['time'] for s in records]);true_v=np.array([s['truth_velocity'] for s in records])
    axes[0,1].plot(times,np.linalg.norm(true_v,axis=1),'k:',lw=1.2,label='Offline truth')
    for kind,color,label in [('mvee','C0','MVEE observer'),('circle','C1','Circle observer')]:
        velocity=np.array([s[kind]['velocity'] for s in records]);axes[0,1].plot(times,np.linalg.norm(velocity,axis=1),color=color,lw=1.1,label=label)
        for h,style in [('1.0','-'),('2.0','--')]:
            axes[1,0].plot(times,[s[kind]['displacement_error'][h] for s in records],style,color=color,lw=1.,label=label+', '+h+' s')
    axes[0,1].set(xlabel='Task time (s)',ylabel='Estimated speed (m/s)',title='Same recorded observations; velocity changes with view')
    axes[0,1].legend(fontsize=8)
    axes[1,0].set(xlabel='Task time (s)',ylabel='Rigid displacement prediction error (m)',title='Fixed centre offsets cancel in this metric')
    axes[1,0].legend(fontsize=7)
    names=['capture_head_on','capture_oblique','capture_overtaking'];labels=['Head-on','Oblique','Overtaking'];x=np.arange(3)
    for offset,kind,color,label in [(-.18,'mvee','C0','MVEE'),(.18,'circle','C1','Circle')]:
        values=[json.loads((root/name/'observer_replay_v2.json').read_text())['summary'][kind]['velocity_rmse'] for name in names]
        axes[1,1].bar(x+offset,values,.36,color=color,label=label)
    axes[1,1].set(xticks=x,xticklabels=labels,ylabel='Velocity RMSE (m/s)',title='Identical-input replay, fixed [4,20] s window');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Offline geometry/motion diagnosis: truth is evaluation only; no oracle radius in fitting',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.96))
    for suffix in ('.png','.pdf'):fig.savefig(out.with_suffix(suffix),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',required=True);a=p.parse_args();plot(a.root,a.output)
