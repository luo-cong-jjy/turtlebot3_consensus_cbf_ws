#!/usr/bin/env python3
"""Scenario drawing; the green constructive witness is NOT controller output."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('qualification');p.add_argument('--output',required=True);a=p.parse_args();prefix=Path(a.output)
    if any(prefix.with_suffix(s).exists() for s in ('.png','.pdf')):raise FileExistsError(prefix)
    q=json.loads(Path(a.qualification).read_text());witness=np.array(q['witness_positions']);times=np.array(q['witness_times'])
    fig,axes=plt.subplots(3,1,figsize=(11,9))
    for ax,(name,r) in zip(axes,q['scenes'].items()):
        spec=r['specification'];start=np.array(spec['start']);velocity=np.array(spec['velocity']);actor=start+times[:,None]*velocity
        ax.plot([.4,6.4],[0,0],'k:',label='Original nominal reference')
        ax.plot(witness[:,0],witness[:,1],color='C2',lw=1.2,label='Constructive witness ONLY (not controller)')
        shown=(times<=16) if name=='active_oblique_crossing' else np.ones(len(times),dtype=bool)
        ax.plot(actor[shown,0],actor[shown,1],'--',color='C3',lw=1.,label='Prescribed actor path')
        ax.add_patch(Circle(start,spec['radius'],color='C3',alpha=.3));ax.add_patch(Circle([.4,0],.13,color='C0',alpha=.4))
        ax.annotate('actor start',start,xytext=(start[0]+.1,start[1]+.3),arrowprops=dict(arrowstyle='->'),fontsize=8)
        at=r['first_nominal_proxy_collision'];point=start+at*velocity;ax.add_patch(Circle(point,spec['radius'],color='C3',alpha=.12))
        ax.set_title(name+' / actor velocity='+str(spec['velocity'])+' m/s',fontsize=10)
        ax.set_xlim(.1,6.6);ax.set_ylim(-1.0,1.1);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.25);ax.set_ylabel('y (m)')
    axes[0].legend(fontsize=7,loc='upper right');axes[-1].set_xlabel('x (m)')
    fig.suptitle('Scenario qualification: feasible active paths, NOT algorithm results',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96))
    for suffix in ('.png','.pdf'):fig.savefig(prefix.with_suffix(suffix),dpi=180)
    plt.close(fig)
