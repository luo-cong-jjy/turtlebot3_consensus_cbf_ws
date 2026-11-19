#!/usr/bin/env python3
"""Recorded-input fallback defect and offline stress; not live fault injection."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw(root,output):
    out=Path(output)
    if any(out.with_suffix(s).exists() for s in ('.png','.pdf')):raise FileExistsError(out)
    data=json.loads((root/'prior_source_copies/v2_circle_oblique_451/prior_retention_replay.json').read_text())
    rows=data['records'];t=np.array([r['time'] for r in rows]);fig,axes=plt.subplots(2,2,figsize=(12,8))
    for kind,color,label in [('circle','C0','Legacy fallback'),('circle_prior','C1','Retained prior')]:
        axes[0,0].plot(t,[r[kind]['centre_error'] for r in rows],color=color,lw=1.2,label=label)
        axes[0,1].plot(t,[r[kind]['velocity_error'] for r in rows],color=color,lw=1.2,label=label)
        axes[1,0].step(t,[int(r[kind]['observation_kind']=='circle') for r in rows],where='post',color=color,lw=1.3,label=label)
    axes[0,0].set(xlim=(10.5,13.5),xlabel='Task time (s)',ylabel='Centre error (m)',title='Previously recorded reference-point jump')
    axes[0,1].set(xlim=(10.5,16),xlabel='Task time (s)',ylabel='Velocity error (m/s)',title='Same recorded observations')
    axes[1,0].set(xlim=(11.5,12.5),ylim=(-.1,1.1),yticks=[0,1],yticklabels=['MVEE','Circle'],xlabel='Task time (s)',title='Representation during the rejected arc')
    summaries=json.loads((root/'offline_stress_summary.json').read_text());x=np.arange(len(summaries))
    for offset,kind,color,label in [(-.18,'circle','C0','Legacy'),(.18,'circle_prior','C1','Retained')]:
        values=[100*r['summary'][kind]['occupied_circle_coverage']['0.0'] for r in summaries]
        axes[1,1].bar(x+offset,values,.36,color=color,label=label)
        for xx,value in zip(x+offset,values):axes[1,1].text(xx,value+1,'%.1f'%value,ha='center',fontsize=8)
    axes[1,1].set(xticks=x,xticklabels=['Oblique451','Oblique211','Head-on','Overtaking'],ylim=(0,108),ylabel='Whole evaluation-circle inclusion (%)',title='Offline 1.5 s quality rejection; fixed [4,20] s')
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[1,1].legend(fontsize=8,loc='lower right')
    fig.suptitle('Shape-prior retention: recorded replay and OFFLINE quality stress (not a physical guarantee)',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.96))
    for suffix in ('.png','.pdf'):fig.savefig(out.with_suffix(suffix),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',required=True);a=p.parse_args();draw(a.root,a.output)
