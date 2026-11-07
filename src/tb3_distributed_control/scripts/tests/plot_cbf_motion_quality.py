#!/usr/bin/env python3
"""Plot recorded errors and angular commands; does not regenerate raw data."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='append',required=True,help='Label=directory')
    p.add_argument('--output',required=True,help='Output prefix (PNG and PDF)');args=p.parse_args()
    prefix=Path(args.output)
    if any(prefix.with_suffix(s).exists() for s in ('.png','.pdf')):raise FileExistsError(prefix)
    fig,axes=plt.subplots(2,2,figsize=(11,6),sharex=True)
    for spec in args.run:
        label,directory=spec.split('=',1)
        for i,robot in enumerate((0,2)):
            rows=list(csv.DictReader((Path(directory)/('tb3_'+str(robot)+'.csv')).open()))
            def col(name):return np.array([float(r[name]) for r in rows])
            t=col('evaluation_time')
            axes[i,0].plot(t,np.hypot(col('consensus_error_x'),col('consensus_error_y')),label=label,lw=1.1)
            axes[i,1].plot(t,col('omega'),label=label,lw=1.1)
            axes[i,0].set_ylabel('Robot %d error (m)'%robot)
            axes[i,1].set_ylabel('Robot %d omega (rad/s)'%robot)
    for ax in axes.ravel():ax.grid(alpha=.25);ax.set_xlim(0,40)
    for ax in axes[-1]:ax.set_xlabel('Measurement time (s)')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Five robots / mixed scene / seed 211 / headless development runs',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));prefix.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.pdf'):fig.savefig(prefix.with_suffix(suffix),dpi=180)
    plt.close(fig)
