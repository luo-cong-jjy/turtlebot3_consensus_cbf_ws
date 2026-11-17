#!/usr/bin/env python3
"""Recorded-input backend equivalence and wall-time diagnostic, not closed loop."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
from mpc_dcbf_cached_solver import continuous_guidance,warm_spatial_rollouts


def benchmark(directories,output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    records=[];maximum=0.;disagreements=0
    for directory in directories:
        source=Path(directory)/'tb3_0_solver_trace.jsonl';memories=[{},{}];timings=[[],[]];count=0
        for line in source.open():
            row=json.loads(line);snap=row.get('manager',{}).get('input_snapshot')
            if not snap:continue
            cfg=dict(snap['config']);cfg.pop('_preview_deadline',None)
            if count==0:warm_spatial_rollouts(cfg)
            values=[]
            for backend,memory in enumerate(memories):
                cfg['spatial_compiled_rollout']=bool(backend)
                memory.update(progress_previous=np.array(snap['previous']),progress_bounds=snap['bounds'])
                begin=time.perf_counter()
                result=continuous_guidance(np.array(snap['state']),np.array(snap['goals']),snap['obstacles'],cfg,memory)
                timings[backend].append(1000*(time.perf_counter()-begin))
                values.append(np.r_[np.asarray(result[0]).ravel(),result[1],
                    np.asarray(result[2]).ravel() if len(result)>2 else []])
            difference=float(np.max(abs(values[0]-values[1]))) if values[0].shape==values[1].shape else 1e9
            maximum=max(maximum,difference);disagreements+=difference>1e-10;count+=1
        records.append(dict(directory=str(Path(directory).resolve()),trace_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            cycles=count,numpy_median_ms=float(np.median(timings[0])),numpy_p95_ms=float(np.percentile(timings[0],95)),
            native_median_ms=float(np.median(timings[1])),native_p95_ms=float(np.percentile(timings[1],95))))
    report=dict(passed=disagreements==0,maximum_reference_difference=maximum,disagreements=disagreements,runs=records,
        source_sha256=hashlib.sha256((PACKAGE/'纯仿真代码/mpc_dcbf_cached_solver.py').read_bytes()).hexdigest(),
        scope='Same measured input sequences; reference memory replayed without wall deadlines, warmup excluded. Not actual callback timing or a closed-loop success proof.')
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directories',nargs='+');p.add_argument('--output',required=True)
    a=p.parse_args();benchmark(a.directories,a.output)
