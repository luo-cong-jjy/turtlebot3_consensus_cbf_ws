#!/usr/bin/env python3
"""Same recorded single-robot inputs, frozen V5 vs current reference functions.

No wall deadline is imposed in this algebraic regression check. It does not
prove closed-loop reproducibility or erase a failed actual repeat.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
import mpc_dcbf_cached_solver as current


def compare(directory,output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    source=subprocess.check_output(['git','show','9104bf8:src/tb3_distributed_control/纯仿真代码/mpc_dcbf_cached_solver.py'],text=True)
    old_tree=ast.parse(source);new_tree=ast.parse(Path(current.__file__).read_text())
    preserved={}
    for name in ('CachedMPCProblem','first_step_mask','barrier_values'):
        nodes=[next(n for n in tree.body if getattr(n,'name',None)==name) for tree in (old_tree,new_tree)]
        preserved[name]=ast.dump(nodes[0])==ast.dump(nodes[1])
    maximum=0.;count=0;memories=[{},{}]
    with tempfile.TemporaryDirectory(prefix='cbf_v5_regression_') as temp:
        file=Path(temp)/'frozen.py';file.write_text(source)
        spec=importlib.util.spec_from_file_location('cbf_frozen_v5',file);frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
        for line in (Path(directory)/'tb3_0_solver_trace.jsonl').open():
            row=json.loads(line);snapshot=row.get('manager',{}).get('input_snapshot')
            if not snapshot:continue
            assert not any(o.get('peer') for o in snapshot['obstacles'])
            cfg=dict(snapshot['config']);cfg.pop('_preview_deadline',None)
            values=[]
            for module,memory in zip((frozen,current),memories):
                memory.update(progress_previous=np.array(snapshot['previous']),progress_bounds=snapshot['bounds'])
                result=module.continuous_guidance(np.array(snapshot['state']),np.array(snapshot['goals']),snapshot['obstacles'],cfg,memory)
                values.append(np.r_[np.asarray(result[0]).ravel(),result[1],np.asarray(result[2]).ravel() if len(result)>2 else []])
            if values[0].shape!=values[1].shape:raise AssertionError('Different reference output shapes')
            maximum=max(maximum,float(np.max(np.abs(values[0]-values[1]))));count+=1
    result=dict(passed=count>0 and maximum<=1e-12 and all(preserved.values()),cycles=count,
        maximum_reference_difference=maximum,preserved_model_components=preserved,
        interpretation=__doc__,frozen_commit='9104bf8',source_directory=str(Path(directory).resolve()))
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('--output',required=True)
    a=p.parse_args();compare(a.directory,a.output)
