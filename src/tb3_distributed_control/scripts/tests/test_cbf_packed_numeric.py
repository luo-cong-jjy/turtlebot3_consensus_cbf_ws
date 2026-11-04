"""Independent scalar formula, frozen previews and cooperative budget checks."""
from pathlib import Path
import json
import math
import sys
import unittest
from unittest.mock import patch
import numpy as np

PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'scripts'),str(PACKAGE/'纯仿真代码')]
from cbf_mpc_core import default_config
from mpc_dcbf_cached_solver import (barrier_values,first_step_mask,obstacle_batch,
    progress_preview,PreviewBudgetExceeded,continuous_guidance,CachedMPCProblem)


def scalar_h(position,obstacle,j,margin):
    center=obstacle.get('prediction_centers',[obstacle['center']]*11)[j]
    a,b=obstacle.get('prediction_axes',[[obstacle['a'],obstacle['b']]]*11)[j]
    angle=obstacle.get('prediction_angles',[obstacle['theta']]*11)[j]
    dx,dy=position-center;c,s=math.cos(angle),math.sin(angle)
    return b*(math.sqrt(((c*dx+s*dy)/a)**2+((-s*dx+c*dy)/b)**2+1e-12)-1)-margin


class PackedNumericChecks(unittest.TestCase):
    def test_completed_feasible_reference_survives_timeout_of_the_other_side(self):
        path=Path(__file__).parent/'fixtures';d=json.loads((path/'local_escape_input.json').read_text())
        cfg=dict(default_config(),**d['config']);cfg.update(progress_guidance=True,execution_backup=True,bounded_local_reference=True)
        memory=dict(progress_previous=np.array(d['previous']),progress_bounds=d['bounds']);calls=[]
        def limited(*args,**kwargs):
            calls.append(args[4])
            if len(calls)>1:raise PreviewBudgetExceeded()
            return progress_preview(*args,**kwargs)
        with patch('mpc_dcbf_cached_solver.progress_preview',side_effect=limited):
            guide=continuous_guidance(np.array(d['state']),np.array(d['goals']),d['obstacles'],cfg,memory)
        self.assertEqual(guide[1],1.)
        self.assertFalse(memory['progress_decision']['search_complete'])
        self.assertEqual(set(memory['progress_decision']['previews']),{'ccw'})
        self.assertEqual(memory['progress_summary']['preview_scope'],'mpc_window')
        problem=CachedMPCProblem(cfg,len(d['obstacles']),False)
        self.assertTrue(problem.validate(memory['progress_plan'],np.array(d['state']),d['obstacles'],d['bounds'],np.array(d['previous'])))

    def test_batched_barriers_and_reused_stage_masks_match_scalar_formula(self):
        rng=np.random.default_rng(197);cfg=default_config();states=rng.normal(size=(11,3))
        obstacles=[]
        for i in range(7):
            o=dict(center=rng.normal(size=2),a=.3,b=.15,theta=.2,gamma=.03 if i%2 else .3)
            if i%2:o.update(prediction_centers=rng.normal(size=(11,2)),prediction_axes=rng.uniform(.1,.5,(11,2)),prediction_angles=rng.normal(size=11))
            obstacles.append(o)
        expected=np.array([[scalar_h(x[:2],o,j,.12) for j,x in enumerate(states)] for o in obstacles])
        np.testing.assert_allclose(barrier_values(states,obstacles,.12),expected,atol=3e-15,rtol=0)
        commands=rng.uniform([-.3,-3.3],[.3,3.3],(311,2));previous=np.array([.03,.1])
        packed=obstacle_batch(obstacles,11,cfg['gamma'])
        for delta in [False,True]:
            cfg['delta_limit']=delta
            for stage in [0,4,9]:
                state=states[stage];h0=np.array([scalar_h(state[:2],o,stage,.12) for o in obstacles]);safe=h0.min()>=-2e-6
                mask=[]
                for v,w in commands:
                    angle=state[2]+.1*w;p=state[:2]+.1*v*np.array([math.cos(angle),math.sin(angle)])
                    h1=np.array([scalar_h(p,o,stage+1,.12) for o in obstacles])
                    valid=(-.22-2e-6<=v<=.22+2e-6 and abs(w)<=2.84+2e-6
                        and np.all(h1-(1-packed.gamma)*h0>=-2e-6) and (not safe or h1.min()>=-2e-6))
                    if delta:valid=valid and np.all(np.abs([v-previous[0],w-previous[1]])<=np.array([.06,.5])+2e-6)
                    mask.append(valid)
                actual,_,_=first_step_mask(state,commands,obstacles,(-.22,.22),previous,cfg,batch=packed,stage=stage)
                np.testing.assert_array_equal(actual,mask)

    def test_new_sensor_snapshot_does_not_reuse_old_numeric_arrays(self):
        o=dict(center=np.array([.7,0.]),a=.2,b=.15,theta=0.)
        before=barrier_values(np.zeros((2,3)),[o],.12)
        packed=obstacle_batch([o],2)
        o['center'][0]=.9
        after=barrier_values(np.zeros((2,3)),[o],.12)
        self.assertGreater(after[0,0],before[0,0])
        self.assertEqual(packed.centers[0,0,0],.7)
        self.assertEqual(barrier_values(np.zeros((3,3)),[],.12).shape,(0,3))

    def test_frozen_previews_preserve_controls_scores_and_feasibility(self):
        path=Path(__file__).parent/'fixtures';data=json.loads((path/'local_escape_input.json').read_text())
        old=json.loads((path/'packed_preview_expected.json').read_text())['results']
        for local in [False,True]:
            cfg=dict(data['config'],progress_local=local,progress_horizon=1. if local else 4.)
            for mode in ['direct','yield','ccw','cw']:
                result=progress_preview(np.array(data['state']),np.array(data['goals']),data['obstacles'],
                    data['selected_obstacle'],mode,cfg,np.array(data['previous']),data['bounds'])
                frozen=old[str(local)+'_'+mode]
                for key in ['states','controls','vectors','cost','min_h','min_slack']:
                    np.testing.assert_allclose(result[key],frozen[key],atol=1e-12,rtol=0)
                self.assertEqual(result['feasible'],frozen['feasible'])

    def test_preview_stops_during_generation_without_returning_a_partial_plan(self):
        path=Path(__file__).parent/'fixtures';d=json.loads((path/'local_escape_input.json').read_text())
        cfg=dict(d['config'],_preview_deadline=1.)
        with patch('mpc_dcbf_cached_solver.time.perf_counter',side_effect=[0.,.2,.8,1.1]):
            with self.assertRaises(PreviewBudgetExceeded):
                progress_preview(np.array(d['state']),np.array(d['goals']),d['obstacles'],
                    d['selected_obstacle'],'ccw',cfg,np.array(d['previous']),d['bounds'])


if __name__=='__main__':unittest.main()
