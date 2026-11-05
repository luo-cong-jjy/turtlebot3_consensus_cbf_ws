import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from test_cbf_packed_numeric import PACKAGE
from cbf_mpc_core import default_config
from mpc_dcbf_cached_solver import relative_obstruction,progress_preview,continuous_guidance,CachedMPCProblem,PreviewBudgetExceeded,bypass_velocity


class RelativeLookaheadChecks(unittest.TestCase):
    def test_relative_velocity_is_translated_back_to_world_before_world_speed_limits(self):
        tangent=np.array([-1.,0.]);normal=np.array([0.,1.]);frame=np.array([0.,.03])
        relative=bypass_velocity(tangent,normal,.06)
        world=bypass_velocity(tangent,normal,.06,frame)
        np.testing.assert_allclose(world-relative,frame,atol=1e-15)
        p=np.array([1.,1.]);c=np.array([.5,.5]);dt=.1
        np.testing.assert_allclose((p+dt*world)-(c+dt*frame),p-c+dt*relative,atol=1e-15)

    def test_zero_motion_reproduces_the_original_world_reference_exactly(self):
        d=json.loads((Path(__file__).parent/'fixtures/relative_lookahead_input.json').read_text())
        for o in d['obstacles']:
            if o.get('guidance_motion') is None:o.pop('guidance_motion',None)
            else:
                for m in o['guidance_motion']['models']:m['velocity']=[0.,0.];m['acceleration']=[0.,0.]
        cfg=dict(default_config(),**d['config']);cfg.update(predictive_obstruction=True)
        args=(np.array(d['state']),np.array(d['goals']),d['obstacles'],d['selected_obstacle'],'cw')
        a=progress_preview(*args,dict(cfg,relative_avoidance_flow=False),np.array(d['previous']),d['bounds'])
        b=progress_preview(*args,dict(cfg,relative_avoidance_flow=True),np.array(d['previous']),d['bounds'])
        np.testing.assert_array_equal(a['controls'],b['controls'])
        self.assertEqual(a['feasible'],b['feasible'])

    def test_partial_opposite_candidate_does_not_assume_the_old_side_was_evaluated(self):
        d=json.loads((Path(__file__).parent/'fixtures/relative_lookahead_input.json').read_text())
        for o in d['obstacles']:
            if o.get('guidance_motion') is None:o.pop('guidance_motion',None)
        cfg=dict(default_config(),**d['config']);cfg.update(progress_guidance=True,execution_backup=True,predictive_obstruction=True,active_side_lock=True)
        memory=dict(progress_previous=np.array(d['previous']),progress_bounds=d['bounds'],progress_state=dict(d['previous_decision'],side_committed=False),progress_counter=1)
        calls=[]
        def limited(*args,**kwargs):
            calls.append(args[4])
            if len(calls)>1:raise PreviewBudgetExceeded()
            return progress_preview(*args,**kwargs)
        with patch('mpc_dcbf_cached_solver.progress_preview',side_effect=limited):
            result=continuous_guidance(np.array(d['state']),np.array(d['goals']),d['obstacles'],cfg,memory)
        self.assertEqual(calls,['cw','ccw']);self.assertEqual(result[1],1.)
        self.assertEqual(memory['progress_summary']['mode'],'cw')
        self.assertFalse(memory['progress_decision']['search_complete'])
        self.assertFalse(memory['progress_decision']['side_committed'])

    def test_rear_catchup_and_lateral_crossing_trigger_relative_approach(self):
        t=np.linspace(0,2,21);axes=np.tile([.15,.15],(21,1));angles=np.zeros(21)
        rear=np.column_stack((-.6+1.4*t,np.zeros(21)))
        crossing=np.column_stack((np.ones(21),-1+t))
        # Both objects lie outside the static forward ray at the current time.
        self.assertTrue(np.all(relative_obstruction(np.zeros(2),np.array([2.,0.]),np.array([rear,crossing]),axes,angles,t,.12,1.)))

    def test_a_receding_rear_object_does_not_lock_a_bypass_side(self):
        t=np.linspace(0,2,21);centers=np.column_stack((-.6+.2*t,np.zeros(21)))[None,:,:]
        result=relative_obstruction(np.zeros(2),np.array([2.,0.]),centers,np.tile([.15,.15],(21,1)),np.zeros(21),t,.12,1.)
        self.assertFalse(result[0])

    def test_recorded_identical_primitives_become_distinct_and_keep_original_constraints(self):
        d=json.loads((Path(__file__).parent/'fixtures/relative_lookahead_input.json').read_text())
        for o in d['obstacles']:
            if o.get('guidance_motion') is None:o.pop('guidance_motion',None)
        cfg=dict(default_config(),**d['config']);cfg.update(progress_guidance=True,execution_backup=True,predictive_obstruction=True,active_side_lock=True)
        args=(np.array(d['state']),np.array(d['goals']),d['obstacles'],d['selected_obstacle'])
        old=[progress_preview(*args,m,dict(cfg,predictive_obstruction=False),np.array(d['previous']),d['bounds']) for m in ('ccw','cw')]
        np.testing.assert_array_equal(old[0]['controls'],old[1]['controls'])
        self.assertFalse(old[0]['circulation_active'])
        new=[progress_preview(*args,m,cfg,np.array(d['previous']),d['bounds']) for m in ('ccw','cw')]
        self.assertTrue(all(v['feasible'] and v['circulation_active'] for v in new))
        self.assertLess(new[1]['cost'],new[0]['cost'])
        memory=dict(progress_previous=np.array(d['previous']),progress_bounds=d['bounds'],progress_state=dict(d['previous_decision'],side_committed=False))
        continuous_guidance(np.array(d['state']),np.array(d['goals']),d['obstacles'],cfg,memory)
        self.assertEqual(memory['progress_summary']['mode'],'cw')
        self.assertTrue(memory['progress_decision']['side_committed'])
        problem=CachedMPCProblem(cfg,len(d['obstacles']),False)
        self.assertTrue(problem.validate(memory['progress_plan'],np.array(d['state']),d['obstacles'],d['bounds'],np.array(d['previous'])))


if __name__=='__main__':unittest.main()
