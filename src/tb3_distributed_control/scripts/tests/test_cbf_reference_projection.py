"""Generate references against the same retained-mean tests used to accept them."""
import json
from pathlib import Path
import unittest
import numpy as np
from test_cbf_packed_numeric import PACKAGE
from cbf_mpc_core import default_config
from mpc_dcbf_cached_solver import progress_preview,continuous_guidance,CachedMPCProblem,rollout
from diagnose_cbf_first_step import certify


class ReferenceProjectionChecks(unittest.TestCase):
    def test_model_infeasibility_bound_covers_all_sampled_bounded_commands(self):
        cfg=default_config();obs=dict(name='closing',center=np.array([.25,0.]),a=.15,b=.15,theta=0.,prediction_centers=np.array([[.25,0.],[.20,0.]]))
        result=certify(np.zeros(3),[obs],(-.22,.22),cfg)
        self.assertTrue(result['certified_model_infeasible'])
        upper=result['certificates'][0]['reachable_upper']
        from mpc_dcbf_cached_solver import barrier_values
        for v in np.linspace(-.22,.22,11):
            for w in np.linspace(-2.84,2.84,11):
                h=barrier_values(rollout(np.zeros(3),[[v,w]],.1),[obs],.12)[0]
                self.assertLessEqual(h[1],upper+1e-12)
                self.assertLess(h[1],(1-cfg['gamma'])*h[0]-2e-6)

    def test_missing_certificate_does_not_claim_feasibility(self):
        cfg=default_config();obs=dict(center=np.array([0.,.29]),a=.15,b=.15,theta=0.,prediction_centers=np.array([[0.,.29],[0.,.275]]))
        result=certify(np.zeros(3),[obs],(-.22,.22),cfg)
        self.assertFalse(result['certified_model_infeasible'])
        self.assertEqual(result['no_certificate_means'],'undecided')
        self.assertNotIn('feasible',result)

    def setUp(self):
        self.data=json.loads((Path(__file__).parent/'fixtures/hypothesis_projection_input.json').read_text())
        self.cfg=dict(default_config(),**self.data['config'])
        self.cfg.update(align_reference_hypotheses=True,execution_backup=True,progress_guidance=True)

    def preview(self,cfg=None):
        d=self.data
        return progress_preview(np.array(d['state']),np.array(d['goals']),d['obstacles'],
            d['selected_obstacle'],'ccw',cfg or self.cfg,np.array(d['previous']),d['bounds'])

    def test_recorded_rejection_can_be_repaired_without_weakening_its_acceptance_checks(self):
        old=self.preview(dict(self.cfg,align_reference_hypotheses=False));new=self.preview()
        self.assertFalse(old['feasible']);self.assertLess(min(old['hypothesis_min_h'].values()),-.002)
        self.assertTrue(new['feasible']);self.assertGreaterEqual(min(new['hypothesis_min_h'].values()),-2e-6)
        np.testing.assert_allclose(new['states'],rollout(self.data['state'],new['controls'],.1),atol=1e-12)
        candidate=dict(new,plan_controls=new['vectors'])
        problem=CachedMPCProblem(self.cfg,len(self.data['obstacles']),False)
        self.assertTrue(problem.validate(candidate,np.array(self.data['state']),self.data['obstacles'],self.data['bounds'],np.array(self.data['previous'])))
        memory=dict(progress_previous=np.array(self.data['previous']),progress_bounds=self.data['bounds'],progress_state=self.data['previous_decision'])
        continuous_guidance(np.array(self.data['state']),np.array(self.data['goals']),self.data['obstacles'],self.cfg,memory)
        self.assertEqual(memory['progress_summary']['mode'],'ccw')

    def test_zero_weight_hypothesis_is_not_silently_dropped_or_initial_intrusion_hidden(self):
        d=self.data;motion=d['obstacles'][0]['guidance_motion']
        motion['weights']=[0.,0.,1.]
        motion['models'][0]['position']=d['state'][:2]
        result=self.preview()
        self.assertFalse(result['feasible'])
        self.assertLess(result['hypothesis_min_h']['tracked_1/0/cv'],0.)

    def test_switching_off_hypothesis_checks_preserves_original_projection(self):
        a=self.preview(dict(self.cfg,progress_hypotheses=False))
        b=self.preview(dict(self.cfg,progress_hypotheses=False,align_reference_hypotheses=False))
        np.testing.assert_array_equal(a['controls'],b['controls'])
        self.assertEqual(a['feasible'],b['feasible'])

    def test_hypothesis_projection_preserves_enabled_input_increment_bounds(self):
        result=self.preview(dict(self.cfg,delta_limit=True))
        u=result['controls'];increments=np.vstack((u[0]-self.data['previous'],np.diff(u,axis=0)))
        self.assertTrue(np.all(np.abs(u)<=np.array([.22,2.84])+2e-6))
        self.assertTrue(np.all(np.abs(increments)<=np.array([.06,.5])+2e-6))


if __name__=='__main__':unittest.main()
