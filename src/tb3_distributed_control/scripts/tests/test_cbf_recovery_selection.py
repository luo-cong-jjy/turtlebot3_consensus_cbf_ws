import unittest
import numpy as np
from test_cbf_packed_numeric import PACKAGE
from cbf_mpc_core import default_config
from mpc_dcbf_cached_solver import adapt_negative_recovery,first_step_mask


def closing(distance=.0218,center=.25):
    return dict(name='closing',center=np.array([center,0.]),a=.15,b=.15,theta=0.,gamma=.03,
        prediction_centers=np.array([[center,0.],[center-distance,0.]]))


class RecoverySelectionChecks(unittest.TestCase):
    def test_reduces_only_unsafe_recovery_rate_and_constructs_a_monotone_witness(self):
        cfg=default_config();state=np.zeros(3);obs=closing()
        peer=dict(name='safe',center=np.array([-1.,0.]),a=.15,b=.15,theta=0.,gamma=.3)
        v,w=np.meshgrid(np.linspace(-.22,.22,33),np.linspace(-2.84,2.84,33));grid=np.column_stack((v.ravel(),w.ravel()))
        self.assertFalse(first_step_mask(state,grid,[obs,peer],(-.22,.22),np.zeros(2),cfg)[0].any())
        result,info=adapt_negative_recovery(state,[obs,peer],(-.22,.22),np.zeros(2),cfg)
        self.assertTrue(info['adapted']);self.assertGreaterEqual(result[0]['gamma'],0.)
        self.assertLess(result[0]['gamma'],.03);self.assertEqual(result[1]['gamma'],.3)
        self.assertEqual(obs['gamma'],.03)
        witness=np.array(info['witness'])[None,:]
        self.assertTrue(first_step_mask(state,witness,result,(-.22,.22),np.zeros(2),cfg)[0][0])
        self.assertGreater(info['witness_next_h'][0],info['initial_h'][0])

    def test_currently_safe_barrier_is_never_relaxed_even_if_its_future_is_infeasible(self):
        obs=closing(distance=.1,center=.28);obs['gamma']=.3
        result,info=adapt_negative_recovery(np.zeros(3),[obs],(-.22,.22),None,default_config())
        self.assertFalse(info['adapted']);self.assertEqual(result[0]['gamma'],.3)

    def test_no_monotone_recovery_keeps_the_original_problem(self):
        result,info=adapt_negative_recovery(np.zeros(3),[closing(distance=.05)],(-.22,.22),None,default_config())
        self.assertFalse(info['adapted']);self.assertEqual(result[0]['gamma'],.03)
        self.assertEqual(info['reason'],'no_monotone_grid_recovery')

    def test_input_increment_limits_are_not_bypassed(self):
        cfg=default_config();cfg['delta_limit']=True
        result,info=adapt_negative_recovery(np.zeros(3),[closing()],(-.22,.22),np.zeros(2),cfg)
        self.assertFalse(info['adapted']);self.assertEqual(result[0]['gamma'],.03)


if __name__=='__main__':unittest.main()
