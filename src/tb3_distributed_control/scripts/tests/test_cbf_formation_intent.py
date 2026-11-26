"""Bounded requests affect soft references only; signed original hard model stays."""
from pathlib import Path
import sys,copy,unittest
import numpy as np
P=Path(__file__).resolve().parents[2];sys.path[:0]=[str(P/'scripts'),str(P/'纯仿真代码')]
from cbf_mpc_core import default_config
from mpc_dcbf_cached_solver import (validate_formation_intent,sample_formation_intent,
    formation_intent_guidance,CachedMPCProblem,rollout,delay_compensated_request)


def inputs(y=.4):
    c=default_config();c.update(robot_id=1,formation_intent_support=True,spatial_pass=True,execution_backup=True)
    state=np.array([0.,y,0.]);goals=np.zeros((11,3));goals[:,0]=.015*np.arange(11);goals[:,1]=y
    packet=[1.,7.,10.,0.,0.,0.,1.,8.,4.,0.,0.,.5]
    obs=[dict(name='peer_0',peer=True,guidance=False,center=np.array([5.,0.]),a=.2,b=.2,theta=0.,
        coordination_stamp=10.1,formation_intent=packet,intent_received_stamp=10.05)]
    memory=dict(progress_previous=np.array([.15,0.]),progress_bounds=[-.22,.22],intent_sequence=8)
    base=(np.zeros((10,2)),0.)
    return c,state,goals,obs,memory,base


class IntentChecks(unittest.TestCase):
    def test_payload_validation_does_not_refresh_old_or_invalid_requests(self):
        p=inputs()[3][0]['formation_intent']
        self.assertTrue(validate_formation_intent(p,10.1)[0])
        self.assertEqual(validate_formation_intent(p,10.1,7)[1],'old_sequence')
        for data,now in [(p[:11],10.1),(p+[0],10.1),(p,10.451),(p,9.94)]:
            self.assertFalse(validate_formation_intent(data,now)[0])
        for k,value in [(0,2),(1,1.5),(6,3),(8,0),(9,4),(10,.5),(11,np.nan)]:
            q=p.copy();q[k]=value;self.assertFalse(validate_formation_intent(q,10.1)[0])

    def test_quintic_request_has_correct_endpoints_and_initial_derivative(self):
        p=inputs()[3][0]['formation_intent'];p[10]=.1
        y,dy=sample_formation_intent(p,np.array([8.,8.00001,12.,13.]))
        self.assertAlmostEqual(y[0],0);self.assertAlmostEqual(y[-1],.5)
        self.assertAlmostEqual(dy[0],.1);self.assertEqual(dy[-1],0)
        self.assertAlmostEqual((y[1]-y[0])/.00001,.1,places=5)

    def test_same_side_support_changes_reference_without_changing_goal_or_peer_prediction(self):
        c,x,g,o,m,base=inputs();before=copy.deepcopy(o);original=g.copy()
        r=formation_intent_guidance(x,g,o,c,m,base)
        self.assertEqual(r[1],1.);self.assertEqual(m['intent_summary']['peer_sources'],[0])
        self.assertGreater(r[0][:,1].mean(),0.)
        self.assertTrue(CachedMPCProblem(c,1,False).validate(m['progress_plan'],x,o,[-.22,.22],m['progress_previous']))
        np.testing.assert_array_equal(g,original);np.testing.assert_array_equal(o[0]['center'],before[0]['center'])
        self.assertEqual(o[0]['formation_intent'],before[0]['formation_intent'])

    def test_opposite_side_does_not_follow_and_demand_is_not_echoed(self):
        c,x,g,o,m,base=inputs(-.4)
        self.assertIs(formation_intent_guidance(x,g,o,c,m,base),base)
        self.assertEqual(m['intent_proposal'][6],0.)
        c,x,g,o,m,base=inputs();formation_intent_guidance(x,g,o,c,m,base)
        self.assertEqual(m['intent_proposal'][6],0.)
        self.assertEqual(m['intent_proposal'][11],0.)

    def test_conflicting_signs_use_owner_order_and_current_hard_check_can_veto(self):
        c,x,g,o,m,base=inputs();m.update(spatial_activation_active=True,spatial_time=2.,
            spatial_selected=dict(returning=False,start=0.,duration=4.,y0=0.,dy0=0.,target=-.5))
        formation_intent_guidance(x,g,o,c,m,base)
        self.assertEqual(m['intent_summary']['priority'],0);self.assertEqual(m['intent_summary']['side'],1.)
        c,x,g,o,m,base=inputs();o[0]['center']=x[:2].copy()
        self.assertIs(formation_intent_guidance(x,g,o,c,m,base),base)
        self.assertEqual(m['intent_summary']['mode'],'rejected_hard_prefix')

    def test_signed_reference_and_expired_request_release_both_obey_original_dynamics(self):
        c,x,g,o,m,base=inputs();x[2]=-np.pi+.1;m['progress_previous']=np.array([-.15,0.])
        formation_intent_guidance(x,g,o,c,m,base);p=m['progress_plan']
        self.assertTrue((p['controls'][:,0]<0).all())
        np.testing.assert_allclose(p['states'],rollout(x,p['controls'],.1),atol=1e-12)
        self.assertTrue(CachedMPCProblem(c,1,False).validate(p,x,o,[-.22,.22],m['progress_previous']))
        c,x,g,o,m,base=inputs();x[1]+=.3;m['intent_was_active']=True;o[0]['coordination_stamp']=10.8
        formation_intent_guidance(x,g,o,c,m,base)
        self.assertEqual(m['intent_summary']['mode'],'release')
        self.assertEqual(m['intent_summary']['sources'],[])

    def test_compensation_advances_local_sampling_clock_but_not_received_packet(self):
        c,x,g,o,m,base=inputs();packet=o[0]['formation_intent'].copy()
        future=delay_compensated_request(dict(state=x,goals=g,obstacles=o,stamp=10.1),np.array([.15,0.]),c)
        self.assertAlmostEqual(future['obstacles'][0]['coordination_stamp'],10.2)
        self.assertEqual(future['obstacles'][0]['formation_intent'],packet)


if __name__=='__main__':unittest.main()
