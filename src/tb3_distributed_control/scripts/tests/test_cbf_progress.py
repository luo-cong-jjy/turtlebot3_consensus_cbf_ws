import unittest
import numpy as np
from test_cbf_dynamic import inline,ellipse
from cbf_mpc_core import default_config
from mpc_dcbf_cached_solver import (progress_centers,progress_obstacles,progress_targets,progress_preview,
    continuous_guidance,_legacy_continuous_guidance,rollout)


def crossing():
    obs=dict(name='crossing',center=np.array([2.1,-.4]),a=.3,b=.22,theta=0.,dynamic=True,guidance=False,
        guidance_motion=dict(stamp=10.,weights=[1.],models=[dict(mode='ca',position=[2.1,-.4],
            velocity=[0.,.1],acceleration=[0.,.025],tau=1.)]))
    obs['prediction_centers']=progress_centers(obs,np.arange(11)*.1,.1)
    obs['prediction_axes']=np.tile([.3,.22],(11,1))
    return obs


class ProgressChecks(unittest.TestCase):
    def setUp(self):
        self.cfg=default_config();self.cfg['progress_guidance']=True
        self.state=np.array([1.5,0.,0.]);self.goals=np.zeros((11,3))
        self.goals[:,0]=1.5+.015*np.arange(11);self.previous=np.array([.15,0.])

    def test_descriptor_reproduces_existing_predictor_means_without_mutation(self):
        for mode in ('cv','ca','singer','frozen','adaptive'):
            tracker=inline.make_dynamic_tracker(mode=mode)
            for k in range(61):tracker.update([ellipse(1+.002*k+.0001*k*k,.001*k)],k*.1)
            obs=tracker.forecast(tracker.snapshot(6.)[0],6.03);before=obs['prediction_centers'].copy()
            predicted=progress_centers(obs,np.arange(11)*.1,.1)
            np.testing.assert_allclose(predicted,before,atol=1e-12)
            np.testing.assert_array_equal(obs['prediction_centers'],before)

    def test_long_preview_keeps_original_nodes_and_holds_later_axes(self):
        obs=crossing();obs['prediction_axes']=np.column_stack((.3+.01*np.arange(11),.22+.005*np.arange(11)))
        extended=progress_obstacles([obs],np.arange(41)*.1,self.cfg)[0]
        np.testing.assert_array_equal(extended['prediction_axes'][:11],obs['prediction_axes'])
        np.testing.assert_allclose(extended['prediction_centers'][:11],obs['prediction_centers'],atol=1e-12)
        np.testing.assert_array_equal(extended['prediction_axes'][11:],np.tile(obs['prediction_axes'][-1],(30,1)))

    def test_original_nominal_target_nodes_are_not_replaced(self):
        goals=self.goals.copy();goals[:,1]=.005*np.arange(11)**2
        targets=progress_targets(goals,np.arange(41)*.1,.1)
        np.testing.assert_allclose(targets[:11],goals[:,:2],atol=1e-14)
        np.testing.assert_allclose(targets[11],goals[-1,:2]+goals[-1,:2]-goals[-2,:2])

    def test_preview_obeys_same_unicycle_and_input_limits(self):
        for mode in ('direct','yield','ccw','cw'):
            result=progress_preview(self.state,self.goals,[crossing()],0,mode,self.cfg,self.previous,(-.22,.22))
            np.testing.assert_allclose(result['states'],rollout(self.state,result['controls'],.1),atol=1e-12)
            self.assertLessEqual(abs(result['controls'][:,0]).max(),.22+1e-12)
            self.assertLessEqual(abs(result['controls'][:,1]).max(),2.84+1e-12)

    def test_yield_requires_predicted_release_not_permanent_waiting(self):
        obs=crossing();obs['center']=np.array([2.1,0.])
        obs['guidance_motion']['models'][0].update(position=[2.1,0.],velocity=[.05,0.],acceleration=[0.,0.])
        obs['prediction_centers']=progress_centers(obs,np.arange(11)*.1,.1)
        result=progress_preview(self.state,self.goals,[obs],0,'yield',self.cfg,self.previous,(-.22,.22))
        self.assertIsNone(result['release_time']);self.assertFalse(result['feasible'])

    def test_unsafe_cheaper_preview_is_not_selected(self):
        memory={'progress_previous':self.previous}
        guide,level=continuous_guidance(self.state,self.goals,[crossing()],self.cfg,memory)
        decision=memory['progress_decision'];self.assertEqual(decision['selected'],'ccw')
        self.assertFalse(decision['previews']['cw']['feasible'])
        self.assertLess(decision['previews']['cw']['cost'],decision['previews']['ccw']['cost'])
        self.assertEqual(level,1.);self.assertEqual(guide.shape,(10,2))
        self.assertLessEqual(np.linalg.norm(guide,axis=1).max(),.22+1e-12)

    def test_short_refresh_has_identical_primitive_prefix(self):
        for mode in ('direct','yield','ccw','cw'):
            full=progress_preview(self.state,self.goals,[crossing()],0,mode,self.cfg,self.previous,(-.22,.22))
            short=progress_preview(self.state,self.goals,[crossing()],0,mode,dict(self.cfg,progress_horizon=1.),self.previous,(-.22,.22))
            np.testing.assert_array_equal(full['vectors'][:10],short['vectors'])

    def test_zero_weight_maneuver_hypothesis_is_retained_in_reference_screen(self):
        obs=crossing();obs['center']=np.array([2.1,-.5])
        obs['guidance_motion']['models']=[
            dict(mode='cv',position=[2.1,-.5],velocity=[0.,0.],acceleration=[0.,0.],tau=1.),
            dict(mode='ca',position=[2.1,-.5],velocity=[0.,.15],acceleration=[0.,.03],tau=1.)]
        obs['guidance_motion']['weights']=[1.,0.]
        obs['prediction_centers']=progress_centers(obs,np.arange(11)*.1,.1)
        result=progress_preview(self.state,self.goals,[obs],0,'direct',self.cfg,self.previous,(-.22,.22))
        self.assertGreater(result['min_h'],0.)
        self.assertLess(result['hypothesis_min_h']['crossing/1/ca'],0.)
        self.assertFalse(result['feasible'])
        memory={'progress_previous':self.previous}
        continuous_guidance(self.state,self.goals,[obs],self.cfg,memory)
        self.assertGreater(memory['progress_summary']['decision_id'],0)

    def test_static_and_no_obstacle_paths_preserve_legacy_guidance(self):
        static=dict(name='static',center=np.array([2.,0.]),a=.2,b=.15,theta=.2)
        inactive=dict(name='inactive',center=np.array([1000.,1000.]),a=.1,b=.1,theta=0.,guidance=False)
        for obstacles in ([],[inactive,static]):
            expected=_legacy_continuous_guidance(self.state,self.goals,obstacles,self.cfg,{})
            actual=continuous_guidance(self.state,self.goals,obstacles,self.cfg,{})
            np.testing.assert_array_equal(actual[0],expected[0]);self.assertEqual(actual[1],expected[1])


if __name__=='__main__':unittest.main()
