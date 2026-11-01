from pathlib import Path
import unittest
import numpy as np
from test_cbf_dynamic import inline
from mpc_dcbf_cached_solver import continuous_guidance,project_relative_velocity
from cbf_mpc_core import default_config


class FrameProjectionChecks(unittest.TestCase):
    def test_feasible_world_velocity_not_shrunk_by_relative_speed(self):
        relative=np.array([.15,-.20]);frame=np.array([0.,.20])
        self.assertGreater(np.linalg.norm(relative),.22)
        np.testing.assert_allclose(project_relative_velocity(relative,frame),relative,atol=1e-15)

    def test_projection_is_world_ball_projection_then_inverse_translation(self):
        rng=np.random.default_rng(91)
        for _ in range(50):
            relative=rng.normal(size=2);frame=rng.normal(size=2)*.2
            world=relative+frame;expected=world*min(1.,.22/np.linalg.norm(world))
            np.testing.assert_allclose(project_relative_velocity(relative,frame)+frame,expected,atol=1e-14)
            self.assertLessEqual(np.linalg.norm(project_relative_velocity(relative,frame)+frame),.22+1e-14)

    def test_legacy_static_cv_ca_guides_match_archived_stage(self):
        data=np.load(Path(__file__).parent/'fixtures/guidance_reference.npz')
        for mode in ('static','cv','ca'):
            centers=np.column_stack((-.008*np.arange(11),.005*np.arange(11)))
            if mode=='ca':centers[:,1]+=.0002*np.arange(11)**2
            obs=dict(name='test',center=np.zeros(2),a=.2,b=.15,theta=.2,dynamic=mode!='static',guidance=mode=='static',prediction_model=mode,prediction_centers=centers)
            guide,level=continuous_guidance(data['state'],data['goals'],[obs],default_config(),{})
            np.testing.assert_allclose(guide,data[mode+'_guide'],rtol=0,atol=1e-12)
            self.assertAlmostEqual(level,float(data[mode+'_level']))

    def test_shifted_frame_guidance_respects_each_stage_world_speed(self):
        cfg=default_config();cfg['moving_frame_projection']=True
        goals=np.zeros((11,3));goals[:,0]=.3+.015*np.arange(11)
        t=.1*np.arange(11);centers=np.column_stack((np.zeros(11),.2*t+.5*.02*t*t))
        obs=dict(name='moving',center=np.zeros(2),a=.2,b=.15,theta=0.,dynamic=True,guidance=False,prediction_model='ca',prediction_centers=centers)
        guide,level=continuous_guidance(np.array([-.6,0.,0.]),goals,[obs],cfg,{})
        self.assertGreater(level,0.)
        self.assertTrue(np.all(np.isfinite(guide)))
        self.assertLessEqual(np.linalg.norm(guide,axis=1).max(),.22+1e-12)


if __name__=='__main__':unittest.main()
