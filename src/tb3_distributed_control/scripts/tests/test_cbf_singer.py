#!/usr/bin/env python3
"""Singer propagation independent checks and CA ablation separation."""
from pathlib import Path
import unittest
import numpy as np
from scipy.linalg import expm
from test_cbf_dynamic import inline,ellipse


class SingerChecks(unittest.TestCase):
    def test_singer_transition_and_diffusion_match_continuous_matrix_exponential(self):
        for tau in (.02,.1,1.,4.):
            tracker=inline.DynamicEllipseTracker(mode='singer',singer_tau=tau)
            for h in (.0001,.1,.65,1.3):
                f=np.array([[0.,1.,0.],[0.,0.,1.],[0.,0.,-1/tau]])
                # Vectorized Lyapunov ODE; stable eigenvalues even for h/tau >> 1.
                generator=np.zeros((10,10));generator[:9,:9]=np.kron(f,np.eye(3))+np.kron(np.eye(3),f)
                diffusion=np.zeros((3,3));diffusion[2,2]=.01;generator[:9,9]=diffusion.ravel()
                expected_q=expm(generator*h)[:9,9].reshape(3,3)
                state=np.arange(9,dtype=float)/10
                actual,q=tracker.propagate(state,np.zeros((9,9)),h)
                indices=[0,5,7]
                np.testing.assert_allclose(actual[indices],expm(f*h)@state[indices],atol=1e-12)
                np.testing.assert_allclose(q[np.ix_(indices,indices)],expected_q,rtol=1e-8,atol=1e-12)
                self.assertGreaterEqual(np.linalg.eigvalsh(q).min(),-1e-12)

    def test_singer_semigroup_and_ca_limit(self):
        x=np.arange(9)/10;p=np.eye(9)*.03
        tracker=inline.DynamicEllipseTracker(mode='singer',singer_tau=.1)
        full,fp=tracker.propagate(x,p,.65)
        first,ap=tracker.propagate(x,p,.2);split,sp=tracker.propagate(first,ap,.45)
        np.testing.assert_allclose(split,full,atol=1e-12);np.testing.assert_allclose(sp,fp,atol=1e-12)
        singer=inline.DynamicEllipseTracker(mode='singer',singer_tau=1e5)
        ca=inline.DynamicEllipseTracker(mode='ca')
        sx,sp=singer.propagate(x,p,.1);cx,cp=ca.propagate(x,p,.1)
        np.testing.assert_allclose(sx,cx,atol=1e-6);np.testing.assert_allclose(sp,cp,atol=1e-6)

    def test_acceleration_decays_without_snapshot_mutation(self):
        tracker=inline.DynamicEllipseTracker(mode='singer',singer_tau=1.)
        for k in range(61):
            t=.1*k;tracker.update([ellipse(1+.05*t+.5*.02*t*t,0.)],t)
        track=tracker.snapshot(6.)[0];original=track['x'].copy();obs=tracker.forecast(track,6.)
        velocity=obs['prediction_velocities'];increments=np.linalg.norm(np.diff(velocity,axis=0),axis=1)
        self.assertTrue(np.all(np.diff(increments)<0))
        np.testing.assert_array_equal(track['x'],original)
        self.assertGreater(np.linalg.norm(obs['acceleration']),.005)

    def test_ablation_only_changes_margin_or_guidance_flag(self):
        trackers=[inline.DynamicEllipseTracker(mode='ca'),
            inline.DynamicEllipseTracker(mode='ca',acceleration_buffer=False),
            inline.DynamicEllipseTracker(mode='ca',stage_velocity_guidance=False)]
        for k in range(61):
            t=.1*k
            for tracker in trackers:tracker.update([ellipse(1+.04*t+.5*.025*t*t,0.)],t)
        full,no_buffer,no_guide=[t.forecast(t.snapshot(6.)[0],6.) for t in trackers]
        for obs in (no_buffer,no_guide):
            np.testing.assert_array_equal(obs['prediction_centers'],full['prediction_centers'])
            np.testing.assert_array_equal(obs['prediction_position_covariances'],full['prediction_position_covariances'])
        self.assertTrue(np.all(no_buffer['prediction_axes']<full['prediction_axes']))
        np.testing.assert_array_equal(no_guide['prediction_axes'],full['prediction_axes'])
        self.assertFalse(no_guide['stage_velocity_guidance'])

    def test_original_ca_predictions_unchanged(self):
        fixtures=Path(__file__).parent/'fixtures'
        data=np.load(fixtures/'cv_tracker_reference.npz')
        expected=np.load(fixtures/'ca_tracker_reference.npz')
        tracker=inline.DynamicEllipseTracker(mode='ca')
        for stamp,cx,cy,a,b,theta in data['measurements']:
            tracker.update([inline.Ellipse(cx,cy,a,b,theta,20)],stamp)
        obs=tracker.forecast(tracker.snapshot(float(data['measurements'][-1,0]))[0],float(data['now']))
        for key,value in [('state',tracker.tracks[1]['x']),('covariance',tracker.tracks[1]['P']),
                ('centers',obs['prediction_centers']),('axes',obs['prediction_axes']),('margins',obs['prediction_execution_margins'])]:
            np.testing.assert_allclose(value,expected[key],rtol=0,atol=1e-12)



if __name__=='__main__':unittest.main()
