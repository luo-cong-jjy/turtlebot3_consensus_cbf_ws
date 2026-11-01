from pathlib import Path
import unittest
import numpy as np
from test_cbf_dynamic import inline,ellipse


class ReliabilityChecks(unittest.TestCase):
    def test_between_model_covariance_is_not_discarded(self):
        mean,p=inline.mixture_moments([[0.,0.],[2.,0.]],[np.eye(2)*.1,np.eye(2)*.1],[.5,.5])
        np.testing.assert_allclose(mean,[1.,0.]);np.testing.assert_allclose(p,[[1.1,0.],[0.,.1]])
        self.assertGreater(np.linalg.eigvalsh(p).min(),0.)

    def test_cold_start_has_no_score_from_future_and_preserves_cv_forecast(self):
        a=inline.make_dynamic_tracker(mode='adaptive');b=inline.DynamicEllipseTracker()
        for k in range(8):
            t=.1*k;a.update([ellipse(1+.02*k,0.)],t);b.update([ellipse(1+.02*k,0.)],t)
        track=a.snapshot(.7)[0];self.assertEqual(track['score_count'],0)
        np.testing.assert_array_equal(track['weights'],[1.,0.,0.])
        adaptive=a.forecast(track,.73);cv=b.forecast(b.snapshot(.7)[0],.73)
        for key in ('prediction_centers','prediction_axes','prediction_position_covariances'):
            np.testing.assert_allclose(adaptive[key],cv[key],atol=1e-12)

    def test_experts_share_one_association_even_when_detections_reorder(self):
        tracker=inline.make_dynamic_tracker(mode='adaptive',association_distance=.35)
        for k in range(41):
            t=.1*k;fits=[ellipse(-.4+.2*t,0.),ellipse(.4-.2*t,.08)]
            if k%2:fits.reverse()
            tracker.update(fits,t)
            for mode,bank in tracker.filters.items():
                for fit in fits:
                    self.assertAlmostEqual(bank.tracks[fit.track_id]['raw_center'][0],fit.cx)
        self.assertEqual(set(tracker.tracks),{1,2})
        self.assertGreater(tracker.tracks[1]['x'][5],.1)
        self.assertLess(tracker.tracks[2]['x'][5],-.1)

    def test_delayed_scoring_is_causal_and_weights_change_smoothly(self):
        tracker=inline.make_dynamic_tracker(mode='adaptive');previous=np.array([1.,0.,0.])
        for k in range(161):
            t=.1*k;u=max(0.,min(t-5.,7.))
            x=1+.03*t+.5*.025*u*u+.175*max(0.,t-12.)
            tracker.update([ellipse(x,0.)],t);track=tracker.tracks[1]
            self.assertLessEqual(np.abs(track['weights']-previous).sum(),2*(-np.expm1(-.1/.3))+1e-10)
            previous=track['weights'].copy()
            if track['score_count']:
                self.assertLessEqual(track['score_origin'],t-1.+1e-9)
                self.assertGreaterEqual(track['scores'].min(),0.)
            if k==100:self.assertGreater(track['weights'][1],.9)
        self.assertGreater(tracker.tracks[1]['weights'][0],.9)

    def test_future_suffix_cannot_change_past_outputs(self):
        a=inline.make_dynamic_tracker(mode='adaptive');b=inline.make_dynamic_tracker(mode='adaptive')
        saved_a=[];saved_b=[]
        for k in range(60):
            t=.1*k;x=1+.01*k
            a.update([ellipse(x,0.)],t);b.update([ellipse(x+(2. if k>40 else 0.),0.)],t)
            if k<=40:
                saved_a.append(a.forecast(a.snapshot(t)[0],t)['prediction_centers'])
                saved_b.append(b.forecast(b.snapshot(t)[0],t)['prediction_centers'])
        np.testing.assert_array_equal(saved_a,saved_b)

    def test_snapshot_prediction_is_immutable_and_covariance_psd(self):
        tracker=inline.make_dynamic_tracker(mode='adaptive')
        for k in range(100):tracker.update([ellipse(1+.002*k+.0001*k*k,0.)],k*.1)
        snapshot=tracker.snapshot(9.9)[0];old=tracker.forecast(snapshot,10.)
        tracker.update([ellipse(2.3,.1)],10.1)
        again=tracker.forecast(snapshot,10.)
        for key in ('prediction_centers','prediction_axes','prediction_position_covariances','model_weights'):
            np.testing.assert_array_equal(old[key],again[key])
        self.assertTrue(np.all(np.linalg.eigvalsh(old['prediction_position_covariances'])>=-1e-12))

    def test_single_expert_limit_matches_all_three_expert_envelopes(self):
        tracker=inline.make_dynamic_tracker(mode='adaptive')
        for k in range(61):tracker.update([ellipse(1+.002*k+.0001*k*k,0.)],k*.1)
        for index,mode in enumerate(tracker.MODES):
            snapshot=tracker.snapshot(6.)[0];snapshot['weights']=np.eye(3)[index]
            actual=tracker.forecast(snapshot,6.03)
            expected=tracker.filters[mode].forecast(snapshot['banks'][mode],6.03)
            for key in ('prediction_centers','prediction_axes','prediction_position_covariances'):
                np.testing.assert_allclose(actual[key],expected[key],atol=1e-12)

    def test_duplicate_clock_misses_expiry_and_reset(self):
        tracker=inline.make_dynamic_tracker(mode='adaptive',timeout=.6)
        for k in range(21):tracker.update([ellipse(1+.01*k,0.)],.1*k)
        track=tracker.tracks[1];count=track['score_count'];weights=track['weights'].copy()
        tracker.update([ellipse(100.,0.)],2.)
        self.assertEqual(track['score_count'],count);np.testing.assert_array_equal(track['weights'],weights)
        tracker.update([],2.2);self.assertEqual(tracker.tracks[1]['score_count'],count)
        self.assertEqual(len(tracker.snapshot(2.4)),1)
        self.assertEqual(tracker.obstacles(tracker.snapshot(2.4),2.7),[])
        tracker.update([],2.7);self.assertFalse(tracker.tracks)
        tracker.update([ellipse(0.,0.)],.1);self.assertEqual(list(tracker.tracks),[1])
        self.assertEqual(tracker.tracks[1]['score_count'],0)

    def test_invalid_reliability_configuration_and_forced_duplicate_ids(self):
        for kwargs in (dict(score_tau=0),dict(weight_tau=float('nan')),dict(switch_ratio=1.)):
            with self.assertRaises(ValueError):inline.make_dynamic_tracker(mode='adaptive',**kwargs)
        with self.assertRaises(ValueError):
            inline.DynamicEllipseTracker().update([ellipse(0.,0.),ellipse(1.,0.)],1.,forced_ids=[1,1])


if __name__=='__main__':unittest.main()
