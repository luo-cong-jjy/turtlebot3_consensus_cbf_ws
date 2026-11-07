"""Time-align a measured contour without changing KF means or uncertainty."""
import unittest
import numpy as np
from test_cbf_dynamic import inline,ellipse


def learned(mode):
    model=inline.make_dynamic_tracker(mode=mode,transport_geometry=True,reaction_time=0.,extra_margin=0.)
    for k in range(60):model.update([ellipse(1+.02*k,0.)],.1*k)
    return model


class GeometryTimeChecks(unittest.TestCase):
    def test_an_old_snapshot_without_a_measurement_reference_keeps_the_legacy_guard(self):
        model=learned('cv');track=model.snapshot(5.9)[0];track.pop('measurement_reference_center')
        new=model.forecast(track,6.2);model.transport_geometry=False;old=model.forecast(track,6.2)
        self.assertFalse(new['geometry_transport_applied'])
        np.testing.assert_array_equal(new['prediction_axes'],old['prediction_axes'])

    def test_rigid_geometry_guard_does_not_expand_just_because_the_contour_is_old(self):
        model=learned('cv');track=model.snapshot(5.9)[0];guards=[]
        for age in [0.,.2,.5]:
            result=model.forecast(track,5.9+age);_,p=model.propagate(track['x'],track['P'],5.9+age-track['stamp'])
            guards.append(result['prediction_axes'][0]-inline.reference_axis_inflation(p))
        np.testing.assert_allclose(guards,np.tile(guards[0],(3,1)),atol=1e-12)
        model.transport_geometry=False;old=model.forecast(track,6.4)
        model.transport_geometry=True;new=model.forecast(track,6.4)
        self.assertGreater(old['a'],new['a']+.02)

    def test_transported_measurement_ellipse_is_contained_for_all_predictors(self):
        for mode in ['cv','ca','singer','adaptive']:
            model=learned(mode);track=model.snapshot(5.9)[0];result=model.forecast(track,6.2)
            raw=track['banks']['cv'] if mode=='adaptive' else track
            angle=np.linspace(-np.pi,np.pi,361)
            c,s=np.cos(raw['raw_theta']),np.sin(raw['raw_theta']);rotation=np.array([[c,-s],[s,c]])
            points=np.column_stack((raw['raw_a']*np.cos(angle),raw['raw_b']*np.sin(angle)))@rotation.T+result['guard_measurement_center']
            c,s=np.cos(result['theta']),np.sin(result['theta']);local=(points-result['center'])@np.array([[c,-s],[s,c]])
            self.assertLessEqual(np.max(np.sum((local/[result['a'],result['b']])**2,axis=1)),1.+1e-10)
            self.assertTrue(result['geometry_transport_applied'])

    def test_transport_can_expand_the_guard_and_is_not_a_uniform_axis_reduction(self):
        model=inline.DynamicEllipseTracker(transport_geometry=True,reaction_time=0.,extra_margin=0.)
        model.update([ellipse(.08,0.)],0.);track=model.snapshot(0.)[0]
        track['x'][:2]=0.;track['x'][5]=.2;track['measurement_reference_center'][:]=0.
        new=model.forecast(track,.3);model.transport_geometry=False;old=model.forecast(track,.3)
        self.assertGreater(new['a'],old['a'])

    def test_means_covariances_and_fresh_measurement_geometry_are_preserved(self):
        for mode in ['cv','ca','singer','adaptive']:
            model=learned(mode);track=model.snapshot(5.9)[0]
            for age in [0.,.3]:
                model.transport_geometry=True
                if mode=='adaptive':
                    for bank in model.filters.values():bank.transport_geometry=True
                new=model.forecast(track,5.9+age)
                model.transport_geometry=False
                if mode=='adaptive':
                    for bank in model.filters.values():bank.transport_geometry=False
                old=model.forecast(track,5.9+age)
                for key in ['prediction_centers','prediction_velocities','prediction_position_covariances','prediction_execution_margins']:
                    np.testing.assert_allclose(new[key],old[key],atol=1e-12)
                if age==0.:np.testing.assert_allclose(new['prediction_axes'],old['prediction_axes'],atol=1e-12)

    def test_missing_measurements_keep_the_last_measurement_reference_immutable(self):
        model=learned('cv');snapshot=model.snapshot(5.9)[0];reference=snapshot['measurement_reference_center'].copy()
        model.update([],6.1);later=model.snapshot(6.1)[0];result=model.forecast(later,6.2)
        np.testing.assert_array_equal(later['measurement_reference_center'],reference)
        np.testing.assert_array_equal(snapshot['measurement_reference_center'],reference)
        self.assertAlmostEqual(result['geometry_reference_stamp'],5.9)
        self.assertAlmostEqual(result['measurement_age'],.3)
        model.update([ellipse(2.24,0.)],6.2)
        np.testing.assert_array_equal(snapshot['measurement_reference_center'],reference)


if __name__=='__main__':unittest.main()
