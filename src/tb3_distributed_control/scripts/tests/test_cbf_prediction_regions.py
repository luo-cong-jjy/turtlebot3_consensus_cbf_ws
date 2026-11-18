"""Display geometry must equal h=0 and remove stale future regions."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock
import unittest
import numpy as np
from test_cbf_inline_geometry import inline, Stamp
from cbf_mpc_core import default_config
from mpc_dcbf_cached_solver import barrier_values,obstacle_batch,progress_obstacles

inline.obstacle_batch=obstacle_batch;inline.progress_obstacles=progress_obstacles


class PredictionRegionsChecks(unittest.TestCase):
    def inputs(self):
        cfg=default_config();centers=np.column_stack((1+.02*np.arange(11),.01*np.arange(11)))
        axes=np.column_stack((.4+.003*np.arange(11),.2+.002*np.arange(11)))
        obs=dict(name='tracked_7',id=7,center=centers[0],a=.4,b=.2,theta=.7,
            prediction_centers=centers,prediction_axes=axes,prediction_angles=.7+.01*np.arange(11))
        return cfg,[obs]

    def test_rotated_filled_regions_are_exact_hard_cbf_zero_boundary(self):
        cfg,obs=self.inputs();before=copy.deepcopy(obs)
        regions=inline.prediction_region_geometry(obs,cfg,2.)
        self.assertEqual([r['stage'] for r in regions],[0,5,10,15,20])
        for r in regions:
            if not r['hard']:continue
            j=r['stage'];np.testing.assert_array_equal(r['center'],obs[0]['prediction_centers'][j])
            np.testing.assert_array_equal([r['prediction_a'],r['prediction_b']],obs[0]['prediction_axes'][j])
            contour=inline.Ellipse(*r['center'],r['a'],r['b'],r['theta'],0).contour()
            original=dict(center=r['center'],a=obs[0]['prediction_axes'][j,0],
                b=obs[0]['prediction_axes'][j,1],theta=obs[0]['prediction_angles'][j])
            h=barrier_values(np.column_stack((contour,np.zeros(len(contour)))),[original],cfg['margin'])
            np.testing.assert_allclose(h,0.,atol=1e-10)
        np.testing.assert_array_equal(obs[0]['prediction_axes'],before[0]['prediction_axes'])
        np.testing.assert_array_equal(obs[0]['prediction_centers'],before[0]['prediction_centers'])

    def test_soft_regions_hold_final_axes_and_do_not_invent_peer_tracks(self):
        cfg,obs=self.inputs();obs.append(dict(obs[0],name='peer_1',id=1,peer=True))
        regions=inline.prediction_region_geometry(obs,cfg,2.)
        self.assertEqual({r['id'] for r in regions},{7})
        hard=next(r for r in regions if r['stage']==10)
        for r in regions:
            if r['hard']:continue
            self.assertEqual((r['a'],r['b']),(hard['a'],hard['b']))
            self.assertGreater(r['center'][0],hard['center'][0])

    def test_expired_input_deletes_only_previous_regions(self):
        inline.rospy=SimpleNamespace(Time=Stamp,Duration=Stamp)
        inline.Marker=SimpleNamespace(CYLINDER=3,DELETE=2)
        inline.MarkerArray=lambda:SimpleNamespace(markers=[])
        n=inline.TurtleBot3CBFController.__new__(inline.TurtleBot3CBFController)
        n.closed=False;n.prediction_region_pub=Mock();n.prediction_region_pub.get_num_connections.return_value=1
        n.display_prediction_pub=None;n.prediction_display_enabled=True;n.scan_timeout=.65
        n.prediction_view_input=(Stamp(1.),[]);n.prediction_visible_keys={('prediction_hard_10',7)}
        n.geometry_marker=lambda *args:SimpleNamespace(ns=args[2],id=args[1],action=0)
        Stamp.current=2.;n.publish_prediction_regions(None)
        markers=n.prediction_region_pub.publish.call_args[0][0].markers
        self.assertEqual([(m.ns,m.id,m.action) for m in markers],[('prediction_hard_10',7,2)])
        self.assertFalse(n.prediction_visible_keys)

    def test_direction_uses_predicted_translation_not_ellipse_angle(self):
        cfg,obs=self.inputs();obs[0]['prediction_centers'][:,0]=1-.004*np.arange(11)
        regions=inline.prediction_region_geometry(obs,cfg,2.)
        motion=inline.prediction_motion_annotation(regions)
        expected=np.array([-.04,.1]);np.testing.assert_allclose(motion['velocity'],expected)
        direction=motion['direction_end']-motion['center']
        self.assertAlmostEqual(np.linalg.norm(direction),.32)
        np.testing.assert_allclose(direction/.32,expected/np.linalg.norm(expected))

    def test_nearly_static_estimate_has_no_fake_direction_arrow(self):
        cfg,obs=self.inputs();obs[0]['prediction_centers'][:]=[1.,.2]
        motion=inline.prediction_motion_annotation(inline.prediction_region_geometry(obs,cfg,2.))
        self.assertEqual(motion['speed'],0.)
        self.assertIsNone(motion['direction_end']);self.assertTrue(np.isfinite(motion['normal']).all())
