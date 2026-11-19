"""Local diagnostics must not turn into raw-lidar communication."""
import unittest
import copy
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from test_cbf_inline_geometry import inline


class DualLocalObservationChecks(unittest.TestCase):
    def test_private_copy_does_not_mutate_public_prediction(self):
        public=dict(observer='tb3_1',stamp=1.,tracks=[],measurements=[dict(center=[1,2],a=.2,b=.2,theta=0.,point_count=3)])
        before=copy.deepcopy(public);points=np.array([[1.,2.],[1.1,2.],[1.,2.1]])
        private=inline.local_cluster_record(public,[points],[0.,0.],'world')
        self.assertEqual(public,before);self.assertNotIn('points',public['measurements'][0])
        self.assertEqual(private['observer'],'tb3_1');self.assertEqual(private['points_frame'],'world')
        np.testing.assert_array_equal(private['measurements'][0]['points'],points)

    def test_new_two_robot_profile_preserves_frozen_team_geometry(self):
        p=Path(__file__).resolve().parents[2]/'launch/two_turtlebot3_cbf_motion_geometry.launch'
        root=ET.parse(p).getroot();args={a.attrib['name']:a.attrib.get('default') for a in root.findall('arg')}
        self.assertEqual(args['motion_geometry_observer'],'circle');self.assertEqual(args['circle_prior_retention'],'true')
        self.assertEqual(args['spatial_activation_threshold'],'0.22');self.assertEqual(args['spatial_predictive_return'],'true')
        include=root.find('include');values={a.attrib['name']:a.attrib['value'] for a in include.findall('arg')}
        self.assertEqual(values['robot_count'],'2');self.assertEqual(values['formation_radius'],'0.5')
        self.assertEqual(values['robot0_init_x'],'0.5');self.assertEqual(values['robot1_init_x'],'-0.5')
        for scope in [root,include]:
            names=[a.attrib['name'] for a in scope.findall('arg')];self.assertEqual(len(names),len(set(names)))
