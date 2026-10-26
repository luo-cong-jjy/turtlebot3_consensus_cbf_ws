#!/usr/bin/env python3
"""Recording regression: unique directories and measured/effective data fidelity."""
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cbf_experiment_results import (create_run_directory,export_results,update_geometry_ranges,
    ORIGINAL_CHARTS,aligned_centroid,formation_snapshots)


class RecordingChecks(unittest.TestCase):
    def test_unique_default_directory_and_existing_explicit_data_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            first,_=create_run_directory(temporary)
            second,_=create_run_directory(temporary)
            self.assertNotEqual(first,second)
            self.assertEqual(first.parent,Path(temporary)/'data_collect')
            file=first/'tb3_0.csv';file.write_text('existing experiment\n')
            with self.assertRaises(FileExistsError):create_run_directory(temporary,str(first))
            self.assertEqual(file.read_text(),'existing experiment\n')

    def test_export_preserves_reference_and_distinguishes_measured_and_size_limited_axes(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)
            (p/'run_metadata.json').write_text(json.dumps(dict(timestamp='test',world_file='')))
            rows={0:[dict(time=0.,x=.3,y=.2,theta=.1,v=.1,omega=.2,
                consensus_error_x=.04,consensus_error_y=.02,reference_x=.26,reference_y=.18,
                observer_x=.1,observer_y=.2,solve_ms=30.,detected_ellipses=4,status='ok',
                heading_error=.03)]}
            stats={}
            for raw in [.06,.08]:
                update_geometry_ranges(stats,[dict(observer='tb3_0',kind='peer',id=1,source='laser',
                    raw_a=raw,raw_b=.04,effective_a=.2,effective_b=.2)])
            with patch('cbf_experiment_results.plot_results'):
                export_results(p,rows,[],stats)
            with (p/'consensus_data_test.pkl').open('rb') as stream:data=pickle.load(stream)
            self.assertEqual(data['robots'][0]['xr'],[.26])
            self.assertEqual(data['robots'][0]['thetae'],[.03])
            self.assertTrue(np.isnan(data['robots'][0]['avoidance_force_magnitude'][0]))
            self.assertIsNone(data['leader'])
            s=data['geometry_statistics']['tb3_0/peer/1']
            self.assertEqual((s['raw_a_min'],s['raw_a_max']),(.06,.08))
            self.assertEqual((s['effective_a_min'],s['effective_a_max']),(.2,.2))
            categories=data['chart_categories']
            self.assertEqual(len(categories),12)
            self.assertEqual(categories['avoidance_forces']['status'],'not_applicable')
            self.assertEqual(categories['obstacle_distance']['status'],'unavailable')
            self.assertEqual(categories['observer_estimation']['status'],'partial')

    def test_original_categories_and_distance_origins_are_preserved(self):
        expected={'trajectory','linear_velocity','angular_velocity','x_error','y_error','heading_error',
            'time_analysis','obstacle_distance','clustering_analysis','avoidance_forces',
            'observer_estimation','tracking_error_xy'}
        self.assertEqual({name for name,title in ORIGINAL_CHARTS},expected)
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary);(p/'run_metadata.json').write_text(json.dumps(dict(timestamp='new')))
            row=dict(time=0.,x=.3,y=.2,theta=.1,v=.1,omega=.2,consensus_error_x=.04,
                consensus_error_y=.02,observer_x=.1,observer_y=.2,solve_ms=120.,detected_ellipses=4,
                detected_peer_ellipses=3,detected_static_ellipses=2,detected_dynamic_ellipses=2,
                total_clusters_count=7,closest_obstacle_distance=.7,
                raw_min_range=.4,status='ok')
            with patch('cbf_experiment_results.plot_results'):data=export_results(p,{0:[row]},[],{})
            self.assertEqual(data['robots'][0]['obstacle_distance'],[.7])
            self.assertEqual(data['robots'][0]['raw_min_distance'],[.4])
            self.assertEqual(data['robots'][0]['total_clusters_count'],[7])
            self.assertEqual(data['robots'][0]['static_clusters_count'],[2])
            self.assertEqual(data['robots'][0]['dynamic_clusters_count'],[2])
            stats=json.loads((p/'experiment_statistics.json').read_text())
            self.assertEqual(stats['0']['overruns'],1)

    def test_centroid_aligns_timestamps_before_averaging(self):
        robots={i:dict(t=[i*.1,1.+i*.1],xc=[i*.1,1.+i*.1],yc=[float(i),float(i)]) for i in range(3)}
        center=aligned_centroid(robots)
        self.assertEqual(center['t'],[1.])
        np.testing.assert_allclose(center['xc'],[1.])
        np.testing.assert_allclose(center['yc'],[1.])

    def test_formation_uses_simultaneous_positions_in_robot_id_order(self):
        robots={}
        for i in reversed(range(5)):
            times=np.arange(21)*.1+i*.01
            robots[i]=dict(t=times.tolist(),xc=(times+i).tolist(),yc=(2*times-i).tolist())
        snapshots=formation_snapshots(robots,times=[.55])
        self.assertEqual(len(snapshots),1)
        np.testing.assert_allclose(snapshots[0]['positions'],[[.55+i,1.1-i] for i in range(5)])

    def test_formation_skips_missing_robots_stale_gaps_and_extrapolation(self):
        robots={i:dict(t=[0.,.1,1.,1.1],xc=[0.,.1,1.,1.1],yc=[i]*4) for i in range(5)}
        snapshots=formation_snapshots(robots,times=[-.1,.05,.5,1.05,1.2])
        self.assertEqual([s['time'] for s in snapshots],[.05,1.05])
        robots[4]=None
        self.assertEqual(formation_snapshots(robots,times=[.05]),[])


if __name__=='__main__':unittest.main()
