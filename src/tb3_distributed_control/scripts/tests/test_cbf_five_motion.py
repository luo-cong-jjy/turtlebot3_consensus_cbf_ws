"""Five-body acceptance rejects nonadjacent collision and partial recovery."""
import sys
from pathlib import Path
import unittest
import numpy as np
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_cbf_five_motion import evaluate_five


class FiveMotionTests(unittest.TestCase):
    def inputs(self):
        p=yaml.safe_load((Path(__file__).resolve().parents[2]/'config/cbf_five_robot_protocol.yaml').read_text())
        # Synthetic unthreatened convoy; isolates pair and motion acceptance.
        p['spatial_required']['active_overtaking']=[False]*5
        t=np.linspace(0,40,801)
        xy=np.asarray(p['robot_offsets'])[:,None,:]+np.stack([.15*t,np.zeros_like(t)],axis=1)[None,:,:]
        actor=np.stack([1.35+.05*t,np.full_like(t,5.)],axis=1)
        return t,xy,actor,p

    def test_all_ten_pairs_and_original_motion_requirements(self):
        t,xy,actor,p=self.inputs();r=evaluate_five(t,xy,actor,.18,p,'active_overtaking')
        self.assertTrue(r['passed']);self.assertEqual(len(r['pairs']),10)
        self.assertAlmostEqual(r['peer_proxy_clearance_min'],.8*np.sin(np.pi/5)-.26)

    def test_nonadjacent_crossing_between_samples_rejected(self):
        t,xy,actor,p=self.inputs();xy[0,300]=[2.,.5];xy[2,300]=[3.,.5]
        xy[0,301]=[3.,.5];xy[2,301]=[2.,.5]
        r=evaluate_five(t,xy,actor,.18,p,'active_overtaking')
        self.assertLess(r['pairs']['0_2']['linear_swept_proxy_clearance_min'],0)
        self.assertFalse(r['checks']['linear_swept_peer_proxy_clearance_positive'])

    def test_fifth_robot_fault_and_pair_recovery_not_hidden(self):
        t,xy,actor,p=self.inputs()
        self.assertFalse(evaluate_five(t,xy,actor,.18,p,'active_overtaking',[0,0,0,0,1])['passed'])
        xy[4,-1,0]+=.15
        r=evaluate_five(t,xy,actor,.18,p,'active_overtaking')
        self.assertFalse(r['checks']['relative_formation_recovered'])
        self.assertTrue(r['robots']['tb3_4']['motion']['checks']['tracking_recovered'])


if __name__=='__main__':unittest.main()
