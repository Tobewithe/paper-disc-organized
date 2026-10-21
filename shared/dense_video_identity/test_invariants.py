"""Functional invariants: real causality, identity uniqueness, exact metric sanity."""
import unittest
from pathlib import Path
import numpy as np
from .tracking import BoundedRecovery, quality_vector
from .common import unit


class RecoveryTests(unittest.TestCase):
    def obs(self,bid,feature,x=10):
        f=unit(np.asarray(feature,np.float32))
        return dict(base_id=bid,det_id=0,box=np.array([x,10,x+20,30.]),feature=f,memory=f,score=.9,write=True)
    def recovery(self):
        return BoundedRecovery(dict(max_delay_frames=6,min_tracklet_frames=3,min_similarity=.75,min_margin=.05,max_normalized_distance=2.),60)
    def test_no_future_rewrite_and_confirmed_gap(self):
        r=self.recovery();first=r.update(1,[self.obs(1,[1,0])]);identity=first[0]["identity"]
        r.update(2,[]);r.update(3,[])
        previous=r.update(4,[self.obs(2,[1,0])])
        self.assertIsNone(previous[0]["identity"])
        r.update(5,[self.obs(2,[1,0])]);last=r.update(6,[self.obs(2,[1,0])])
        self.assertEqual(last[0]["identity"],identity)
        self.assertIsNone(previous[0]["identity"])
        self.assertEqual(r.events[0]["delay_frames"],2)
    def test_ambiguous_recovery_abstains_and_never_duplicates(self):
        r=self.recovery();r.update(1,[self.obs(1,[1,0]),self.obs(2,[1,0],14)])
        r.update(2,[]);r.update(3,[])
        for f in range(4,11):
            out=r.update(f,[self.obs(3,[1,0],12),self.obs(4,[1,0],13)])
            ids=[o["identity"] for o in out if o["identity"] is not None]
            self.assertEqual(len(ids),len(set(ids)))
        self.assertFalse(any(e["type"]=="recovered" for e in r.events))
        self.assertEqual(len([e for e in r.events if e["type"]=="abstained_new_id"]),2)
    def test_quality_has_no_future_or_gt_input(self):
        q=quality_vector(np.array([1.,0]),np.array([1.,0]),[.8,.4,.3,.02],[np.array([0.,1])],10,2)
        self.assertEqual(q.shape,(8,));self.assertAlmostEqual(q[5],1.)
    def test_deadline_advances_even_when_provisional_disappears(self):
        r=self.recovery();old=r.update(1,[self.obs(1,[1,0])])[0]["identity"]
        r.update(2,[]);r.update(3,[])
        self.assertIsNone(r.update(4,[self.obs(2,[1,0])])[0]["identity"])
        for frame in range(5,30):r.update(frame,[])
        final=r.update(30,[self.obs(2,[1,0])])
        self.assertNotEqual(final[0]["identity"],old)
        self.assertTrue(all(e["delay_frames"]<=6 for e in r.events))


if __name__=="__main__":
    unittest.main()
