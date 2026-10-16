import unittest
import numpy as np
from math_ops import weighted_median,draw_directions
from audit_statistics import summarize
from verify import spans,independent_values

CONFIG=dict(seed=20261017,donors_per_pair=8,bootstrap_resamples=2000,minimum_effect=.005,pressure_arms=['S','R','M','E','MU+','MU-'])


class TestCET(unittest.TestCase):
    def test_independent_compressed_rle(self):
        self.assertEqual(spans(dict(size=[2,2],counts='4')),[])
        self.assertEqual(spans(dict(size=[2,2],counts='04')),[(0,4)])
        self.assertEqual(spans(dict(size=[2,2],counts='121')),[(1,3)])
        self.assertEqual(spans(dict(size=[2,4],counts='032N0')),[(0,3),(5,6)])
        result=independent_values(dict(size=[2,2],counts='121'),[(0,2)],[(0,4)])
        self.assertEqual(result,dict(iou=1/3,intersection=1,union=3,native_TP_to_FN=1,native_FP_to_TN=1))

    def test_weighted_median_uses_images(self):
        self.assertEqual(weighted_median(np.array([1.,2.,10.]),np.array([.25,.25,.5])),2.)

    def test_draw_antisymmetry_and_mismatch(self):
        errors=np.arange(128,dtype=float).reshape(4,32);mu=errors.mean(0);ids=np.array([1,1,2,2])
        a,labels=draw_directions(errors,mu,ids,100,3,CONFIG);b,unused=draw_directions(errors,mu,ids,100,3,CONFIG)
        self.assertTrue(np.array_equal(a,b));self.assertEqual(len(a),58)
        for j in range(0,56,7):
            for k in (0,2,4):self.assertTrue(np.array_equal(a[j+k],-a[j+k+1]))
            p=labels[j+4]['permutation'];self.assertTrue(all(i!=x for i,x in enumerate(p)))
            self.assertTrue(np.array_equal(a[j+4],a[j][p]))
        with self.assertRaises(ValueError):draw_directions(errors,mu,ids,1,3,CONFIG)

    def cases(self,s,r,m):
        return [dict(image_id=i,N=.5,C=.8,G=.51,trials=[dict(arm=a,iou=v) for a,v in zip(CONFIG['pressure_arms'],[s,r,m,.6,.6,.6])]) for i in range(12)]

    def test_positive_requires_both_controls(self):
        summary,_,_=summarize(self.cases(.6,.62,.62),CONFIG)
        self.assertEqual(summary['decision'],'TRANSFERABLE_DIRECTIONAL_RISK')
        summary,_,_=summarize(self.cases(.6,.62,.6),CONFIG)
        self.assertEqual(summary['decision'],'NO_MATERIAL_TRANSFER_AT_SCOPE')

    def test_negative_sign_and_units(self):
        summary,_,_=summarize(self.cases(.7,.6,.6),CONFIG)
        self.assertAlmostEqual(summary['effects']['D_S-D_R']['mean'],-.1)
        self.assertEqual(summary['decision'],'NO_MATERIAL_TRANSFER_AT_SCOPE')

    def test_repeats_not_independent_images(self):
        data=self.cases(.6,.62,.62)
        s,draws,ids=summarize(data,CONFIG)
        self.assertEqual(draws.shape,(2000,12));self.assertEqual(len(ids),12)
        for row in data:row['trials']*=8
        q,_,_=summarize(data,CONFIG)
        self.assertAlmostEqual(s['effects']['D_S-D_R']['mean'],q['effects']['D_S-D_R']['mean'])

if __name__=='__main__':unittest.main()
