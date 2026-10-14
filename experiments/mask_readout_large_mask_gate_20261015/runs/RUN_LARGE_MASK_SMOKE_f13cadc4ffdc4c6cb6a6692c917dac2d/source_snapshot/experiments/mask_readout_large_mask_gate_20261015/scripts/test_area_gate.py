import copy
import unittest
import numpy as np
from pycocotools import mask as masks
from area_gate import mix_row, adjudicate


def row(n):
    mask = np.zeros((100, 100), dtype=np.uint8, order='F')
    mask.ravel(order='F')[:n] = 1
    rle=masks.encode(mask);rle['counts']=rle['counts'].decode('ascii')
    return dict(image_id=7,category_id=1,score=.8,bbox=[0,0,100,100],segmentation=rle)


class GateTests(unittest.TestCase):
    def test_fixed_boundary_uses_native_area(self):
        a,b=row(9215),row(10000)
        original=copy.deepcopy((a,b))
        result,selection=mix_row(a,b)
        self.assertEqual(result,a);self.assertFalse(selection['apply_delta'])
        self.assertEqual((a,b),original)
        a,b=row(9216),row(100)
        result,selection=mix_row(a,b)
        self.assertEqual(result,b);self.assertTrue(selection['apply_delta'])
        self.assertEqual(selection['native_mask_area'],9216)

    def test_empty_native_and_nonmask_drift(self):
        result,selection=mix_row(row(0),row(10000))
        self.assertFalse(selection['apply_delta']);self.assertEqual(result,row(0))
        bad=row(100);bad['score']=.7
        with self.assertRaises(ValueError):mix_row(row(100),bad)

    def test_original_gate_strictness_and_precision(self):
        t=dict(mean=.005,ci95=[0.,.01]);o=dict(mean=-.001)
        self.assertEqual(adjudicate(t,o,dict(gate='damage_guardrail_passed'))[0],'failed')
        t['ci95'][0]=.00001
        self.assertEqual(adjudicate(t,o,dict(gate='damage_guardrail_passed'))[0],'passed')
        self.assertEqual(adjudicate(t,o,dict(gate='blocked_insufficient_precision'))[0],'blocked')


if __name__=='__main__':unittest.main()
