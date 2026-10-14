import unittest
import torch
from torch.nn import functional as F
from test_decomposition import DecompositionTests
from decomposition_v2 import partition,closure_receipt


class RoundingAccountingTests(unittest.TestCase):
    def test_known_failure_roundoff_is_explicit_not_assigned_to_pixels(self):
        d={'totals':{'delta_loss':-.07509910392885408},'direct_zero_BCE_FP64':.6762915298653146,
           'direct_B_BCE_FP64':.6011924259364605}
        receipt=closure_receipt(d,.6762914657592773,.6011924743652344)
        self.assertGreater(abs(receipt['raw_pixel_delta_minus_FP32_delta']),1e-7)
        self.assertLess(receipt['BCE_delta_closure_abs_error'],1e-12)
        self.assertAlmostEqual(receipt['delta_FP32_reduction_roundoff'],1.1253481113637775e-7,places=17)
        self.assertEqual(d['totals']['delta_loss'],-.07509910392885408)

    def test_independent_direct_sum_and_partition(self):
        a=torch.tensor([[[.1,-.2,3.,-4.]]]);b=torch.tensor([[[-.3,-2.,4.,5.]]]);y=torch.tensor([[[1.,0.,1.,0.]]])
        roi=torch.tensor([[[True,True,True,False]]])
        l0=F.binary_cross_entropy_with_logits(a,y,reduction='none');l1=F.binary_cross_entropy_with_logits(b,y,reduction='none')
        d=partition(a,b,y,roi,l0,l1)[0]
        receipt=closure_receipt(d,float((l0*roi).sum()/roi.sum()),float((l1*roi).sum()/roi.sum()))
        self.assertLess(d['FP64_partition_abs_error'],1e-12)
        self.assertLess(receipt['BCE_delta_closure_abs_error'],1e-12)


if __name__=='__main__':unittest.main()
