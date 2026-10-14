import unittest
import torch
from torch.nn import functional as F
from decomposition_statistics import GROUPS, partition, iou_terms, adjudicate


class DecompositionTests(unittest.TestCase):
    def test_all_eight_cells_strict_threshold_and_common_denominator(self):
        gt = torch.tensor([[[0.,0.,0.,0.,1.,1.,1.,1.,1.]]])
        a = torch.tensor([[[0.,-1.,1.,1.,-1.,-1.,1.,1.,50.]]])
        b = torch.tensor([[[-2.,1.,-1.,2.,-2.,1.,0.,2.,-50.]]])
        roi = torch.tensor([[[True]*8+[False]]])
        l0, l1 = F.binary_cross_entropy_with_logits(a,gt,reduction='none'), F.binary_cross_entropy_with_logits(b,gt,reduction='none')
        result = partition(a,b,gt,roi,l0,l1)[0]
        self.assertEqual([result['groups'][g]['pixels'] for g in GROUPS], [1]*8)
        self.assertAlmostEqual(result['totals']['zero_loss'], float(l0.double()[roi].mean()), places=14)
        self.assertAlmostEqual(result['totals']['B_loss'], float(l1.double()[roi].mean()), places=14)
        for g in ('TP_to_FN','TN_to_FP'):
            self.assertEqual(result['groups'][g]['decrease_credit'], 0.)
            self.assertGreater(result['groups'][g]['increase_debit'], 0.)

    def test_stable_confidence_can_pay_for_harmful_flip(self):
        a=torch.tensor([[[.1,.1,.1]]]); b=torch.tensor([[[5.,5.,-.1]]]); gt=torch.ones_like(a); roi=torch.ones_like(a,dtype=torch.bool)
        result=partition(a,b,gt,roi,F.binary_cross_entropy_with_logits(a,gt,reduction='none'),F.binary_cross_entropy_with_logits(b,gt,reduction='none'))[0]
        self.assertLess(result['totals']['delta_loss'],0)
        self.assertLess(result['iou_decomposition']['delta_iou'],0)
        self.assertEqual(result['stable_credit_share'],1)

    def test_iou_decomposition_mixed_and_empty_union(self):
        r=iou_terms({'tp':8,'fn':2,'fp':2,'tn':8},{'tp':7,'fn':3,'fp':4,'tn':6})
        self.assertAlmostEqual(r['foreground_TP_term']+r['background_FP_term'],7/14-8/12,places=14)
        e=iou_terms({'tp':0,'fn':0,'fp':0,'tn':5},{'tp':0,'fn':0,'fp':2,'tn':3})
        self.assertEqual(e['foreground_TP_term'],0)
        self.assertEqual(e['background_FP_term'],-1)
        self.assertTrue(e['empty_union_boundary'])

    def test_strict_majority(self):
        self.assertFalse(adjudicate({'stable_credit_majority_fraction':.5})['hypothesis_supported'])
        self.assertTrue(adjudicate({'stable_credit_majority_fraction':.5001})['hypothesis_supported'])

    def test_empty_roi_rejected(self):
        t=torch.zeros(1,1,2)
        with self.assertRaises(ValueError):
            partition(t,t,t,torch.zeros_like(t,dtype=torch.bool),t,t)


if __name__=='__main__':
    unittest.main()
