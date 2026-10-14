"""Behavioral checks against the already-frozen reference loss and join rules."""
import unittest
import torch
from adapter_primitives import grouped_mask_loss
from adapter_model import CoefficientAdapter, vector_grouped_loss, loss_grids
from adapter_data import training_join
from s1_geometry import predicted_roi


class AdapterTests(unittest.TestCase):
    def test_separate_loss_values_and_gradients(self):
        torch.manual_seed(16)
        a = torch.randn(9, 7, 11, dtype=torch.float64, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        baseline = torch.randn_like(a)
        roi = torch.rand_like(a) > .3
        roi[2] = False
        targets = {i: (torch.rand(7, 11)>.5).double() for i in (0, 2, 3)}
        expected, accounting = grouped_mask_loss(a, baseline, roi, targets)
        actual, got = vector_grouped_loss(b, baseline, roi, targets)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-12, rtol=0))
        expected.backward(); actual.backward()
        self.assertTrue(torch.allclose(a.grad, b.grad, atol=1e-12, rtol=0))
        for field in ('matched_supervised_instances','preserved_instances','empty_roi_rows','no_output_image'):
            self.assertEqual(accounting[field], got[field])

    def test_empty_and_unmatched_images(self):
        for count, valid in ((0, False), (3, False), (3, True)):
            logits = torch.zeros(count, 4, 5, requires_grad=True)
            roi = torch.full_like(logits, valid, dtype=torch.bool)
            value, report = vector_grouped_loss(logits, logits.detach(), roi, {})
            value.backward()
            self.assertEqual(report['matched_supervised_instances'], 0)
            self.assertEqual(report['preserved_instances'], count if valid else 0)
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_join_tie_order_and_ignored_gt(self):
        pred = [dict(bbox=[0,0,10,10], category_id=1, score=.9) for _ in range(3)]
        anns = [dict(id=i,bbox=[0,0,10,10],category_id=1,iscrowd=int(i==1)) for i in (3,1,2)]
        self.assertEqual(training_join(pred,anns), {0:2,1:3})
        pred[0]['category_id']=2
        self.assertEqual(training_join(pred,anns), {1:2,2:3})

    def test_zero_initialization_and_full_roi_geometry(self):
        torch.manual_seed(0); b=CoefficientAdapter('B')
        torch.manual_seed(0); c=CoefficientAdapter('C')
        self.assertTrue(all(torch.equal(v,c.state_dict()[n]) for n,v in b.state_dict().items()))
        cache={'image_id':43,'proto':torch.randn(32,8,10),'c0':torch.randn(3,32),
               'boxes_input':torch.tensor([[0.,0,39,31],[-4.,1,20,29],[40,0,41,2]]),
               'input_hw':[32,40],'head_feature_shapes':[[256,4,5],[512,2,3],[512,1,2]],
               'features':[torch.randn(256),torch.randn(512),torch.randn(512)],
               'raw_locations':[{'level':i} for i in range(3)],'scores':torch.ones(3)*.5,'classes':torch.tensor([0,3,5])}
        self.assertTrue(torch.equal(b(cache,'eval:0'), torch.zeros(3,32)))
        self.assertTrue(torch.equal(c(cache,'eval:0'), torch.zeros(3,32)))
        _,_,roi=loss_grids(cache,cache['c0'])
        self.assertTrue(all(torch.equal(roi[i],predicted_roi(cache,i)) for i in range(3)))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
