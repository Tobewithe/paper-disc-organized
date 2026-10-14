"""Meaningful loss behavior tests; smoke additionally checks real cached tensors."""
from pathlib import Path
import sys
import unittest
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'experiments/mask_readout_mechanism_v2_20261012/scripts'))
from adapter_model import vector_grouped_loss
from soft_iou_loss import soft_iou_grouped_loss, EPSILON
torch.set_num_threads(1)


class LossTests(unittest.TestCase):
    def test_formula_instance_grouping_and_preservation(self):
        torch.manual_seed(42)
        z = torch.randn(3, 3, 4, requires_grad=True)
        base = torch.randn_like(z)
        roi = torch.zeros_like(z, dtype=torch.bool)
        roi[0, :2, :3] = True
        roi[1, :, :2] = True
        targets = {0: (torch.rand(3, 4) > .5).float(), 1: torch.zeros(3, 4), 2: torch.ones(3, 4)}
        result, info = soft_iou_grouped_loss(z, base, roi, targets)
        _, previous = vector_grouped_loss(z, base, roi, targets)
        for key in ('preservation_loss', 'matched_supervised_instances', 'preserved_instances', 'empty_roi_rows', 'no_output_image'):
            self.assertEqual(info[key], previous[key])
        manual = []
        for i in (0, 1):
            p, g = z[i][roi[i]].sigmoid(), targets[i][roi[i]]
            manual.append(1-(sum(p*g)+EPSILON)/(sum(p+g-p*g)+EPSILON))
        self.assertAlmostEqual(info['gt_loss'], float(torch.stack(manual).mean().detach()), places=6)
        self.assertEqual(info['empty_GT_in_ROI_instances'], 1)
        result.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(torch.count_nonzero(z.grad[~roi]).item(), 0)

    def test_preservation_exact_value_and_gradient(self):
        torch.manual_seed(7)
        z = torch.randn(4, 5, 6, requires_grad=True)
        baseline = torch.randn_like(z, requires_grad=True)
        roi = torch.rand_like(z) > .4
        a, _ = vector_grouped_loss(z, baseline, roi, {})
        b, _ = soft_iou_grouped_loss(z, baseline, roi, {})
        self.assertTrue(torch.equal(a, b))
        ga = torch.autograd.grad(a, z, retain_graph=True)[0]
        gb = torch.autograd.grad(b, z)[0]
        self.assertTrue(torch.equal(ga, gb))
        self.assertIsNone(baseline.grad)

    def test_GT_gradient_foreground_background_and_empty_GT(self):
        z = torch.zeros(1, 2, 3, requires_grad=True)
        roi = torch.tensor([[[True, True, False], [True, True, False]]])
        g = torch.tensor([[1., 0., 1.], [1., 0., 1.]])
        total, _ = soft_iou_grouped_loss(z, z.detach(), roi, {0: g})
        total.backward()
        self.assertTrue((z.grad[0][roi[0] & (g == 1)] < 0).all())
        self.assertTrue((z.grad[0][roi[0] & (g == 0)] > 0).all())
        self.assertEqual(torch.count_nonzero(z.grad[~roi]).item(), 0)
        empty = torch.zeros_like(z, requires_grad=True)
        loss, info = soft_iou_grouped_loss(empty, empty.detach(), roi, {0: torch.zeros_like(g)})
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue((empty.grad[roi] > 0).all())
        self.assertEqual(info['matched_supervised_instances'], 1)

    def test_matching_mask_beats_inverse_and_empty_ROI(self):
        g = torch.tensor([[1., 0.], [1., 0.]])
        roi = torch.ones(1, 2, 2, dtype=torch.bool)
        z = ((g*2-1)*12)[None].requires_grad_()
        _, good = soft_iou_grouped_loss(z, z.detach(), roi, {0: g})
        _, bad = soft_iou_grouped_loss(-z, z.detach(), roi, {0: g})
        self.assertLess(good['gt_loss'], 1e-4)
        self.assertGreater(bad['gt_loss'], .99)
        empty, info = soft_iou_grouped_loss(z, z.detach(), ~roi, {0: g})
        self.assertEqual(float(empty.detach()), 0)
        self.assertEqual(info['matched_supervised_instances'], 0)
        self.assertEqual(info['empty_roi_rows'], [0])
        empty.backward()
        self.assertEqual(torch.count_nonzero(z.grad).item(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
