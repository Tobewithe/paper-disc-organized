import unittest
import torch
from proto_statistics import tensor_counts, counted_iou, direction, summarize, adjudicate


class ProtoBehaviorTests(unittest.TestCase):
    def test_same_roi_strict_zero_and_integer_counts(self):
        logits = torch.tensor([[[0., 2., -2.], [5., -3., 5.]]])
        gt = torch.tensor([[[1., 1., 0.], [0., 0., 1.]]])
        roi = torch.tensor([[[True, True, True], [True, True, False]]])
        row = tensor_counts(logits, gt, roi)[0]
        self.assertEqual((row['tp'], row['fp'], row['fn'], row['tn']), (1, 1, 1, 2))
        self.assertEqual(row['iou'], 1/3)

    def test_empty_union_retained_but_empty_roi_rejected(self):
        self.assertEqual(counted_iou(0, 0, 0, 5)['iou'], 1.)
        self.assertTrue(counted_iou(0, 0, 0, 5)['empty_union'])
        self.assertEqual(counted_iou(0, 2, 0, 3)['iou'], 0.)
        with self.assertRaises(ValueError):
            counted_iou(0, 0, 0, 0)

    def test_direction_boundaries(self):
        self.assertEqual([direction(v) for v in (-2e-12, -1e-12, 0., 1e-12, 2e-12)],
                         ['down', 'tie', 'tie', 'tie', 'up'])
        self.assertEqual(direction(-1e-6, 1e-6), 'tie')

    def test_complete_counts_and_strict_majority(self):
        base = {'direction': 'loss_down_iou_down', 'zero_proto': counted_iou(1, 1, 0, 1),
                'B_proto': counted_iou(1, 0, 0, 2), 'delta_proto_iou': .5}
        rows = [{**base, 'proto_direction': d} for d in ('down', 'down', 'tie', 'up')]
        result = summarize(rows)
        self.assertEqual(result['discrepant_proto_directions'], {'down': 2, 'tie': 1, 'up': 1})
        self.assertFalse(adjudicate(result)['majority_early_reversal'])
        self.assertTrue(adjudicate(summarize(rows[:3]))['majority_early_reversal'])

    def test_invalid_grid_rejected(self):
        with self.assertRaises(ValueError):
            tensor_counts(torch.ones(1, 1, 1), torch.full((1, 1, 1), .5), torch.ones(1, 1, 1, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
