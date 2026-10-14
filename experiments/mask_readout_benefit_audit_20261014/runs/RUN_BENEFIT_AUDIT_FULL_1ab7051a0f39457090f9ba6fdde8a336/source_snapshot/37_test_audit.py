"""Behavioral checks for identity-independent diagnostic statistics."""
import unittest
import numpy as np
from audit_statistics import counted, weighted_auc, failure_stage, STAGES, FEATURES, decide


class AuditTests(unittest.TestCase):
    def test_counts_and_empty(self):
        self.assertEqual(counted(2, 1, 1, 4)['iou'], .5)
        self.assertEqual(counted(0, 0, 0, 4)['iou'], 1)
        with self.assertRaises(ValueError): counted(-1, 1, 0, 1)

    def test_auc_direction_ties_and_replication(self):
        self.assertEqual(weighted_auc([0, 1], [0, 1]), 1)
        self.assertEqual(weighted_auc([1, 0], [0, 1]), 0)
        self.assertEqual(weighted_auc([1, 1], [0, 1]), .5)
        scores, labels, weights = np.array([1, 2, 2, 3]), np.array([0, 1, 0, 1]), np.array([2, 3, 1, 2])
        repeated = np.repeat(np.arange(4), weights)
        self.assertEqual(weighted_auc(scores, labels, weights), weighted_auc(scores[repeated], labels[repeated]))
        self.assertIsNone(weighted_auc([1, 2], [1, 1]))

    def test_stage_uses_fixed_success_mask_before_matching(self):
        stages = {k: {'delta': .01} for k in STAGES}
        stages['binary_original']['after'] = {'iou': .74}
        stages['binary_original']['delta'] = -.02
        row = {'task_damaged': True, 'stages': stages}
        self.assertEqual(failure_stage(row), 'input_to_original_reversal')
        stages['binary_roi']['delta'] = -.01
        self.assertEqual(failure_stage(row), 'proto_binary_reversal')
        stages['binary_original']['after']['iou'] = .76
        self.assertEqual(failure_stage(row), 'matching_despite_fixed_mask75')

    def test_priority_uses_frozen_order_not_highest_auc(self):
        summary = {'signals': {'features': {f: {'priority_gate': True} for f in FEATURES}}}
        self.assertEqual(decide(summary)['selected_signal'], FEATURES[0])


if __name__ == '__main__':
    unittest.main()
