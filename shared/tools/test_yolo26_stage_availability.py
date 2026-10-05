"""Small checks for the two failure-prone stage-audit invariants."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_yolo26_stage_availability import box_mask_coverage, first_loss


class StageAvailabilityTest(unittest.TestCase):
    def test_crop_envelope_is_a_coverage_upper_bound(self):
        gt = np.ones((7, 9), dtype=bool)
        boxes = np.array([[-1.2, 0.3, 5.4, 6.1], [1.2, 3.0, 8.8, 3.0], [2.8, 1.1, 8.2, 6.8]])
        y, x = np.indices(gt.shape)
        for box, bound in zip(boxes, box_mask_coverage(boxes, gt)):
            crop = (x >= box[0]) & (x < box[2]) & (y >= box[1]) & (y < box[3])
            self.assertLessEqual(crop.sum() / gt.sum(), bound)

    def test_nested_stage_loss_and_invalid_counts(self):
        self.assertEqual(first_loss(dict(raw=9, topk=2, conf=0, final=0)), "conf")
        self.assertEqual(first_loss(dict(raw=9, topk=2, conf=1, final=1)), "survives_final")
        with self.assertRaises(ValueError):
            first_loss(dict(raw=0, topk=1, conf=0, final=0))


if __name__ == "__main__":
    unittest.main()
