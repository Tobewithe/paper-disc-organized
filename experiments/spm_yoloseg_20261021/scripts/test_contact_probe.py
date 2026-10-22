"""Small scientific correctness checks, independent of real-data outcomes."""
import unittest
import numpy as np
from contact_probe import average_precision, boundaries


class ContactCorrectness(unittest.TestCase):
    def test_same_class_seam_is_distinct_from_outer_boundary(self):
        ids = np.zeros((640, 640), np.int32)
        ids[100:540, 100:320] = 1
        ids[100:540, 320:540] = 2
        lab, _ = boundaries(ids, np.zeros_like(ids, bool))
        self.assertEqual(lab[320, 320], 1)
        self.assertEqual(lab[320, 100], 0)
        self.assertEqual(lab[320, 200], 2)
        ids[ids == 2] = 1
        lab, _ = boundaries(ids, np.zeros_like(ids, bool))
        self.assertNotEqual(lab[320, 320], 1)

    def test_conflicting_ground_truth_is_ignored(self):
        ids = np.zeros((640, 640), np.int32)
        ids[100:540, 100:320] = 1
        ids[100:540, 320:540] = 2
        bad = np.zeros_like(ids, bool); bad[320, 320] = True
        lab, _ = boundaries(ids, bad)
        self.assertEqual(lab[320, 320], -1)

    def test_AP_handles_ties_without_label_order_leakage(self):
        self.assertAlmostEqual(average_precision([1, 0, 1], [.9, .8, .7]), 5/6)
        self.assertAlmostEqual(average_precision([1, 0, 1], [0, 0, 0]), 2/3)
        self.assertAlmostEqual(average_precision([0, 1, 1], [0, 0, 0]), 2/3)


if __name__ == "__main__": unittest.main()
