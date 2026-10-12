"""Synthetic tests for the read-only GT/TAL category semantics.

Run: python -m unittest discover -s project/audits/candidate_eval_20261012 -p 'test_summarize_gt_tal_cached.py'
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_gt_tal_cached import load_task_outcomes, state, STATES


def row(b, m):
    return {"box_iou": b, "mask_iou": m}


class TaxonomyTests(unittest.TestCase):
    def test_partition_with_explicit_no_assignment(self):
        cases = [
            ([], "unassigned"),
            ([row(.87, .81)], "joint_good"),
            ([row(.90, .42)], "box_only"),
            ([row(.42, .90)], "mask_only"),
            ([row(.20, .60)], "neither"),
            ([row(.82, .61), row(.60, .90)], "separate_good"),
        ]
        labels = [state(rows) for rows, _ in cases]
        self.assertEqual(labels, [name for _, name in cases])
        self.assertEqual(set(labels), set(STATES))

    def test_joint_must_be_same_raw_not_independent_max(self):
        self.assertEqual(state([row(.82, .60), row(.60, .90)]), "separate_good")
        self.assertEqual(state([row(.82, .77), row(.60, .90)]), "joint_good")

    def test_threshold_boundary(self):
        self.assertEqual(state([row(.75, .75)]), "joint_good")
        self.assertEqual(state([row(.749999, .75)]), "mask_only")

    def test_task_outcome_is_external_not_auto_inferred(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "output.jsonl"
            path.write_text(json.dumps({
                "image_id": 1, "annotation_id": 2, "state": "failure"
            }) + "\n", encoding="utf-8")
            self.assertEqual(load_task_outcomes(path), {(1, 2): "failure"})
            self.assertEqual(load_task_outcomes(None), {})

    def test_duplicate_external_task_gt_fails(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "dup.jsonl"
            row1 = json.dumps({"image_id": 1, "annotation_id": 2,
                               "state": "success"})
            path.write_text(row1 + "\n" + row1 + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicated"):
                load_task_outcomes(path)


if __name__ == "__main__":
    unittest.main()
