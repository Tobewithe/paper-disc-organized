import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_yolo26_strict_quality import assert_count_invariants, final_source_mapping, quality_flags, source_stage


class StrictQualityTests(unittest.TestCase):
    def test_quality_edges_and_nested_source_stages(self) -> None:
        self.assertEqual(quality_flags(0.75, 0.75, 0.75), {"iou50": 1, "coverage75_purity75": 1, "iou75": 1})
        self.assertEqual(quality_flags(0.50, 0.74, 1.00), {"iou50": 1, "coverage75_purity75": 0, "iou75": 0})
        self.assertEqual(source_stage(4, {4}, {4}, {4}), "final")
        self.assertEqual(source_stage(4, {4}, {4}, set()), "conf")
        self.assertEqual(source_stage(4, {4}, set(), set()), "topk")
        self.assertEqual(source_stage(4, set(), set(), set()), "raw")

    def test_final_mapping_rejects_duplicates_and_preserves_prediction_indices(self) -> None:
        record = {
            "final_predictions": [{"pred_id": 8}, {"pred_id": 3}],
            "final_pred_to_source_candidate": [
                {"pred_id": 3, "source_candidate_id": 6},
                {"pred_id": 8, "source_candidate_id": 2},
            ],
        }
        final_by_source, final_index = final_source_mapping(record, "test", 1)
        self.assertEqual(final_by_source[6]["pred_id"], 3)
        self.assertEqual(final_index, {6: 1, 2: 0})

        with self.assertRaisesRegex(ValueError, "Duplicate final prediction ID"):
            final_source_mapping({**record, "final_predictions": [{"pred_id": 8}, {"pred_id": 8}]}, "test", 1)
        with self.assertRaisesRegex(ValueError, "Duplicate final mapping prediction ID"):
            final_source_mapping({**record, "final_pred_to_source_candidate": [{"pred_id": 8, "source_candidate_id": 2}, {"pred_id": 8, "source_candidate_id": 6}]}, "test", 1)
        with self.assertRaisesRegex(ValueError, "Duplicate final mapping source ID"):
            final_source_mapping({**record, "final_pred_to_source_candidate": [{"pred_id": 8, "source_candidate_id": 2}, {"pred_id": 3, "source_candidate_id": 2}]}, "test", 1)
        with self.assertRaisesRegex(ValueError, "differ from final predictions"):
            final_source_mapping({**record, "final_pred_to_source_candidate": [{"pred_id": 8, "source_candidate_id": 2}, {"pred_id": 7, "source_candidate_id": 6}]}, "test", 1)

    def test_count_invariants(self) -> None:
        counts = [{
            "iou50": {"raw": 4, "topk": 3, "conf": 2, "final": 1},
            "coverage75_purity75": {"raw": 2, "topk": 2, "conf": 1, "final": 1},
            "iou75": {"raw": 1, "topk": 1, "conf": 1, "final": 0},
        }]
        assert_count_invariants(counts, "test", 1)
        counts[0]["iou50"]["final"] = 3
        with self.assertRaisesRegex(ValueError, "Stage nesting failed"):
            assert_count_invariants(counts, "test", 1)
        counts[0]["iou50"]["final"] = 1
        counts[0]["coverage75_purity75"] = {"raw": 2, "topk": 2, "conf": 2, "final": 2}
        with self.assertRaisesRegex(ValueError, "Coverage/purity subset failed"):
            assert_count_invariants(counts, "test", 1)


if __name__ == "__main__":
    unittest.main()
