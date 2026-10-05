"""CPU checks for R005c's frozen stratum and assignment contract."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_yolo26_confidence_diagnostic.py")
SPEC = importlib.util.spec_from_file_location("r005c", SCRIPT)
assert SPEC and SPEC.loader
r005c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r005c)


class ConfidenceDiagnosticTest(unittest.TestCase):
    def test_stratum_boundary_and_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            per_gt, edges = root / "per_gt.csv", root / "edge_metrics.csv"
            fields = ["dataset", "image_id", "annotation_id", "baseline_class", "coverage75_purity75_topk_available", "coverage75_purity75_conf_available", "coverage75_purity75_final_available"]
            with per_gt.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                writer.writerows([
                    dict(zip(fields, ["PigLife_public_test", 1, 10, "O", 1, 0, 0])),
                    dict(zip(fields, ["PigLife_public_test", 1, 11, "M", 1, 0, 0])),
                    dict(zip(fields, ["PigLife_public_test", 1, 12, "C", 1, 0, 0])),
                    dict(zip(fields, ["FaroPigSeg_test", 2, 20, "MISS", 1, 1, 1])),
                ])
            edge_fields = ["dataset", "image_id", "annotation_id", "source_candidate_id", "source_score", "global_rank", "topk_member", "conf_member", "final_member", "iou", "coverage", "purity", "quality_coverage75_purity75"]
            base = {"dataset": "PigLife_public_test", "image_id": 1, "topk_member": 1, "conf_member": 0, "final_member": 0, "iou": .5, "coverage": .75, "purity": .75, "quality_coverage75_purity75": 1}
            with edges.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=edge_fields); writer.writeheader()
                writer.writerows([
                    {**base, "annotation_id": 10, "source_candidate_id": 3, "source_score": .05, "global_rank": 4},
                    {**base, "annotation_id": 11, "source_candidate_id": 3, "source_score": .02, "global_rank": 5},
                    {**base, "annotation_id": 10, "source_candidate_id": 4, "source_score": .01, "global_rank": 6},
                ])
            targets = r005c.target_rows(per_gt, edges, {("PigLife_public_test", 1): (480, 640)})
            self.assertEqual([(row["annotation_id"], row["eligible_edge_count"]) for row in targets], [(10, 1), (11, 1)])
            self.assertTrue(all((row["image_height"], row["image_width"]) == (480, 640) for row in targets))
            chosen = r005c.choose_oracle(targets, r005c.load_tool("run_yolo26_candidate_perturbation.py").maximum_cardinality_assignment)
            self.assertEqual(sum(row["selected_source_candidate_id"] is not None for row in chosen), 1)
            self.assertEqual({row["selected_source_candidate_id"] for row in chosen if row["selected_source_candidate_id"] is not None}, {3})
            class Fixed:
                @staticmethod
                def read_json(_path):
                    return {"images": [{"id": 1, "height": 480, "width": 640}]}
            r005c.attach_manifest_geometry(Fixed(), ({"name": "PigLife_public_test", "manifest": root / "manifest.json"},), chosen)
            self.assertTrue(all((row["image_height"], row["image_width"]) == (480, 640) for row in chosen))

    def test_prediction_export_mapping_is_one_to_one(self) -> None:
        record = {
            "image_id": 7,
            "final_predictions": [{"pred_id": 1}, {"pred_id": 2}],
            "final_pred_to_source_candidate": [
                {"pred_id": 1, "source_candidate_id": 20},
                {"pred_id": 2, "source_candidate_id": 21},
            ],
        }
        self.assertEqual(r005c.prediction_pairs(record), [(1, 20), (2, 21)])
        record["final_pred_to_source_candidate"][1]["source_candidate_id"] = 20
        with self.assertRaisesRegex(RuntimeError, "Invalid prediction mapping"):
            r005c.prediction_pairs(record)

    def test_oracle_guard_rejects_mapping_mutation(self) -> None:
        choices = [{"dataset": "PigLife_public_test", "image_id": 1, "annotation_id": 2, "selected_source_candidate_id": 3}]
        removable = {("PigLife_public_test", 1, 2): {7}}
        frozen = r005c.frozen_oracle_tuple(choices, removable)
        choices[0]["selected_source_candidate_id"] = 4
        with self.assertRaisesRegex(RuntimeError, "mapping drift"):
            r005c.assert_frozen_oracle(frozen, choices, removable, "test")


if __name__ == "__main__":
    unittest.main()
