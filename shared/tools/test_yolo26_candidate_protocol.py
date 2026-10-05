"""Minimal CPU checks for oracle matching and final-source parity helpers."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PIGCV_SCRIPTS = Path(r"C:\Dpan\codexproject\pigcv_research\scripts")
sys.path.insert(0, str(PIGCV_SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


oracle = load_module("candidate_perturbation", ROOT / "run_yolo26_candidate_perturbation.py")
legacy = load_module("legacy_stage_diagnostic", PIGCV_SCRIPTS / "analyze_yolo26_diagnostic_cache_full.py")


class Fixed:
    @staticmethod
    def rle(value):
        return value


class CandidateProtocolTests(unittest.TestCase):
    def test_oracle_assignment_is_maximum_cardinality(self) -> None:
        def choice(source, iou):
            return {"source_candidate_id": source, "raw_mask_iou": iou, "raw_gt_coverage": iou, "score": iou, "global_rank": source + 1}

        assignment = oracle.maximum_cardinality_assignment({10: [choice(0, 1.0)], 11: [choice(1, 0.9), choice(2, 0.8)], 12: [choice(0, 0.8), choice(1, 0.7)]})
        self.assertEqual({target: selected["source_candidate_id"] for target, selected in assignment.items()}, {10: 0, 11: 2, 12: 1})

    def test_geometry_box_screen_uses_coverage_not_purity(self) -> None:
        gt = np.array([[1, 1], [0, 0]], dtype=bool)
        box_iou, coverage, purity = legacy.rectangle_metrics([gt], np.array([[0.0, 0.0, 2.0, 2.0]]))
        self.assertGreaterEqual(coverage[0, 0], 0.75)
        self.assertLess(purity[0, 0], 0.75)
        self.assertTrue(legacy.box_feasibility_edges("I", box_iou, coverage)[0, 0])

    def test_final_source_parity_reports_xor(self) -> None:
        from pycocotools import mask as mask_utils

        expected = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = mask_utils.encode(np.asfortranarray(expected))
        record = {"image_id": 1, "final_predictions": [{"pred_id": 1, "mask_rle": {"size": [2, 2], "counts": rle["counts"].decode("ascii")}}], "final_pred_to_source_candidate": [{"pred_id": 1, "source_candidate_id": 7}]}
        exact = oracle.final_source_parity(Fixed(), record, {7: expected.astype(bool)})
        self.assertTrue(exact["all_exact"])
        mismatch = oracle.final_source_parity(Fixed(), record, {7: np.zeros((2, 2), dtype=bool)})
        self.assertFalse(mismatch["all_exact"])
        self.assertEqual(mismatch["xor_pixels_total"], 2)


if __name__ == "__main__":
    unittest.main()
