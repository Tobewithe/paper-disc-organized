"""Unit tests for the prospective cross-audit decision gate."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decide_cross


EXPECTED = decide_cross.EXPECTED


def _contrast(delta: float, low: float, high: float) -> dict[str, object]:
    return {"delta": delta, "ci95": [low, high], "scope": "image_cluster_image_macro"}


def _summary(*, winner: str = "B", qualifies: bool = True, tie: bool = False) -> dict[str, object]:
    tables: dict[str, object] = {}
    for split, counts in EXPECTED.items():
        if split == "dev":
            b_delta, c_delta = (0.02, 0.02) if tie else ((0.02, 0.01) if winner == "B" else (0.01, 0.02))
            b_ci = [0.005, 0.03] if qualifies and winner == "B" else [-0.01, 0.03]
            c_ci = [0.005, 0.03] if qualifies and winner == "C" else [-0.01, 0.03]
        elif split == "val":
            b_delta, c_delta = ((0.018, 0.008) if winner == "B" else (0.008, 0.018))
            b_ci = [0.004, 0.03] if qualifies and winner == "B" else [-0.01, 0.03]
            c_ci = [0.004, 0.03] if qualifies and winner == "C" else [-0.01, 0.03]
        else:
            b_delta, c_delta, b_ci, c_ci = 0.01, 0.01, [-0.01, 0.03], [-0.01, 0.03]
        a = 0.75
        image_macro = {
            "A": a,
            "B": a + b_delta,
            "C": a + c_delta,
            "B_minus_A": _contrast(b_delta, b_ci[0], b_ci[1]),
            "C_minus_A": _contrast(c_delta, c_ci[0], c_ci[1]),
            "C_minus_B": _contrast(c_delta - b_delta, -0.01, 0.01),
        }
        candidate = dict(image_macro)
        tables[split] = {"iou": {"n_candidates": counts["n_candidates"], "n_images": counts["n_images"], "image_macro": image_macro, "candidate": candidate}}
    return {"tables": tables}


def _write_candidates(path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for split, counts in EXPECTED.items():
            for i in range(counts["n_candidates"]):
                image_id = 100000 * (list(EXPECTED).index(split) + 1) + (i % counts["n_images"])
                row = {"split": split, "image_id": image_id, "annotation_id": i, "raw_id": i, "pyramid_level": i % 3, "iou_A": 0.7, "iou_B": 0.71, "iou_C": 0.72}
                f.write(json.dumps(row) + "\n")


class DecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(tempfile.mkdtemp(prefix="cross-decision-"))
        base = cls.root / "base"
        base.mkdir()
        (base / "run.json").write_text(json.dumps({"status": "evaluating", "map_validation": {"old160": {"converged": True, "gradient_inf_max": 8e-7}, "official640": {"converged": True, "gradient_inf_max": 9e-7}}}), encoding="utf-8")
        (base / "SUMMARY.json").write_text(json.dumps(_summary()), encoding="utf-8")
        _write_candidates(base / "PER_CANDIDATE.jsonl")
        cls.base = base

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def _case(self, name: str) -> Path:
        dst = self.root / name
        shutil.copytree(self.base, dst)
        return dst

    def test_positive_selects_b_and_starts_aligned_training(self) -> None:
        d = decide_cross.decide(self._case("positive"))
        self.assertEqual(d["status"], "proceed_aligned_training")
        self.assertEqual(d["winner"], "B")

    def test_negative_stops_affine_expansion(self) -> None:
        d = self._case("negative")
        (d / "SUMMARY.json").write_text(json.dumps(_summary(winner="C", qualifies=False)), encoding="utf-8")
        out = decide_cross.decide(d)
        self.assertEqual(out["status"], "stop_affine_method_expansion")
        self.assertEqual(out["next_step"], "synthesize_full_fit_to_heldout_gap_in_mechanism_report")

    def test_exact_tie_selects_b(self) -> None:
        d = decide_cross.decide(self._case("tie"))
        self.assertEqual(d["winner"], "B")

    def test_count_invalid_is_insufficient(self) -> None:
        d = self._case("count-invalid")
        summary = json.loads((d / "SUMMARY.json").read_text(encoding="utf-8"))
        summary["tables"]["fit"]["iou"]["n_candidates"] = 1
        (d / "SUMMARY.json").write_text(json.dumps(summary), encoding="utf-8")
        out = decide_cross.decide(d)
        self.assertEqual(out["status"], "insufficient_inputs")

    def test_missing_gradient_is_insufficient(self) -> None:
        d = self._case("missing-gradient")
        run = json.loads((d / "run.json").read_text(encoding="utf-8"))
        run.pop("map_validation")
        (d / "run.json").write_text(json.dumps(run), encoding="utf-8")
        out = decide_cross.decide(d)
        self.assertEqual(out["status"], "insufficient_inputs")

    def test_incomplete_outputs_are_insufficient(self) -> None:
        d = self._case("incomplete")
        (d / "PER_CANDIDATE.jsonl").unlink()
        out = decide_cross.decide(d)
        self.assertEqual(out["status"], "insufficient_inputs")

    def test_nan_contrast_is_insufficient(self) -> None:
        d = self._case("nan")
        summary = json.loads((d / "SUMMARY.json").read_text(encoding="utf-8"))
        summary["tables"]["dev"]["iou"]["image_macro"]["B_minus_A"]["delta"] = float("nan")
        (d / "SUMMARY.json").write_text(json.dumps(summary, allow_nan=True), encoding="utf-8")
        out = decide_cross.decide(d)
        self.assertEqual(out["status"], "insufficient_inputs")


if __name__ == "__main__":
    unittest.main()
