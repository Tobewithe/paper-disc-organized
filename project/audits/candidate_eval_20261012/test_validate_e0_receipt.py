"""Synthetic fixture tests for the read-only E0 receipt gate."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_e0_receipt as audit


class E0ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.old_expected = audit.EXPECTED
        audit.EXPECTED = {"fit": (2, 3), "dev": (1, 2), "val": (1, 1)}
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.eval = self.root / "eval"
        self.eval.mkdir()
        self.ref = self.root / "official.jsonl"
        self.result = self.root / "E0_GATE.json"
        self.rows = [
            self.row("fit", 1, 101, 0, 0.4),
            self.row("fit", 1, 102, 1, 0.5),
            self.row("fit", 2, 103, 0, 0.6),
            self.row("dev", 3, 104, 0, 0.7),
            self.row("dev", 3, 105, 1, 0.8),
            self.row("val", 4, 106, 0, 0.9),
        ]
        self.write()

    def tearDown(self):
        audit.EXPECTED = self.old_expected
        self.tmp.cleanup()

    @staticmethod
    def row(split, image_id, annotation_id, raw_id, baseline):
        return {"split": split, "image_id": image_id, "annotation_id": annotation_id,
                "branch": "one2one", "raw_id": raw_id, "pyramid_level": 0,
                "iou_A": baseline, "iou_B": baseline + 0.03, "iou_C": baseline - 0.02}

    def write(self, gradient=1e-7, mutate_ref=None, omit_ref=False):
        run = {"status": "complete", "map_validation": {
            "old160": {"converged": True, "gradient_inf_max": gradient},
            "official640": {"converged": True, "gradient_inf_max": gradient}}}
        (self.eval / "run.json").write_text(json.dumps(run))
        blocks = {}
        for split in audit.EXPECTED:
            subset = [x for x in self.rows if x["split"] == split]
            imgs = defaultdict(list)
            for row in subset:
                imgs[row["image_id"]].append(row)
            macro = {arm: sum(sum(x[f"iou_{arm}"] for x in grp) / len(grp)
                             for grp in imgs.values()) / len(imgs) for arm in "ABC"}
            for a, b, name in (("B", "A", "B_minus_A"), ("C", "A", "C_minus_A"),
                               ("C", "B", "C_minus_B")):
                macro[name] = {"delta": macro[a] - macro[b], "ci95": [0, 0]}
            blocks[split] = {"iou": {"n_images": len(imgs),
                                      "n_candidates": len(subset), "image_macro": macro}}
        (self.eval / "SUMMARY.json").write_text(json.dumps({"tables": blocks}))
        (self.eval / "PER_CANDIDATE.jsonl").write_text(
            "\n".join(json.dumps(x) for x in self.rows) + "\n")
        if omit_ref and self.ref.exists():
            self.ref.unlink()
        if not omit_ref:
            reference = [{**r, "iou_B": 0.0, "iou_C": 1.0} for r in self.rows]
            if mutate_ref:
                mutate_ref(reference)
            self.ref.write_text("\n".join(json.dumps(x) for x in reference) + "\n")

    def call(self):
        status = audit.main(["--eval-dir", str(self.eval),
                             "--official-per-candidate", str(self.ref),
                             "--out", str(self.result)])
        return status, json.loads(self.result.read_text())

    def test_accepts_same_native_when_historical_B_C_differ(self):
        status, result = self.call()
        self.assertEqual(status, 0)
        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["G2_official_native_A_parity"]["detail"]["native_only_comparison"])

    def test_blocks_baseline_mismatch(self):
        self.write(mutate_ref=lambda r: r[0].update(iou_A=0.9))
        status, result = self.call()
        self.assertEqual(status, 2)
        self.assertFalse(result["checks"]["G2_official_native_A_parity"]["passed"])

    def test_blocks_incomplete_reference(self):
        self.write(mutate_ref=lambda r: r.pop())
        self.assertEqual(self.call()[0], 2)

    def test_blocks_missing_reference(self):
        self.write(omit_ref=True)
        self.assertEqual(self.call()[0], 2)

    def test_blocks_nonconverged(self):
        self.write(gradient=1e-3)
        self.assertEqual(self.call()[0], 2)


if __name__ == "__main__":
    unittest.main()
