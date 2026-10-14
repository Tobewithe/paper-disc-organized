"""The native COCO result must not silently fall back to internal SegMetrics."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared/evaluation"))
from native_output_bridge import make_trace_validator, coco_metric_receipt


class NativeReceiptTests(unittest.TestCase):
    def fixture(self, folder):
        import numpy as np
        from pycocotools import mask
        r = mask.encode(np.asfortranarray(np.ones((10, 10), dtype=np.uint8)))
        r["counts"] = r["counts"].decode()
        gt = {"info": {}, "images": [{"id": 1, "height": 10, "width": 10}],
              "categories": [{"id": 1, "name": "object"}], "annotations": [
              {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10],
               "area": 100, "iscrowd": 0, "segmentation": r}]}
        # Box AP=0, mask AP=100: catches task cross-contamination in result receipts.
        preds = [{"image_id": 1, "category_id": 1, "score": .9,
                  "bbox": [20, 20, 10, 10], "segmentation": r}]
        (folder / "annotations").mkdir()
        g, p = folder / "annotations/instances_val2017.json", folder / "predictions.json"
        g.write_text(json.dumps(gt), encoding="utf8")
        p.write_text(json.dumps(preds), encoding="utf8")
        cls = make_trace_validator(folder, [1], "smoke", 1000, "unused")
        v = object.__new__(cls)
        v.args, v.is_coco, v.is_lvis, v.jdict = SimpleNamespace(save_json=True), True, False, preds
        v.dataloader = SimpleNamespace(dataset=SimpleNamespace(im_files=["1.jpg"]))
        v.save_dir, v.data = folder, {"path": folder}
        return v, g, p

    def test_capture_exact_native_objects_and_all_12_metrics(self):
        import faster_coco_eval
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        original = faster_coco_eval.COCOeval_faster
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()):
            v, g, p = self.fixture(Path(t))
            v.eval_json({})
            self.assertIs(faster_coco_eval.COCOeval_faster, original)
            self.assertAlmostEqual(v.native_coco_receipts["bbox"]["metrics"]["AP"], 0.)
            self.assertAlmostEqual(v.native_coco_receipts["segm"]["metrics"]["AP"], 100.)
            for task in ("bbox", "segm"):
                gt = COCO(str(g))
                e = COCOeval(gt, gt.loadRes(str(p)), task)
                e.evaluate(); e.accumulate(); e.summarize()
                expected = coco_metric_receipt(e)["metrics"]
                for name, value in expected.items():
                    if value is None:
                        self.assertIsNone(v.native_coco_receipts[task]["metrics"][name])
                    else:
                        self.assertAlmostEqual(value, v.native_coco_receipts[task]["metrics"][name])

    def test_swallowed_native_error_is_a_hard_failure(self):
        class BrokenEvaluator:
            def __init__(self, *a, **kw):
                raise RuntimeError("deliberate backend failure")
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()):
            v, g, p = self.fixture(Path(t))
            with patch("faster_coco_eval.COCOeval_faster", BrokenEvaluator):
                with self.assertRaisesRegex(ValueError, "fallback forbidden"):
                    v.eval_json({})


if __name__ == "__main__":
    unittest.main()
