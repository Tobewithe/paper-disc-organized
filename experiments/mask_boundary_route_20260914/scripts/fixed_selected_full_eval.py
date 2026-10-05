"""Final frozen evaluation of the dev-selected scale-aware mask threshold."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "D:/coco_wire/scripts")
sys.path.insert(0, "D:/coco_wire/py")

from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

import boundary_threshold_probe as probe


ROOT = Path("D:/coco_wire")
END2END = os.environ.get("PROBE_END2END", "true").strip().lower() not in {"0", "false", "no"}
OUT = ROOT / (
    "results/mask_boundary_route_20260914/fixed_selected_full"
    if END2END
    else "results/mask_boundary_route_20260914/fixed_selected_full_one2many"
)
OUT.mkdir(parents=True, exist_ok=True)
AREA_GATE = 48**2
LOGIT_THRESHOLD = 0.75


def evaluate(coco, predictions, image_ids):
    result = coco.loadRes(copy.deepcopy(predictions))
    evaluator = COCOeval(coco, result, "segm")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return [float(x) for x in evaluator.stats]


def encode(binary):
    encoded = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    encoded["size"] = [int(x) for x in encoded["size"]]
    return encoded


def main():
    coco = COCO(str(probe.ANN_FILE))
    model = YOLO(str(probe.WEIGHTS))
    model.model.model[-1].end2end = END2END
    cat_by_name = {c["name"]: int(c["id"]) for c in coco.loadCats(coco.getCatIds())}
    captures = probe.collect_prediction_captures(model)
    baseline_predictions = []
    method_predictions = []
    applied = 0
    empty_after = 0
    for image_name, capture in sorted(captures.items()):
        image_id = int(Path(image_name).stem)
        logits, _ = probe.decode_logits(
            capture["proto"], capture["pred"], capture["img_shape"], capture["orig_shape"]
        )
        pred = capture["pred"].numpy()
        for i in range(pred.shape[0]):
            category_id = cat_by_name.get(str(model.names[int(pred[i, 5])]))
            if category_id is None:
                continue
            baseline = logits[i].gt(0.0).numpy()
            if baseline.any():
                baseline_predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": encode(baseline),
                        "score": float(pred[i, 4]),
                    }
                )
            threshold = LOGIT_THRESHOLD if int(baseline.sum()) < AREA_GATE else 0.0
            if threshold > 0:
                applied += 1
            binary = logits[i].gt(threshold).numpy()
            if not binary.any():
                empty_after += 1
                continue
            method_predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": encode(binary),
                    "score": float(pred[i, 4]),
                }
            )

    image_ids = sorted(coco.getImgIds())
    heldout_ids = image_ids[500:]
    baseline_full = evaluate(coco, baseline_predictions, image_ids)
    method_full = evaluate(coco, method_predictions, image_ids)
    baseline_heldout = evaluate(coco, baseline_predictions, heldout_ids)
    method_heldout = evaluate(coco, method_predictions, heldout_ids)
    payload = {
        "method": "if baseline predicted mask area < 48^2 pixels, threshold raw mask logit at 0.75; otherwise use official threshold 0",
        "selection": "area gate and threshold selected once on the first 500 sorted COCO val images",
        "uses_gt_at_inference": False,
        "end2end": END2END,
        "inference_branch": "one-to-one NMS-free" if END2END else "one-to-many plus NMS",
        "ultralytics": __import__("ultralytics").__version__,
        "images": len(captures),
        "baseline_predictions": len(baseline_predictions),
        "method_predictions": len(method_predictions),
        "applied": applied,
        "empty_after": empty_after,
        "full": {
            "baseline": baseline_full,
            "method": method_full,
            "delta": [m - b for b, m in zip(baseline_full, method_full)],
        },
        "heldout": {
            "images": len(heldout_ids),
            "baseline": baseline_heldout,
            "method": method_heldout,
            "delta": [m - b for b, m in zip(baseline_heldout, method_heldout)],
        },
    }
    with (OUT / "predictions_baseline.json").open("w", encoding="utf-8") as f:
        json.dump(baseline_predictions, f, ensure_ascii=False)
    with (OUT / "predictions_method.json").open("w", encoding="utf-8") as f:
        json.dump(method_predictions, f, ensure_ascii=False)
    with (OUT / "SUMMARY.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
