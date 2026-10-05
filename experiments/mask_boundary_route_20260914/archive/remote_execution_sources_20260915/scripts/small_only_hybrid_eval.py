"""Evaluate a prediction-only small-mask gate for adaptive boundary calibration."""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, "D:/coco_wire/py")

from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT = Path("D:/coco_wire")
OUT = ROOT / "results/mask_boundary_route_20260914"
ANN = ROOT / "data/annotations/instances_val2017.json"
PREDICTED_MASK_AREA_GATE = 32**2


def key(item):
    return (int(item["image_id"]), int(item["category_id"]), round(float(item["score"]), 12))


def evaluate(coco, predictions, image_ids):
    result = coco.loadRes(predictions)
    evaluator = COCOeval(coco, result, "segm")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return [float(x) for x in evaluator.stats]


def main():
    baseline = json.loads((OUT / "predictions_baseline.json").read_text(encoding="utf-8"))
    adaptive = json.loads((OUT / "predictions_adaptive.json").read_text(encoding="utf-8"))
    adaptive_by_key = defaultdict(deque)
    for item in adaptive:
        adaptive_by_key[key(item)].append(item)
    hybrid = []
    applied = 0
    missing_adaptive = 0
    for base in baseline:
        chosen = base
        bucket = adaptive_by_key.get(key(base))
        adaptive_item = bucket.popleft() if bucket else None
        base_area = float(mask_utils.area(base["segmentation"]))
        if base_area < PREDICTED_MASK_AREA_GATE:
            if adaptive_item is not None:
                chosen = adaptive_item
                applied += 1
            else:
                missing_adaptive += 1
        hybrid.append(chosen)

    coco = COCO(str(ANN))
    image_ids = sorted(coco.getImgIds())
    split = len(image_ids) // 2
    # Use the exact policy split from matched records: images without a match
    # are excluded from both halves in the original policy fit/evaluation.
    import csv

    with (OUT / "matched_records.csv").open("r", encoding="utf-8", newline="") as f:
        matched_ids = sorted({int(r["image_id"]) for r in csv.DictReader(f)})
    heldout_ids = matched_ids[len(matched_ids) // 2 :]

    payload = {
        "gate": "apply adaptive mask only when baseline predicted mask area < 32^2 pixels",
        "gate_uses_gt": False,
        "baseline_predictions": len(baseline),
        "hybrid_predictions": len(hybrid),
        "adaptive_applied": applied,
        "adaptive_missing_fallback_baseline": missing_adaptive,
        "full": evaluate(coco, hybrid, image_ids),
        "heldout": evaluate(coco, hybrid, heldout_ids),
    }
    with (OUT / "SMALL_ONLY_HYBRID_EVAL.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with (OUT / "predictions_small_only_hybrid.json").open("w", encoding="utf-8") as f:
        json.dump(hybrid, f, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
