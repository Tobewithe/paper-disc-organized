"""Recompute COCO segmentation metrics on the image-level held-out split."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "D:/coco_wire/py")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT = Path("D:/coco_wire")
OUT = ROOT / "results/mask_boundary_route_20260914"
COCO_FILE = ROOT / "data/annotations/instances_val2017.json"


def evaluate(coco, predictions, image_ids):
    canonical = [
        {
            "image_id": int(p["image_id"]),
            "category_id": int(p["category_id"]),
            "segmentation": p["segmentation"],
            "score": float(p["score"]),
        }
        for p in predictions
    ]
    result = coco.loadRes(canonical)
    evaluator = COCOeval(coco, result, "segm")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return [float(x) for x in evaluator.stats]


def main():
    with (OUT / "matched_records.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    image_ids = sorted({int(r["image_id"]) for r in rows})
    split = max(1, len(image_ids) // 2)
    train_ids = image_ids[:split]
    test_ids = image_ids[split:]
    coco = COCO(str(COCO_FILE))
    metrics = {}
    for label in ("baseline", "adaptive"):
        with (OUT / f"predictions_{label}.json").open("r", encoding="utf-8") as f:
            predictions = json.load(f)
        metrics[label] = evaluate(coco, predictions, test_ids)
    payload = {
        "split_rule": "sorted image IDs with matched records; second half held out from policy selection",
        "train_images": len(train_ids),
        "test_images": len(test_ids),
        "train_first": train_ids[0] if train_ids else None,
        "train_last": train_ids[-1] if train_ids else None,
        "test_first": test_ids[0] if test_ids else None,
        "test_last": test_ids[-1] if test_ids else None,
        "coco_eval_segmentation_test_only": metrics,
    }
    with (OUT / "HELDOUT_COCO_EVAL.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
