"""Final evaluation on the 4500 images untouched by 500-image development."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, "D:/coco_wire/py")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT = Path("D:/coco_wire")
OUT = ROOT / "results/mask_boundary_route_20260914"
ANN = ROOT / "data/annotations/instances_val2017.json"


def canonical(item):
    return {
        "image_id": int(item["image_id"]),
        "category_id": int(item["category_id"]),
        "segmentation": item["segmentation"],
        "score": float(item["score"]),
    }


def evaluate(coco, predictions, ids):
    selected = [canonical(p) for p in predictions if int(p["image_id"]) in ids]
    result = coco.loadRes(copy.deepcopy(selected))
    evaluator = COCOeval(coco, result, "segm")
    evaluator.params.imgIds = sorted(ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return [float(x) for x in evaluator.stats]


def main():
    coco = COCO(str(ANN))
    all_ids = sorted(coco.getImgIds())
    dev_ids = set(all_ids[:500])
    test_ids = set(all_ids[500:])
    baseline = json.loads((OUT / "predictions_baseline.json").read_text(encoding="utf-8"))
    method = json.loads((OUT / "fixed_selected_full/predictions.json").read_text(encoding="utf-8"))
    base_stats = evaluate(coco, baseline, test_ids)
    method_stats = evaluate(coco, method, test_ids)
    payload = {
        "protocol": "first 500 sorted COCO image IDs used for development; remaining 4500 used once for final evaluation",
        "development_images": len(dev_ids),
        "evaluation_images": len(test_ids),
        "baseline": base_stats,
        "method": method_stats,
        "delta": [float(y - x) for x, y in zip(base_stats, method_stats)],
    }
    (OUT / "DEV500_TEST4500_EVAL.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
