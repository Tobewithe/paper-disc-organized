"""Evaluate the frozen method on disjoint COCO image blocks.

Block 0 contains the 500 development images and is reported only as a
reference. Blocks 1-5 are disjoint from development and are the robustness
evidence.
"""

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


def evaluate(coco, predictions, image_ids):
    # Keep only canonical result fields. COCO.loadRes mutates dictionaries by
    # adding bbox/area/id; reloading those derived fields routes evaluation
    # through the bbox branch and corrupts segmentation area stratification.
    subset = [
        {
            "image_id": int(p["image_id"]),
            "category_id": int(p["category_id"]),
            "segmentation": p["segmentation"],
            "score": float(p["score"]),
        }
        for p in predictions
        if int(p["image_id"]) in image_ids
    ]
    result = coco.loadRes(copy.deepcopy(subset))
    evaluator = COCOeval(coco, result, "segm")
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return [float(x) for x in evaluator.stats]


def main():
    coco = COCO(str(ANN))
    all_ids = sorted(coco.getImgIds())
    baseline = json.loads((OUT / "predictions_baseline.json").read_text(encoding="utf-8"))
    selected = json.loads(
        (OUT / "fixed_selected_full/predictions.json").read_text(encoding="utf-8")
    )
    blocks = [all_ids[:500]]
    remaining = all_ids[500:]
    block_size = 900
    blocks.extend(remaining[i : i + block_size] for i in range(0, len(remaining), block_size))
    payload = {
        "split": "500 sorted-ID development images followed by five disjoint 900-image evaluation blocks",
        "blocks": [],
    }
    for index, ids in enumerate(blocks):
        b = evaluate(coco, baseline, set(ids))
        m = evaluate(coco, selected, set(ids))
        payload["blocks"].append(
            {
                "index": index,
                "role": "development" if index == 0 else "evaluation",
                "images": len(ids),
                "first_image_id": ids[0],
                "last_image_id": ids[-1],
                "baseline": b,
                "method": m,
                "delta": [float(y - x) for x, y in zip(b, m)],
            }
        )
    (OUT / "BLOCK_ROBUSTNESS_EVAL.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
