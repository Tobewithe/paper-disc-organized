"""Compare simple fixed logit thresholds behind a prediction-size gate on 500 dev images."""

from __future__ import annotations

import json
import sys
import copy
import os
from collections import defaultdict
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
    "results/mask_boundary_route_20260914/fixed_dev500"
    if END2END
    else "results/mask_boundary_route_20260914/fixed_dev500_one2many"
)
OUT.mkdir(parents=True, exist_ok=True)
THRESHOLDS = (0.0, 0.5, 0.75, 1.0)
AREA_GATES = (16**2, 24**2, 32**2, 48**2)


def tkey(value: float) -> str:
    return str(value).replace(".", "p")


def evaluate(coco, predictions, image_ids):
    result = coco.loadRes(copy.deepcopy(predictions))
    evaluator = COCOeval(coco, result, "segm")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return [float(x) for x in evaluator.stats]


def main():
    probe.IMAGE_DIR = ROOT / "data/images/val500"
    coco = COCO(str(probe.ANN_FILE))
    model = YOLO(str(probe.WEIGHTS))
    model.model.model[-1].end2end = END2END
    cat_by_name = {c["name"]: int(c["id"]) for c in coco.loadCats(coco.getCatIds())}
    captures = probe.collect_prediction_captures(model)
    outputs = {t: [] for t in THRESHOLDS}
    image_ids = []
    for image_name, capture in sorted(captures.items()):
        image_id = int(Path(image_name).stem)
        image_ids.append(image_id)
        logits, _ = probe.decode_logits(
            capture["proto"], capture["pred"], capture["img_shape"], capture["orig_shape"]
        )
        pred = capture["pred"].numpy()
        for i in range(pred.shape[0]):
            category_id = cat_by_name.get(str(model.names[int(pred[i, 5])]))
            if category_id is None:
                continue
            for threshold in THRESHOLDS:
                binary = logits[i].gt(threshold).numpy().astype(np.uint8)
                if not binary.any():
                    continue
                encoded = mask_utils.encode(np.asfortranarray(binary))
                encoded["counts"] = encoded["counts"].decode("ascii")
                encoded["size"] = [int(x) for x in encoded["size"]]
                outputs[threshold].append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": encoded,
                        "score": float(pred[i, 4]),
                    }
                )

    baseline = outputs[0.0]
    baseline_by_image = defaultdict(list)
    for item in baseline:
        baseline_by_image[int(item["image_id"])].append(item)
    variants_by_image = {}
    for threshold in THRESHOLDS[1:]:
        grouped = defaultdict(list)
        for item in outputs[threshold]:
            grouped[int(item["image_id"])].append(item)
        variants_by_image[threshold] = grouped

    hybrids = {}
    for area_gate in AREA_GATES:
        for threshold in THRESHOLDS[1:]:
            hybrid = []
            variants = variants_by_image[threshold]
            for image_id in image_ids:
                var_index = {
                    (int(x["category_id"]), round(float(x["score"]), 12)): x
                    for x in variants.get(image_id, [])
                }
                for base in baseline_by_image.get(image_id, []):
                    chosen = base
                    if float(mask_utils.area(base["segmentation"])) < area_gate:
                        chosen = var_index.get(
                            (int(base["category_id"]), round(float(base["score"]), 12)), base
                        )
                    hybrid.append(chosen)
            hybrids[(area_gate, threshold)] = hybrid

    metrics = {"baseline": evaluate(coco, baseline, image_ids)}
    for (area_gate, threshold), hybrid in hybrids.items():
        metrics[f"area_lt_{area_gate}_t{tkey(threshold)}"] = evaluate(coco, hybrid, image_ids)
    payload = {
        "images": len(image_ids),
        "end2end": END2END,
        "inference_branch": "one-to-one NMS-free" if END2END else "one-to-many plus NMS",
        "area_gates": list(AREA_GATES),
        "gate_uses_gt": False,
        "metrics": metrics,
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
