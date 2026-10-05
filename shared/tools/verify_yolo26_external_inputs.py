"""Validate expanded split masks/classification against historical cached subsets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

import run_yolo26_external_full as task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    plan = task.read(output / "plan.json")
    fixed = task.load_module("external_preflight_fixed", task.LEGACY / "scripts/analyze_yolo26_diagnostic_cache_full.py")
    old_specs = {s["name"]: s for s in fixed.DATASETS}
    cache_root = task.LEGACY / "artifacts/inference_cache/yolo26seg_diagnostic_20260901_single_forward_imgsz1024_rectfalse"
    with (task.LEGACY / "artifacts/analysis/yolo26seg_diagnostic_full_single_forward_20260901_final02/gt_analysis.csv").open(encoding="utf-8-sig", newline="") as handle:
        historical = {(r["dataset"], int(r["image_id"]), int(r["annotation_id"])): r["primary_class"] for r in csv.DictReader(handle)}
    reports = []
    for spec in [s for s in plan["splits"] if s["split"] in ("test", "eval")]:
        name = f"{spec['dataset']}_{spec['split']}"
        old_spec = old_specs[name]
        old_coco = task.read(old_spec["manifest"])
        old_anns = {(a["image_id"], a["id"]): a for a in old_coco["annotations"]}
        coco = task.read(spec["manifest"])
        new_images = {i["id"]: i for i in coco["images"]}
        with (cache_root / old_spec["cache"] / "diagnostic_cache.jsonl").open(encoding="utf-8") as handle:
            cache = {r["image_id"]: r for r in map(json.loads, handle)}
        translated = {i["id"]: {"predictions": cache[i["source_image_id"]]["final_predictions"]} for i in coco["images"]}
        pixels = 0
        for ann in coco["annotations"]:
            image = new_images[ann["image_id"]]
            old = old_anns[(image["source_image_id"], ann["source_annotation_id"])]
            before = fixed.normalize_gt(old, image["height"], image["width"])
            after = fixed.normalize_gt(ann, image["height"], image["width"])
            pixels += int(np.count_nonzero(mask_utils.decode(before) != mask_utils.decode(after)))
        rows = task.classify(fixed, coco, translated)
        changed = [r for r in rows if historical[(name, r["source_image_id"], r["source_annotation_id"])] != r["primary_class"]]
        if pixels or changed:
            raise RuntimeError(f"Historical mask/classification mismatch: {name}, pixels={pixels}, gt={len(changed)}")
        reports.append({"dataset": name, "images": len(coco["images"]), "gt": len(rows), "mask_xor_pixels": pixels, "classification_mismatches": len(changed)})
    result = {"status": "PASS", "scope": "Historical subset masks and unchanged fixed-classifier semantics, using historical predictions", "subsets": reports, "verifier_sha256": task.sha(Path(__file__))}
    task.write(output / "historical_input_verification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
