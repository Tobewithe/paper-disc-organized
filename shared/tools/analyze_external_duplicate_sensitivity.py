"""Evaluate fixed, prediction-independent policies for Bama duplicate records."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import run_yolo26_external_full as task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("Preserve existing sensitivity results; use a fresh output directory")
    files = [source / p for p in ("manifests/BamaPig2D_all.json", "BamaPig2D_all_predictions.json",
        "BamaPig2D_all_gt_analysis.csv", "duplicate_annotation_audit.json", "summary.json", "output_audit.json")]
    files.extend([Path(__file__).resolve(), task.ROOT / "experiments/yolo26_external_duplicate_sensitivity_20260907_protocol.md"])
    hashes = {str(p): task.sha(p) for p in files}
    if task.read(source / "output_audit.json")["status"] != "PASS":
        raise RuntimeError("Source completion audit did not pass")
    coco = task.read(files[0])
    predictions = task.read(files[1])
    with files[2].open(encoding="utf-8", newline="") as handle:
        gt_rows = list(csv.DictReader(handle))
    duplicates = task.read(files[3])
    baseline = next(r for r in task.read(files[4])["results"] if r["dataset"] == "BamaPig2D" and r["split"] == "all")
    groups = [r["image_ids"] for r in duplicates["details"] if r["dataset"] == "BamaPig2D"]
    duplicate_ids = {i for group in groups for i in group}
    all_ids = {i["id"] for i in coco["images"]}
    if len(groups) != 77 or len(duplicate_ids) != 154 or not duplicate_ids <= all_ids:
        raise RuntimeError("Unexpected duplicate group scope")
    if len(all_ids) != 3340 or len(gt_rows) != baseline["gt"] or len(predictions) != baseline["predictions"]:
        raise RuntimeError("Baseline accounting mismatch")
    singleton_ids = all_ids - duplicate_ids
    selections = {"keep_min_id": singleton_ids | {min(g) for g in groups},
                  "keep_max_id": singleton_ids | {max(g) for g in groups},
                  "singleton_only": singleton_ids}
    output.mkdir(parents=True)
    results = []
    for policy, ids in selections.items():
        gt = COCO()
        gt.dataset = {**coco, "images": [i for i in coco["images"] if i["id"] in ids],
            "annotations": [a for a in coco["annotations"] if a["image_id"] in ids]}
        gt.createIndex()
        # COCO.loadRes mutates annotation dictionaries; isolate each policy.
        selected_preds = [{k: p[k] for k in ("image_id", "category_id", "score", "segmentation")}
                          for p in predictions if p["image_id"] in ids]
        evaluator = COCOeval(gt, gt.loadRes(selected_preds), "segm")
        evaluator.params.imgIds = sorted(ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        stats = [float(x) for x in evaluator.stats]
        if not all(math.isfinite(x) for x in stats):
            raise RuntimeError("Nonfinite metrics")
        selected_rows = [r for r in gt_rows if int(r["image_id"]) in ids]
        counts = Counter(r["primary_class"] for r in selected_rows)
        if len(selected_rows) != len(gt.anns):
            raise RuntimeError("GT class accounting mismatch")
        failures = len(selected_rows) - counts["C"]
        row = {"policy": policy, "images": len(ids), "gt": len(selected_rows), "predictions": len(selected_preds),
            "mask_ap": stats[0], "mask_ap50": stats[1], "mask_ap75": stats[2], "mask_ar100": stats[8],
            "failed_gt": failures, "failure_rate": failures / len(selected_rows),
            "ap_delta_points": 100 * (stats[0] - baseline["mask_ap"]),
            "failure_rate_delta_points": 100 * (failures / len(selected_rows) - baseline["failure_rate"]),
            **{k: counts[k] for k in task.PRIMARY}}
        task.write(output / f"{policy}.json", {**row, "selected_image_ids": sorted(ids), "coco_stats": stats})
        results.append(row)
    if any(task.sha(p) != digest for p, digest in hashes.items()):
        raise RuntimeError("Source files changed during evaluation")
    task.write(output / "input_sha256.json", hashes)
    task.csv_write(output / "summary.csv", results)
    task.write(output / "summary.json", {"status": "completed", "dataset": "BamaPig2D", "baseline": baseline,
        "results": results, "input_hashes_verified": len(hashes),
        "scope": "Fixed representative/singleton sensitivity, not independent samples or optimal label selection"})
    print("DUPLICATE_SENSITIVITY_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
