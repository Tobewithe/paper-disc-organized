"""Check complete output accounting, duplicate labels and historical mask parity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as masks
from scipy.optimize import linear_sum_assignment

import run_yolo26_external_full as task


def canonical_masks(coco, image_id):
    image = next(i for i in coco["images"] if i["id"] == image_id)
    keys = []
    for ann in coco["annotations"]:
        if ann["image_id"] == image_id:
            rle = masks.merge(masks.frPyObjects(ann["segmentation"], image["height"], image["width"]))
            keys.append(rle["counts"].decode("ascii"))
    return sorted(keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duplicates-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    plan = task.read(output / "plan.json")
    datasets = {d: task.read(output / "manifests" / f"{d}_all.json") for d in ("FaroPigSeg", "BamaPig2D")}
    roots = {s["dataset"]: Path(s["image_root"]) for s in plan["splits"]}
    images_by_path = {str(roots[d] / i["file_name"]): (d, i) for d, coco in datasets.items() for i in coco["images"]}
    groups = task.read(output / "exact_duplicate_files.json")
    duplicate_reports = []
    for paths in groups:
        entries = [images_by_path[p] for p in paths]
        label_keys = [canonical_masks(datasets[d], i["id"]) for d, i in entries]
        pair_quality = []
        image = entries[0][1]
        left = [{"size": [image["height"], image["width"]], "counts": k.encode("ascii")} for k in label_keys[0]]
        for keys in label_keys[1:]:
            right = [{"size": [image["height"], image["width"]], "counts": k.encode("ascii")} for k in keys]
            ious = masks.iou(left, right, [0] * len(right))
            rows, cols = linear_sum_assignment(-ious)
            matched = ious[rows, cols]
            pair_quality.append({"instance_counts": [len(left), len(right)],
                "matched_iou_mean": float(matched.mean()), "matched_iou_min": float(matched.min()),
                "matched_below_075": int((matched < .75).sum()),
                "unmatched_instances": abs(len(left) - len(right))})
        duplicate_reports.append({"dataset": entries[0][0], "image_ids": [i["id"] for _, i in entries],
            "source_splits": [i["source_split"] for _, i in entries], "paths": paths,
            "same_mask_multiset": all(k == label_keys[0] for k in label_keys[1:]), "pair_quality": pair_quality})
    qualities = [q for report in duplicate_reports for q in report["pair_quality"]]
    duplicate_summary = {"groups": len(groups), "redundant_file_count": sum(len(g) - 1 for g in groups),
        "cross_split_groups": sum(len(set(r["source_splits"])) > 1 for r in duplicate_reports),
        "different_annotation_groups": sum(not r["same_mask_multiset"] for r in duplicate_reports),
        "different_instance_count_pairs": sum(q["unmatched_instances"] > 0 for q in qualities),
        "pairs_with_matched_iou_below_075": sum(q["matched_below_075"] > 0 for q in qualities),
        "pair_mean_iou_median": float(np.median([q["matched_iou_mean"] for q in qualities])),
        "unique_content_images": plan["total_images"] - sum(len(g) - 1 for g in groups), "details": duplicate_reports}
    task.write(output / "duplicate_annotation_audit.json", duplicate_summary)
    if args.duplicates_only:
        print(json.dumps({k: v for k, v in duplicate_summary.items() if k != "details"}))
        return
    summary = task.read(output / "summary.json")
    if summary["status"] != "completed":
        raise RuntimeError("Full inference/evaluation has no completion receipt")
    checks = []
    for spec in plan["splits"]:
        run = output / "inference" / f"{spec['dataset']}_{spec['split']}"
        config = task.read(run / "inference_config.json")
        if config["model_meta"]["ultralytics_version"] != "8.4.100" or config["config"]["conf"] != .05:
            raise RuntimeError("Inference configuration drift")
        with (run / "inference_cache.jsonl").open(encoding="utf-8") as handle:
            cache = list(map(json.loads, handle))
        source_images = {i["id"]: i for i in task.read(spec["manifest"])["images"]}
        if len(cache) != spec["images"] or set(source_images) != {r["image_id"] for r in cache}:
            raise RuntimeError("Image scope mismatch")
        exported = task.read(run / "predictions.json")
        export_map = {(p["image_id"], p["pred_id"]): p for p in exported}
        if len(export_map) != len(exported):
            raise RuntimeError("Duplicate exported prediction key")
        for record in cache:
            for pred in record["predictions"]:
                exported_pred = export_map[(record["image_id"], pred["pred_id"])]
                if pred["score"] != exported_pred["score"] or pred["mask_rle"] != exported_pred["segmentation"]:
                    raise RuntimeError("Cache/export score or RLE mismatch")
        checks.append({"dataset": spec["dataset"], "split": spec["split"], "images": len(cache), "predictions": len(exported)})
    historical = []
    old_root = task.LEGACY / "artifacts/inference_cache/yolo26seg_diagnostic_20260901_single_forward_imgsz1024_rectfalse"
    for name, cache_name in (("FaroPigSeg_test", "faropigseg_test"), ("BamaPig2D_eval", "bamapig2d_eval")):
        coco = task.read(output / "manifests" / f"{name}.json")
        source_ids = {i["id"]: i["source_image_id"] for i in coco["images"]}
        with (old_root / cache_name / "diagnostic_cache.jsonl").open(encoding="utf-8") as handle:
            old = {r["image_id"]: r["final_predictions"] for r in map(json.loads, handle)}
        with (output / "inference" / name / "inference_cache.jsonl").open(encoding="utf-8") as handle:
            new = {source_ids[r["image_id"]]: r["predictions"] for r in map(json.loads, handle)}
        changed = [i for i in new if new[i] != old[i]]
        historical.append({"dataset": name, "images": len(new), "different_prediction_images": len(changed), "image_ids": changed})
    hashes = task.read(output / "input_sha256.json")
    changed_inputs = [p for p, digest in hashes.items() if task.sha(p) != digest]
    if changed_inputs:
        raise RuntimeError(f"Changed inputs: {changed_inputs}")
    result = {"status": "PASS", "scope": "Deterministic image/prediction accounting, exact cache/export equality, duplicate masks, input hashes and historical prediction comparison; not method acceptance",
              "splits": checks, "historical_prediction_comparison": historical,
              "duplicates": {k: v for k, v in duplicate_summary.items() if k != "details"},
              "hashes_verified": len(hashes), "auditor_sha256": task.sha(Path(__file__))}
    task.write(output / "output_audit.json", result)
    task.write(output / "output_sha256.json", {str(p.relative_to(output)): task.sha(p) for p in output.rglob("*") if p.is_file() and p.name != "output_sha256.json"})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
