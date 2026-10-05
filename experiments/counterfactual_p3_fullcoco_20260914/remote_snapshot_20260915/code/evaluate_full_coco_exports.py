"""Rescore immutable YOLO exports on all COCO val images with explicit category IDs.

No inference/training is performed. For the 20260914 full-data exports, the YAML
uses an image directory, so Ultralytics exports contiguous one-based categories.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def class_mapping(data_yaml, gt, space):
    names = yaml.safe_load(data_yaml.read_text())["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(k): str(v) for k, v in names.items()}
    assert sorted(names) == list(range(80)), "Expected the complete COCO80 class order"
    by_name = {v["name"]: int(k) for k, v in gt.cats.items()}
    assert set(names.values()) == set(by_name), "YAML and raw COCO category names differ"
    if space == "contiguous_one_based":
        return {k + 1: by_name[name] for k, name in names.items()}
    return {k: k for k in gt.cats}


def task_predictions(predictions, mapping, task, gt):
    result = []
    for row in predictions:
        image_id = int(row["image_id"])
        assert image_id in gt.imgs, f"Unexpected image {image_id}"
        category_id = mapping[int(row["category_id"])]
        item = {"image_id": image_id, "category_id": category_id, "score": row["score"]}
        if task == "segm":
            mask = row["segmentation"]
            info = gt.imgs[image_id]
            assert mask["size"] == [info["height"], info["width"]], "Mask/image coordinate mismatch"
            # No bbox/area: COCO.loadRes must compute detection area from the RLE.
            item["segmentation"] = mask
        else:
            item["bbox"] = row["bbox"]
        result.append(item)
    return result


def average_valid(values):
    values = values[values >= 0]
    return float(values.mean()) if values.size else None


def evaluate(gt, predictions, arm, task):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(predictions)
        ev = COCOeval(gt, dt, task)
        ev.params.imgIds = sorted(gt.imgs)  # includes images with zero predictions
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    names = ["ap", "ap50", "ap75", "aps", "apm", "apl", "ar1", "ar10", "ar100", "ars", "arm", "arl"]
    metrics = {name: float(value) for name, value in zip(names, ev.stats)}
    precision = ev.eval["precision"][:, :, :, 0, -1]
    recall = ev.eval["recall"][:, :, 0, -1]
    i50 = int(np.argmin(abs(ev.params.iouThrs - .5)))
    i75 = int(np.argmin(abs(ev.params.iouThrs - .75)))
    metrics.update(ar50=average_valid(recall[i50]), ar75=average_valid(recall[i75]))
    categories = []
    for k, cat in enumerate(ev.params.catIds):
        categories.append(dict(arm=arm, task=task, category_id=int(cat), category_name=gt.cats[cat]["name"],
                               ap=average_valid(precision[:, :, k]), ap50=average_valid(precision[i50, :, k]),
                               ap75=average_valid(precision[i75, :, k]), ar50=float(recall[i50, k]), ar75=float(recall[i75, k])))
    return dict(arm=arm, task=task, images=len(gt.imgs), predictions=len(predictions), **metrics), categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--category-id-space", required=True, choices=["contiguous_one_based", "coco"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(args.annotation))
    assert len(gt.imgs) == 5000, "This run is the complete COCO val2017 comparison"
    mapping = class_mapping(args.data_yaml, gt, args.category_id_space)
    summary, categories, sources = [], [], []
    for arm in ("baseline_s0", "cfp3r_s0"):
        path = args.exports / arm / "predictions.json"
        predictions = json.loads(path.read_text())
        predicted_images = {int(row["image_id"]) for row in predictions}
        sources.append(dict(arm=arm, path=str(path), predictions=len(predictions),
                            images_without_predictions=sorted(set(gt.imgs) - predicted_images),
                            exported_category_ids=sorted({int(r["category_id"]) for r in predictions})))
        for task in ("bbox", "segm"):
            rows = task_predictions(predictions, mapping, task, gt)
            metrics, per_category = evaluate(gt, rows, arm, task)
            summary.append(metrics)
            categories.extend(per_category)
            write_csv(args.out / "official_metrics.csv", summary)
            print(json.dumps(metrics), flush=True)
            del rows
        del predictions
    differences = []
    for task in ("bbox", "segm"):
        base = next(r for r in summary if r["arm"] == "baseline_s0" and r["task"] == task)
        method = next(r for r in summary if r["arm"] == "cfp3r_s0" and r["task"] == task)
        for key in base:
            if key not in {"arm", "task", "images", "predictions"}:
                differences.append(dict(task=task, metric=key, baseline=base[key], method=method[key], delta_points=100*(method[key]-base[key])))
    write_csv(args.out / "paired_metrics.csv", differences)
    write_csv(args.out / "per_category_metrics.csv", categories)
    manifest = dict(status="complete", image_count=5000, category_id_space=args.category_id_space,
                    category_mapping=mapping, sources=sources, metrics=summary,
                    protocol="Official pycocotools COCOeval; all 5000 images; maxDets=[1,10,100]; original COCO GT; mask area from RLE; one-to-one saved predictions",
                    limitation="One training seed and one full epoch. AP deltas are point estimates without confidence intervals.")
    (args.out / "COMPLETE.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
