"""Measure mask changes from official multi-segment merging at source resolution."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from pycocotools.coco import COCO


def ratio(a, b):
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 1.0


def summarize(rows):
    if not rows:
        return {"n": 0}
    n = len(rows)
    return {
        "n": n,
        "mean_merged_vs_separate_iou": sum(r["merged_vs_separate_iou"] for r in rows) / n,
        "median_merged_vs_separate_iou": float(np.median([r["merged_vs_separate_iou"] for r in rows])),
        "merged_vs_separate_iou_below75": sum(r["merged_vs_separate_iou"] < .75 for r in rows),
        "merged_vs_separate_iou_below90": sum(r["merged_vs_separate_iou"] < .9 for r in rows),
        "mean_extra_fraction_of_gt": sum(r["merge_extra_pixels"] / r["gt_pixels"] for r in rows) / n,
        "mean_lost_fraction_of_gt": sum(r["merge_lost_pixels"] / r["gt_pixels"] for r in rows) / n,
        "extra_pixels": sum(r["merge_extra_pixels"] for r in rows),
        "lost_pixels": sum(r["merge_lost_pixels"] for r in rows),
        "mean_separate_vs_coco_iou": sum(r["separate_vs_coco_iou"] for r in rows) / n,
    }


def main(args):
    sys.path.insert(0, str(args.root / "shared/vendor/ultralytics_8_4_100"))
    from ultralytics.data.converter import merge_multi_segment
    from ultralytics.data.utils import polygon2mask

    coco = COCO(str(args.annotations))
    rows_by_id = {}
    with args.joined.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows_by_id[row["annotation_id"]] = row
    assert len(rows_by_id) == 36335
    ordinary = [a for a in coco.anns.values() if not a.get("iscrowd", 0) and not a.get("ignore", 0)]
    multi = [a for a in ordinary if len(a["segmentation"]) > 1]
    single = [a for a in ordinary if len(a["segmentation"]) == 1]
    assert len(multi) == 3522
    controls = random.Random(20260923).sample(sorted(single, key=lambda a: a["id"]), 1000)
    results = []
    for number, ann in enumerate(multi + controls, 1):
        image = coco.imgs[ann["image_id"]]
        h, w = image["height"], image["width"]
        segments = [np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in ann["segmentation"]]
        separate = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(separate, [s.astype(np.int32) for s in segments], color=1)
        merged_parts = merge_multi_segment(ann["segmentation"]) if len(segments) > 1 else segments
        merged = polygon2mask((h, w), [np.concatenate(merged_parts).reshape(-1)], color=1, downsample_ratio=1)
        gt = coco.annToMask(ann).astype(bool)
        separate = separate.astype(bool)
        merged = merged.astype(bool)
        assert gt.any()
        prior = rows_by_id[ann["id"]]
        results.append({
            "image_id": ann["image_id"], "annotation_id": ann["id"],
            "polygon_count": len(segments), "area_group": prior["area_group"],
            "raw_state": prior["geometry_state"],
            "gt_pixels": int(gt.sum()),
            "separate_vs_coco_iou": ratio(separate, gt),
            "merged_vs_coco_iou": ratio(merged, gt),
            "merged_vs_separate_iou": ratio(merged, separate),
            "merge_extra_pixels": int((merged & ~separate).sum()),
            "merge_lost_pixels": int((separate & ~merged).sum()),
        })
        if number % 500 == 0:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "PROGRESS.json").write_text(json.dumps({"annotations": number, "total": len(multi) + len(controls)}), encoding="utf-8")
    groups = defaultdict(list)
    for row in results:
        key = "multi" if row["polygon_count"] > 1 else "single_control"
        groups[key].append(row)
        groups[f"{key}:{row['raw_state']}"].append(row)
        groups[f"{key}:{row['area_group']}"].append(row)
    summary = {
        "source": {"multi": len(multi), "single_controls": len(controls)},
        "groups": {name: summarize(rows) for name, rows in groups.items()},
        "limitations": "At original grid, merge effect isolated from later label rounding, polygon resampling, letterbox, overlap resolution and optimizer; checkpoint does not contain historical label masks",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ANNOTATIONS.jsonl").write_text("\n".join(json.dumps(row) for row in results) + "\n", encoding="utf-8")
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"annotations": len(results)}), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("root", "annotations", "joined", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
