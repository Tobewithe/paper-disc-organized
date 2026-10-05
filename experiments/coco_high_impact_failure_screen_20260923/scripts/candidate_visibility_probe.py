"""Check whether large raw-mask repair opportunities are reachable in normal output."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO


def summarize(rows):
    result = {"n": len(rows)}
    for key in ("oracle_raw_in_top300", "oracle_raw_retained", "oracle_raw_class_agrees",
                "oracle_raw_gt_class_retained",
                "normal_box75", "oracle_raw_equals_box_matched_raw"):
        result[key] = sum(row[key] for row in rows)
    result["mean_oracle_gt_class_score"] = float(np.mean([r["oracle_gt_class_score"] for r in rows]))
    result["median_oracle_gt_class_score"] = float(np.median([r["oracle_gt_class_score"] for r in rows]))
    return result


def main(args):
    coco = COCO(str(args.annotations))
    category_by_index = sorted(coco.cats)
    assert len(category_by_index) == 80 and coco.cats[category_by_index[0]]["name"] == "person"
    records = []
    with args.joined.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["area_group"] != "small":
                continue
            ann = coco.anns[row["annotation_id"]]
            fill = ann["area"] / (ann["bbox"][2] * ann["bbox"][3])
            if fill >= .5 or row["geometry_state"] not in ("box_good_mask_unavailable", "joint_good"):
                continue
            records.append(row)
    sample_ids = set()
    if args.optimized_objects:
        with args.optimized_objects.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["group"] != "joint_good_control":
                    sample_ids.add(row["annotation_id"])
    results = []
    for number, row in enumerate(records, 1):
        representative = row["best_mask_with_box75"]
        raw_id = representative["raw_id"]
        with np.load(args.archive / "images" / f"{row['image_id']:012d}.npz") as archive:
            top_ids = archive["top_ids"]
            top_scores = archive["top_scores"]
            top_classes = archive["top_classes"]
            positions = np.flatnonzero(top_ids == raw_id)
        group = "failure" if row["geometry_state"] == "box_good_mask_unavailable" else "success_control"
        results.append({
            "group": group, "image_id": row["image_id"], "annotation_id": row["annotation_id"],
            "oracle_raw_id": raw_id,
            "oracle_raw_in_top300": bool(len(positions)),
            "oracle_raw_retained": bool(any(top_scores[pos] > .001 for pos in positions)),
            "oracle_raw_class_agrees": bool(any(category_by_index[int(top_classes[pos])] == representative["predicted_class"] for pos in positions)),
            "oracle_raw_gt_class_retained": bool(any(
                top_scores[pos] > .001 and category_by_index[int(top_classes[pos])] == row["category_id"]
                for pos in positions
            )),
            "normal_box75": bool(row["bbox_coco75"]),
            "oracle_raw_equals_box_matched_raw": raw_id == row["bbox_raw_id"],
            "oracle_gt_class_score": representative["gt_class_score"],
            "in_optimization_sample": row["annotation_id"] in sample_ids,
        })
        if number % 500 == 0:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "PROGRESS.json").write_text(json.dumps({"done": number, "total": len(records)}), encoding="utf-8")
    groups = defaultdict(list)
    for row in results:
        groups[row["group"]].append(row)
        if row["in_optimization_sample"]:
            groups["optimized_failure_sample"].append(row)
    output = {"groups": {key: summarize(rows) for key, rows in groups.items()}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "OBJECTS.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")
    (args.out / "SUMMARY.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"objects": len(results)}), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("joined", "archive", "annotations", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--optimized-objects", type=Path)
    main(parser.parse_args())
