"""Screen all fixed-Box75, raw-Mask75-unavailable COCO objects by pixel error."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def summarize(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    counts = Counter(r["repair"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    neighbor = sum(r["neighbor_fp"] for r in rows)
    background = sum(r["background_fp"] for r in rows)
    crowd = sum(r["crowd_fp"] for r in rows)
    assert neighbor + background + crowd == fp
    return {
        "n": n,
        "normal_mask75_failed": sum(not r["normal_success"] for r in rows),
        "repair_counts": dict(counts),
        "fp_only_capable": sum(r["remove_fp_iou"] >= .75 for r in rows),
        "fn_only_capable": sum(r["fill_fn_iou"] >= .75 for r in rows),
        "crop_support_below75": sum(r["support"] < .75 for r in rows),
        "mean_mask_iou": sum(r["mask_iou"] for r in rows) / n,
        "mean_coverage": sum(r["coverage"] for r in rows) / n,
        "mean_purity": sum(r["purity"] for r in rows) / n,
        "mean_crop_support": sum(r["support"] for r in rows) / n,
        "fp_pixels": fp,
        "neighbor_fp_pixels": neighbor,
        "background_fp_pixels": background,
        "crowd_fp_pixels": crowd,
        "neighbor_fraction_of_fp": neighbor / fp if fp else 0,
        "background_fraction_of_fp": background / fp if fp else 0,
        "median_fill": sorted(r["fill"] for r in rows)[n // 2],
        "mask_iou_below50": sum(r["mask_iou"] < .5 for r in rows),
        "mask_iou_50_to70": sum(.5 <= r["mask_iou"] < .7 for r in rows),
        "mask_iou_70_to75": sum(.7 <= r["mask_iou"] < .75 for r in rows),
    }


def main(args):
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    ann_by_id = {a["id"]: a for a in annotations["annotations"]}
    assert len(annotations["images"]) == 5000
    groups = defaultdict(list)
    eligible = Counter()
    total = Counter()
    all_rows = 0
    with args.joined.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            all_rows += 1
            annotation = ann_by_id[row["annotation_id"]]
            fill = annotation["area"] / (annotation["bbox"][2] * annotation["bbox"][3])
            size = row["area_group"]
            fill_group = "low_fill" if fill < .5 else ("medium_fill" if fill < .75 else "high_fill")
            total[(size, fill_group)] += 1
            if row["box_max"] >= .75:
                eligible[(size, fill_group)] += 1
            if row["geometry_state"] != "box_good_mask_unavailable":
                continue
            w = row["best_mask_with_box75"]
            assert w is not None and w["mask_iou"] < .75 and row["box_max"] >= .75
            remove = row["fp_removal_iou"] >= .75
            fill_only = row["fn_fill_iou"] >= .75
            repair = ("either_single_edit" if remove and fill_only else
                      "fp_only" if remove else "fn_only" if fill_only else "both_edits_needed")
            rec = {
                "image_id": row["image_id"], "annotation_id": row["annotation_id"],
                "size": size, "fill_group": fill_group, "fill": fill,
                "mask_iou": w["mask_iou"], "coverage": w["coverage"],
                "purity": w["purity"], "support": w["crop_support"],
                "fp": w["fp"], "neighbor_fp": w["neighbor_fp"],
                "background_fp": w["background_fp"], "crowd_fp": w["crowd_fp"],
                "remove_fp_iou": row["fp_removal_iou"],
                "fill_fn_iou": row["fn_fill_iou"],
                "normal_success": row["segm_coco75"], "repair": repair,
            }
            groups["all"].append(rec)
            groups[size].append(rec)
            groups[fill_group].append(rec)
            groups[f"{size}:{fill_group}"].append(rec)
    assert all_rows == 36335 and len(groups["all"]) == 6688
    result = {
        "scope": "Existing GT-assisted best-Box75 representative of each raw-unavailable Mask75 GT; diagnostic only, not AP or achievable model gain",
        "source_rows": all_rows,
        "eligible_box75_by_size_fill": {f"{s}:{f}": eligible[(s, f)] for s in ("small", "medium", "large") for f in ("low_fill", "medium_fill", "high_fill")},
        "group_summaries": {name: summarize(rows) for name, rows in groups.items()},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SCREEN.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"gt": all_rows, "failure_gt": len(groups["all"])}), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("joined", "annotations", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
