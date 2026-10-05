"""Analyze paired Mask75 outcomes for the preselected failure cohort."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def main(args):
    summary = json.loads((args.run / "SUMMARY.json").read_text(encoding="utf-8"))
    matches = json.loads((args.run / "MATCHED_GT75.json").read_text(encoding="utf-8"))
    image_ids = set(summary["image_ids"])
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))["annotations"]
    image_of = {ann["id"]: ann["image_id"] for ann in annotations if ann["image_id"] in image_ids}
    box_area = {ann["id"]: ann["bbox"][2] * ann["bbox"][3] for ann in annotations
                if ann["image_id"] in image_ids}
    target = set()
    with args.joined.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            ann_id = row["annotation_id"]
            if row["image_id"] not in image_ids or row.get("geometry_state") != "box_good_mask_unavailable":
                continue
            if row["area_group"] == "small" and box_area[ann_id] > 0 and row["area"] / box_area[ann_id] < .5:
                target.add(ann_id)
    all_ids = set(image_of)
    cohorts = {"preselected_failure": target, "other_gt": all_ids - target, "all_gt": all_ids}
    original = set(matches["baseline"])
    rng = np.random.default_rng(20260923)
    result = {"images": len(image_ids), "cohorts": {}, "paired_comparisons": {}}
    for cohort_name, ids in cohorts.items():
        arms = {}
        for arm, match_ids in matches.items():
            found = set(match_ids)
            repairs = (found - original) & ids
            damages = (original - found) & ids
            arms[arm] = {"matched75": len(found & ids), "repairs75": len(repairs),
                         "damages75": len(damages), "net75": len(repairs) - len(damages)}
        result["cohorts"][cohort_name] = {"gt": len(ids), "arms": arms}
    if "local4_05" in matches:
        pairs = [("project_0_05", "coeff_05"), ("project_0_10", "coeff_10"),
                 ("project_4_10", "coeff_guard_10"), ("project_0_05", "local4_05")]
    else:
        pairs = [("project_4_05", "coeff_025"), ("project_0_025", "coeff_025"),
                 ("project_0_05", "coeff_05"), ("spatial_05", "project_4_10")]
    for cohort_name, ids in cohorts.items():
        for method, comparator in pairs:
            by_image = defaultdict(int)
            method_ids, comparator_ids = set(matches[method]), set(matches[comparator])
            for ann_id in (method_ids ^ comparator_ids) & ids:
                by_image[image_of[ann_id]] += 1 if ann_id in method_ids else -1
            samples = np.array([by_image[i] for i in summary["image_ids"]], dtype=int)
            indices = rng.integers(0, len(samples), size=(2000, len(samples)))
            totals = samples[indices].sum(1)
            result["paired_comparisons"][f"{cohort_name}:{method}-vs-{comparator}"] = {
                "net_matched75_difference": int(samples.sum()),
                "image_bootstrap_95_ci": np.quantile(totals, [.025, .975]).tolist(),
                "discordant_gt": int(len((method_ids ^ comparator_ids) & ids)),
            }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--joined", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
