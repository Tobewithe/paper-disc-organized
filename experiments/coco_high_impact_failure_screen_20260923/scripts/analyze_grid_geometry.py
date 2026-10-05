"""Control simple GT geometry when comparing grid construction across outcomes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO


def mean(values):
    return float(np.mean(values)) if values else None


def main(args):
    coco = COCO(str(args.annotations))
    rows = []
    with args.objects.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            ann = coco.anns[record["annotation_id"]]
            x, y, w, h = ann["bbox"]
            record.update({
                "area": ann["area"], "category_id": ann["category_id"],
                "aspect": w / h, "polygon_count": len(ann["segmentation"]),
                "grid_pass": record["grid_best_iou"] >= .75,
                "area_bin": next((f"<{edge}" for edge in (64, 128, 256, 512, 1024) if ann["area"] < edge), ">=1024"),
                "fill_bin": "<.25" if record["fill"] < .25 else ".25-.5" if record["fill"] < .5 else ">=.5",
            })
            rows.append(record)
    small_low = [row for row in rows if row["group"] in ("small_low_fail", "small_low_good")]
    bins = defaultdict(list)
    for row in small_low:
        bins[(row["group"], row["area_bin"], row["fill_bin"])].append(row)
    bin_summary = [
        {"group": key[0], "area_bin": key[1], "fill_bin": key[2], "n": len(group),
         "grid_pass_fraction": mean([r["grid_pass"] for r in group]),
         "mean_area": mean([r["area"] for r in group]), "mean_fill": mean([r["fill"] for r in group])}
        for key, group in sorted(bins.items())
    ]
    failures = [row for row in small_low if row["group"] == "small_low_fail"]
    controls = [row for row in small_low if row["group"] == "small_low_good"]
    controls_by_category = defaultdict(list)
    for control in controls:
        controls_by_category[control["category_id"]].append(control)
    pairs = []
    for failure in failures:
        eligible = [r for r in controls_by_category[failure["category_id"]]
                    if .5 <= r["area"] / failure["area"] <= 2
                    and abs(r["fill"] - failure["fill"]) <= .1
                    and .5 <= r["aspect"] / failure["aspect"] <= 2]
        if not eligible:
            continue
        match = min(eligible, key=lambda r: abs(np.log(r["area"] / failure["area"]))
                    + abs(r["fill"] - failure["fill"]) + .25 * abs(np.log(r["aspect"] / failure["aspect"])))
        pairs.append((failure, match))
    result = {
        "source_counts": {"failure": len(failures), "control": len(controls)},
        "area_fill_bins": bin_summary,
        "matched": {"n": len(pairs), "unique_control_ids": len({b["annotation_id"] for _, b in pairs}),
                    "failure_grid_pass_fraction": mean([a["grid_pass"] for a, _ in pairs]),
                    "control_grid_pass_fraction": mean([b["grid_pass"] for _, b in pairs]),
                    "failure_mean_area": mean([a["area"] for a, _ in pairs]),
                    "control_mean_area": mean([b["area"] for _, b in pairs]),
                    "failure_mean_fill": mean([a["fill"] for a, _ in pairs]),
                    "control_mean_fill": mean([b["fill"] for _, b in pairs])},
        "note": "Matching is descriptive, with replacement, on annotation category, area, fill and aspect. It is not causal and retains outcome-selected groups."
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SUMMARY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"matched_pairs": len(pairs)}), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("objects", "annotations", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
