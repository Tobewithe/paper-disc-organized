"""Stratify full-val matched-instance results by COCO GT area."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ANN = Path("assets/datasets/coco/annotations/instances_val2017.json")


def size_group(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def tkey(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def main():
    summary = json.loads((RESULTS / "SUMMARY_full5000.json").read_text(encoding="utf-8"))
    policy = summary["policy"]["selected"]
    anns = json.loads(ANN.read_text(encoding="utf-8"))["annotations"]
    area_by_ann = {int(a["id"]): float(a["area"]) for a in anns}
    with (RESULTS / "matched_records_full5000.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    image_ids = sorted({int(r["image_id"]) for r in rows})
    test_ids = set(image_ids[len(image_ids) // 2 :])
    groups = defaultdict(list)
    for row in rows:
        if int(row["image_id"]) not in test_ids:
            continue
        if row["target"] != "1" and row["mask_good"] != "1":
            continue
        area = area_by_ann[int(row["ann_id"])]
        threshold = policy["high"] if float(row[policy["feature"]]) >= policy["cut"] else policy["low"]
        adaptive_iou = float(row[f"iou_{tkey(threshold)}"])
        item = {
            "base": float(row["base_iou"]),
            "adaptive": adaptive_iou,
            "support_base": float(row["base_support"]),
            "support_adaptive": float(row[f"support_{tkey(threshold)}"]),
            "success_base": float(row["base_iou"]) >= 0.75,
            "success_adaptive": adaptive_iou >= 0.75,
        }
        cohort = "target" if row["target"] == "1" else "mask_good"
        groups[(cohort, size_group(area))].append(item)
    output = {
        "split": "held-out second half of sorted image IDs",
        "policy": policy,
        "groups": {},
    }
    for (cohort, size), items in sorted(groups.items()):
        n = len(items)
        def precision(iou, recall):
            den = (1.0 / max(iou, 1e-12)) - (1.0 / max(recall, 1e-12)) + 1.0
            return 1.0 / den if den > 0 else 0.0

        output["groups"][f"{cohort}_{size}"] = {
            "n": n,
            "mean_iou_base": sum(x["base"] for x in items) / n,
            "mean_iou_adaptive": sum(x["adaptive"] for x in items) / n,
            "mean_iou_delta": sum(x["adaptive"] - x["base"] for x in items) / n,
            "mean_support_base": sum(x["support_base"] for x in items) / n,
            "mean_support_adaptive": sum(x["support_adaptive"] for x in items) / n,
            "mean_precision_base": sum(precision(x["base"], x["support_base"]) for x in items) / n,
            "mean_precision_adaptive": sum(precision(x["adaptive"], x["support_adaptive"]) for x in items) / n,
            "success_base": sum(x["success_base"] for x in items) / n,
            "success_adaptive": sum(x["success_adaptive"] for x in items) / n,
        }
    (RESULTS / "SCALE_STRATIFIED_HELDOUT.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
