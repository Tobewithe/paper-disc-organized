"""Paired image-cluster contrast and high-similarity audit for 7N rows."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def main(args):
    with gzip.open(args.rows, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 71269
    image = np.asarray([r["image_id"] for r in rows], dtype=np.int64)
    mask = np.asarray([r["original_iou"] for r in rows], dtype=np.float64)
    box = np.asarray([r["box_iou"] for r in rows], dtype=np.float64)
    similarity = np.asarray([r["h_feature_cos"] for r in rows], dtype=np.float64)
    nearest = np.asarray([r["h_top1"] for r in rows], dtype=np.float64)
    gain = np.asarray([r["h_top5"] - r["random5"] for r in rows], dtype=np.float64)
    radius = np.asarray([r["radius"] for r in rows], dtype=np.float64)
    good = box >= .75
    failing = mask < .75
    groups = {
        "failure_good_box": good & failing,
        "success_good_box": good & ~failing,
        "failure_good_box_high_h_similarity": good & failing & (similarity >= .9),
        "success_good_box_high_h_similarity": good & ~failing & (similarity >= .9),
        "severe_failure_high_h_similarity": (mask < .5) & (similarity >= .9),
    }
    measures = {}
    for name, selected in groups.items():
        measures[name] = {
            "candidates": int(selected.sum()),
            "images": len(set(image[selected])),
            "mean_h_feature_cos": float(similarity[selected].mean()),
            "mean_h_top1_effect_cos": float(nearest[selected].mean()),
            "h_top1_nonpositive_fraction": float(np.mean(nearest[selected] <= 0)),
            "mean_h_top5_minus_random5": float(gain[selected].mean()),
            "median_oracle_radius": float(np.median(radius[selected])),
        }
    unique, inv = np.unique(image, return_inverse=True)
    rng = np.random.default_rng(20260927)
    draws = rng.integers(0, len(unique), size=(args.bootstrap, len(unique)), dtype=np.int32)
    totals = {}
    for name in ("failure_good_box", "success_good_box"):
        selected = groups[name]
        totals[name] = {
            "sum": np.bincount(inv, weights=np.where(selected, gain, 0), minlength=len(unique)),
            "count": np.bincount(inv, weights=selected.astype(np.int32), minlength=len(unique)),
        }
    f, s = totals["failure_good_box"], totals["success_good_box"]
    boot = (f["sum"][draws].sum(axis=1) / f["count"][draws].sum(axis=1) -
            s["sum"][draws].sum(axis=1) / s["count"][draws].sum(axis=1))
    diff = float(measures["failure_good_box"]["mean_h_top5_minus_random5"] -
                 measures["success_good_box"]["mean_h_top5_minus_random5"])
    result = {
        "groups": measures,
        "failure_minus_success_good_box_h_retrieval_gain": {
            "mean": diff, "image_cluster_ci95": np.quantile(boot, [.025, .975]).tolist(),
            "bootstrap": args.bootstrap,
        },
        "high_similarity_threshold": .9,
        "status": "descriptive_followup_to_frozen_retrieval",
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    main(parser.parse_args())
