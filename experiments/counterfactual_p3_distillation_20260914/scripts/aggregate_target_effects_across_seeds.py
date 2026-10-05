"""Bootstrap target effects after averaging paired deltas across seeds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    "raw_p3_best_box_iou", "raw_best_box_iou", "raw_box50", "raw_center_error_norm",
    "final_box_iou", "mask_iou", "target_coverage", "prediction_purity",
    "same_neighbor_leak_pred", "background_leak_pred", "boundary_f1", "final_mask75",
]


def interval(values: np.ndarray, seed: int, reps: int = 10000):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(reps)])
    lo, hi = np.quantile(means, [.025, .975])
    return float(values.mean()), float(lo), float(hi)


def main() -> None:
    seed_frames = []
    for seed in range(3):
        data = pd.read_csv(ROOT / f"rich_targeted_s{seed}" / "per_target.csv")
        baseline = data[data.arm == "baseline"]
        method = data[data.arm == "method"]
        paired = baseline.merge(method, on=["cohort", "pair_id", "image_id", "annotation_id"], suffixes=("_baseline", "_method"))
        frame = paired[["cohort", "pair_id", "image_id", "annotation_id"]].copy()
        frame["seed"] = seed
        for metric in METRICS:
            frame[metric] = paired[f"{metric}_method"] - paired[f"{metric}_baseline"]
        seed_frames.append(frame)
    all_seeds = pd.concat(seed_frames, ignore_index=True)
    averaged = all_seeds.groupby(["cohort", "pair_id", "image_id", "annotation_id"], as_index=False)[METRICS].mean()

    rows = []
    for cohort, group in averaged.groupby("cohort"):
        for index, metric in enumerate(METRICS):
            mean, low, high = interval(group[metric].to_numpy(), 1414 + index)
            rows.append({"analysis": "cohort", "cohort": cohort, "metric": metric, "n": len(group), "delta": mean, "ci_low": low, "ci_high": high})

    failure = averaged[averaged.cohort == "raw_geometry_small"]
    control = averaged[averaged.cohort == "matched_small_control"]
    joined = failure.merge(control, on="pair_id", suffixes=("_failure", "_control"))
    for index, metric in enumerate(METRICS):
        values = (joined[f"{metric}_failure"] - joined[f"{metric}_control"]).to_numpy()
        mean, low, high = interval(values, 2414 + index)
        rows.append({"analysis": "failure_minus_control", "cohort": "paired", "metric": metric, "n": len(joined), "delta": mean, "ci_low": low, "ci_high": high})

    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "rich_across_seed_target_effects.csv", index=False)
    focus = result[((result.analysis == "cohort") & (result.cohort == "raw_geometry_small")) | (result.analysis == "failure_minus_control")]
    (ROOT / "rich_across_seed_target_effects.json").write_text(json.dumps(focus.to_dict("records"), indent=2), encoding="utf-8")
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
