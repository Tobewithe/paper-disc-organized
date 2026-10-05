import argparse
import json
from pathlib import Path

import numpy as np


def one_group(rows, rng):
    image_ids = np.array([x["image_id"] for x in rows])
    unique, inverse = np.unique(image_ids, return_inverse=True)
    keys = [
        "h_nearest_cos", "h_random_cos", "oracle_radius",
        "retrieved_1_effect_cos", "random_1_effect_cos",
        "retrieved_5_effect_cos", "random_5_effect_cos",
        "h_only_learned_effect_cos", "box_iou", "original_mask_iou",
    ]
    vals = np.array([[x[k] for k in keys] for x in rows], dtype=np.float64)
    dif1 = vals[:, keys.index("retrieved_1_effect_cos")] - vals[:, keys.index("random_1_effect_cos")]
    dif5 = vals[:, keys.index("retrieved_5_effect_cos")] - vals[:, keys.index("random_5_effect_cos")]
    boot = np.empty((2000, 2), dtype=np.float64)
    for b in range(2000):
        counts = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))
        weights = counts[inverse]
        boot[b, 0] = np.average(dif1, weights=weights)
        boot[b, 1] = np.average(dif5, weights=weights)
    return {
        "images": len(unique), "instances": len(rows),
        "fallback": int(sum(x["fallback"] for x in rows)),
        "means": {k: float(vals[:, i].mean()) for i, k in enumerate(keys)},
        "median_oracle_radius": float(np.median(vals[:, keys.index("oracle_radius")])),
        "paired_1": {"mean": float(dif1.mean()), "ci95": np.quantile(boot[:, 0], [.025, .975]).tolist()},
        "paired_5": {"mean": float(dif5.mean()), "ci95": np.quantile(boot[:, 1], [.025, .975]).tolist()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = json.loads(args.rows.read_text(encoding="utf-8"))
    rng = np.random.default_rng(20260925)
    groups = {
        "all": rows,
        "original_failure": [x for x in rows if x["original_mask_iou"] < .75],
        "original_success": [x for x in rows if x["original_mask_iou"] >= .75],
        "failure_good_box": [x for x in rows if x["original_mask_iou"] < .75 and x["box_iou"] >= .75],
    }
    summary = {name: one_group(sub, rng) for name, sub in groups.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
