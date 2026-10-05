"""Image-clustered uncertainty for Experiment 7A's paired instance records."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


def mean(rows, key):
    values = [r[key] for r in rows if key in r and r[key] is not None]
    return sum(values) / len(values) if values else None


def metrics(rows):
    base = mean(rows, "initial_grid_iou")
    threshold = mean(rows, "best_threshold_iou")
    oracle = mean(rows, "oracle_grid_iou")
    bias = mean(rows, "fitted_bias_iou")
    affine = mean(rows, "fitted_affine_iou")
    threshold_gain = threshold - base
    oracle_gain = oracle - base
    boundary_pixel_fraction = sum(r["target_boundary_pixels"] / r["support_pixels"] for r in rows) / len(rows)
    outside_pixel_fraction = sum(r["outside_pixels"] / r["support_pixels"] for r in rows) / len(rows)
    return dict(
        instances=len(rows), images=len({r["image_id"] for r in rows}),
        initial_grid_iou=base, best_threshold_grid_iou=threshold, oracle_grid_iou=oracle,
        initial_roc_auc=mean(rows, "initial_roc_auc"), oracle_roc_auc=mean(rows, "oracle_roc_auc"),
        roc_auc_gain=mean(rows, "oracle_roc_auc") - mean(rows, "initial_roc_auc"),
        fitted_bias_grid_iou=bias, fitted_affine_grid_iou=affine,
        threshold_gain=threshold_gain, oracle_gain=oracle_gain,
        oracle_minus_threshold=oracle - threshold,
        threshold_fraction_of_oracle_gain=threshold_gain / oracle_gain if oracle_gain > 0 else None,
        initial_grid_mask75=sum(r["initial_grid_iou"] >= .75 for r in rows),
        threshold_grid_mask75=sum(r["best_threshold_iou"] >= .75 for r in rows),
        oracle_grid_mask75=sum(r["oracle_grid_iou"] >= .75 for r in rows),
        constant_delta_energy_share=mean(rows, "constant_delta_energy_share"),
        affine_delta_energy_share=mean(rows, "affine_delta_energy_share"),
        oracle_logit_saturation_fraction=mean(rows, "oracle_logit_saturation_fraction"),
        target_boundary_pixel_fraction=boundary_pixel_fraction,
        target_boundary_energy_share=mean(rows, "target_boundary_abs_energy_share"),
        outside_pixel_fraction=outside_pixel_fraction,
        outside_energy_share=mean(rows, "outside_abs_energy_share"),
        box_edge_pixel_fraction=mean(rows, "box_edge_pixel_fraction"),
        box_edge_energy_share=mean(rows, "box_edge_abs_energy_share"),
        target_interior_probability_delta=mean(rows, "target_interior_probability_delta_mean"),
        target_boundary_probability_delta=mean(rows, "target_boundary_probability_delta_mean"),
        neighbor_probability_delta=mean(rows, "neighbor_probability_delta_mean"),
        background_probability_delta=mean(rows, "background_probability_delta_mean"),
    )


def percentile(values, q):
    values = sorted(values)
    at = (len(values) - 1) * q
    lo = int(at)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - at) + values[hi] * (at - lo) if hi > lo else values[lo]


def bootstrap(rows, repetitions, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["image_id"]].append(row)
    image_ids = sorted(groups)
    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(repetitions):
        selected = [row for iid in rng.choices(image_ids, k=len(image_ids)) for row in groups[iid]]
        for key, value in metrics(selected).items():
            if value is not None:
                samples[key].append(value)
    return {key: [percentile(values, .025), percentile(values, .975)]
            for key, values in samples.items() if key not in ("instances", "images")}


def main(a):
    rows = json.loads(a.rows.read_text(encoding="utf-8"))
    groups = dict(all=rows,
        original_mask75_failed=[r for r in rows if r["initial_iou"] < .75],
        oracle_repaired_original_mask75=[r for r in rows if r["initial_iou"] < .75 <= r["oracle_iou"]],
        good_box_original_mask75_failed=[r for r in rows if r["box_iou"] >= .75 and r["initial_iou"] < .75])
    result = {name: dict(metrics=metrics(subset), image_cluster_bootstrap_95ci=bootstrap(
        subset, a.bootstrap, a.seed + index)) for index, (name, subset) in enumerate(groups.items())}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "SUMMARY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({name: value["metrics"] for name, value in result.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260924)
    main(parser.parse_args())
