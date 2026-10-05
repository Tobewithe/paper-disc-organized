"""Image-clustered summary of the predeclared 7D regularization curve."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


LAMBDAS = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003)


def mean(values):
    valid = [value for value in values if value is not None]
    return statistics.fmean(valid) if valid else None


def percentile(values, fraction):
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    lo = int(at)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (hi - at) + ordered[hi] * (at - lo) if hi > lo else ordered[lo]


def row_metrics(rows):
    result = dict(instances=len(rows), images=len({r["image_id"] for r in rows}),
                  norm_median=statistics.median(r["delta_norm"] for r in rows),
                  norm_mean=mean([r["delta_norm"] for r in rows]),
                  stationary_max=max(r["stationary_norm"] for r in rows),
                  old_coefficient_cos=mean([r["old_coefficient_cos"] for r in rows]),
                  old_response_cos=mean([r["old_response_cos"] for r in rows]),
                  mask75_gain=mean([int(r["new_iou"] >= .75) - int(r["initial_iou"] >= .75) for r in rows]),
                  simultaneous_benefit=mean([r["new_coverage"] >= r["initial_coverage"] and
                                             r["new_fpr"] < r["initial_fpr"] for r in rows
                                             if None not in (r["new_coverage"], r["initial_coverage"],
                                                             r["new_fpr"], r["initial_fpr"])]))
    for metric in ("iou", "coverage", "fpr", "auc", "loss", "positive_loss", "negative_loss"):
        result[f"{metric}_delta"] = mean([r[f"new_{metric}"] - r[f"initial_{metric}"]
                                          if None not in (r[f"new_{metric}"], r[f"initial_{metric}"]) else None
                                          for r in rows])
    return result


def group_summary(rows, seed):
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    image_ids = list(by_image)
    point = row_metrics(rows)
    rng = random.Random(seed)
    estimates = defaultdict(list)
    for _ in range(2000):
        drawn = [row for iid in rng.choices(image_ids, k=len(image_ids)) for row in by_image[iid]]
        for key, value in row_metrics(drawn).items():
            if key not in ("instances", "images", "stationary_max") and value is not None:
                estimates[key].append(value)
    ci = {key: [percentile(v, .025), percentile(v, .975)] for key, v in estimates.items()}
    return dict(point=point, image_cluster_ci=ci)


def stability_summary(rows):
    if not rows:
        return None
    return dict(instances=len(rows), images=len({r["image_id"] for r in rows}),
                alternate_relative_distance_mean=mean([r["alternate_relative_distance"] for r in rows]),
                alternate_relative_distance_max=max(r["alternate_relative_distance"] for r in rows),
                alternate_objective_gap_abs_max=max(abs(r["alternate_objective_gap"]) for r in rows),
                alternate_stationary_max=max(r["alternate_stationary_norm"] for r in rows),
                subsample_full_cos_mean=mean([v for r in rows for v in r["sample_direction_cos"]]),
                subsample_pair_cos_mean=mean([r["sample_pair_cos"] for r in rows]),
                subsample_pair_cos_median=statistics.median(r["sample_pair_cos"] for r in rows),
                subsample_pair_cos_min=min(r["sample_pair_cos"] for r in rows),
                subsample_relative_distance_mean=mean([v for r in rows for v in r["sample_relative_distance"]]),
                subsample_stationary_max=max(v for r in rows for v in r["sample_stationary_norm"]))


def main(args):
    rows = json.loads((args.source / "ROWS.json").read_text())
    checks = json.loads((args.source / "STABILITY.json").read_text())
    ids = {(r["image_id"], r["annotation_id"]) for r in rows}
    assert len(rows) == len(ids) * len(LAMBDAS)
    out = dict(source_run_id=args.source.name, sample=dict(instances=len(ids), images=len({iid for iid, _ in ids})),
               penalty_curve={}, stability=stability_summary(checks))
    for j, penalty in enumerate(LAMBDAS):
        subset = [r for r in rows if r["penalty"] == penalty]
        groups = dict(all=subset,
                      original_failure=[r for r in subset if r["original_mask_iou"] < .75],
                      original_success=[r for r in subset if r["original_mask_iou"] >= .75],
                      failure_good_box=[r for r in subset if r["original_mask_iou"] < .75 and r["box_iou"] >= .75])
        out["penalty_curve"][str(penalty)] = {name: group_summary(group, 20260925 + j * 7 + k)
                                              for k, (name, group) in enumerate(groups.items())}
    by_instance = defaultdict(dict)
    for row in rows:
        by_instance[(row["image_id"], row["annotation_id"])][row["penalty"]] = row["delta_norm"]
    out["distance_monotonic_fraction"] = mean([
        all(values[a] <= values[b] + 1e-5 for a, b in zip(LAMBDAS[:-1], LAMBDAS[1:]))
        for values in by_instance.values()])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SUMMARY.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({penalty: {group: {key: value["point"][key] for key in
                                       ("instances", "norm_median", "iou_delta", "coverage_delta", "fpr_delta", "auc_delta")}
                                 for group, value in groups.items() if group in ("all", "original_failure")}
                      for penalty, groups in out["penalty_curve"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
