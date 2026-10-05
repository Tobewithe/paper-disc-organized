"""Paired image-clustered summary for fixed-candidate 7E validation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


ARMS = ("h_only", "true_local", "wrong_instance", "wrong_image")
KEYS = ("coefficient_cos", "effect_cos", "radius_log_abs", "image_iou_delta",
        "grid_iou_delta", "coverage_delta", "fpr_delta", "auc_delta", "mask75_net")


def mean(values):
    valid = [x for x in values if x is not None]
    return statistics.fmean(valid) if valid else None


def percentile(values, q):
    ordered = sorted(values)
    x = (len(ordered) - 1) * q
    lo = int(x)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (hi - x) + ordered[hi] * (x - lo) if hi > lo else ordered[lo]


def paired_metrics(row, arm):
    result = row["arms"][arm]
    grid, baseline = result["grid"], row["original_grid"]
    return dict(coefficient_cos=result["coefficient_cos"], effect_cos=result["effect_cos"],
                radius_log_abs=result["radius_log_abs"],
                image_iou_delta=result["image_iou"] - row["original_image_iou"],
                grid_iou_delta=grid["iou"] - baseline["iou"],
                coverage_delta=grid["coverage"] - baseline["coverage"]
                if None not in (grid["coverage"], baseline["coverage"]) else None,
                fpr_delta=grid["fpr"] - baseline["fpr"]
                if None not in (grid["fpr"], baseline["fpr"]) else None,
                auc_delta=grid["auc"] - baseline["auc"]
                if None not in (grid["auc"], baseline["auc"]) else None,
                mask75_net=int(result["image_iou"] >= .75) - int(row["original_image_iou"] >= .75))


def point(rows, arm):
    values = [paired_metrics(row, arm) for row in rows]
    result = {key: mean([value[key] for value in values]) for key in KEYS}
    result.update(instances=len(rows), images=len({row["image_id"] for row in rows}),
                  repairs=sum(row["original_image_iou"] < .75 <= row["arms"][arm]["image_iou"] for row in rows),
                  damages=sum(row["arms"][arm]["image_iou"] < .75 <= row["original_image_iou"] for row in rows))
    return result


def bootstrap(rows, arm, seed, repetitions=2000):
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    ids = list(by_image)
    rng = random.Random(seed)
    estimates = defaultdict(list)
    for _ in range(repetitions):
        draw = [row for iid in rng.choices(ids, k=len(ids)) for row in by_image[iid]]
        for key, value in point(draw, arm).items():
            if key not in ("instances", "images", "repairs", "damages") and value is not None:
                estimates[key].append(value)
    return {key: [percentile(values, .025), percentile(values, .975)] for key, values in estimates.items()}


def arm_difference(rows, first, second, seed, repetitions=3000):
    selected = ("coefficient_cos", "effect_cos", "image_iou_delta", "auc_delta", "mask75_net")
    by_image = defaultdict(lambda: defaultdict(list))
    for row in rows:
        a, b = paired_metrics(row, first), paired_metrics(row, second)
        for key in selected:
            if None not in (a[key], b[key]):
                by_image[row["image_id"]][key].append(a[key] - b[key])
    ids = list(by_image)
    rng = random.Random(seed)
    result = {}
    for key in selected:
        totals = {iid: (sum(by_image[iid][key]), len(by_image[iid][key])) for iid in ids}
        sum_all, count_all = zip(*totals.values())
        samples = []
        for _ in range(repetitions):
            draw = rng.choices(ids, k=len(ids))
            numerator = sum(totals[iid][0] for iid in draw)
            denominator = sum(totals[iid][1] for iid in draw)
            if denominator:
                samples.append(numerator / denominator)
        result[key] = dict(point=sum(sum_all) / sum(count_all),
                           image_cluster_ci=[percentile(samples, .025), percentile(samples, .975)])
    return result


def hybrid_summary(rows, arm, name):
    result = {}
    for metric in ("iou", "coverage", "fpr", "auc"):
        result[f"{metric}_delta"] = mean([
            row["arms"][arm]["hybrids"][name][metric] - row["original_grid"][metric]
            if None not in (row["arms"][arm]["hybrids"][name][metric], row["original_grid"][metric]) else None
            for row in rows])
    return result


def main(args):
    rows = json.loads((args.evaluation / "ROWS.json").read_text())
    groups = dict(all=rows, original_failure=[row for row in rows if row["original_image_iou"] < .75],
                  original_success=[row for row in rows if row["original_image_iou"] >= .75],
                  failure_good_box=[row for row in rows if row["original_image_iou"] < .75 and row["box_iou"] >= .75])
    summary = dict(evaluation_run_id=args.evaluation.name, groups={}, contrasts={}, hybrid_diagnostics={})
    for group_number, (group_name, subset) in enumerate(groups.items()):
        summary["groups"][group_name] = {arm: dict(point=point(subset, arm),
                        image_cluster_ci=bootstrap(subset, arm, 20260925 + group_number * 10 + j))
                        for j, arm in enumerate(ARMS)}
        summary["contrasts"][group_name] = {
            other: arm_difference(subset, "true_local", other, 20260925 + group_number * 10 + j)
            for j, other in enumerate(("h_only", "wrong_instance", "wrong_image"))}
        summary["hybrid_diagnostics"][group_name] = {
            arm: {name: hybrid_summary(subset, arm, name)
                  for name in ("direction_with_teacher_radius", "teacher_direction_with_radius")}
            for arm in ARMS}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({name: {arm: {key: value["point"][key] for key in
                                  ("instances", "effect_cos", "coefficient_cos", "radius_log_abs",
                                   "image_iou_delta", "mask75_net", "auc_delta")}
                            for arm, value in grouped.items()}
                      for name, grouped in summary["groups"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
