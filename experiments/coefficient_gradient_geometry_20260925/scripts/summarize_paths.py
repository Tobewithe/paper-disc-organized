"""Summarize direct-to-oracle and LBFGS paths without mixing their supports."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


METRICS = ("loss", "positive_loss", "negative_loss", "iou", "coverage", "fpr", "auc")


def mean(values):
    valid = [value for value in values if value is not None]
    return statistics.fmean(valid) if valid else None


def percentile(values, fraction):
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    lo = int(at)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (hi - at) + ordered[hi] * (at - lo) if hi > lo else ordered[lo]


def paired(records, xkey, picks):
    grouped = defaultdict(dict)
    for row in records:
        grouped[row["annotation_id"]][row[xkey]] = row
    out = {}
    for point in picks:
        pairs = [(items[picks[0]], items[point]) for items in grouped.values()
                 if picks[0] in items and point in items]
        if not pairs:
            continue
        answer = {f"{key}_delta": mean([b[key] - a[key] if a[key] is not None and b[key] is not None else None
                                         for a, b in pairs]) for key in METRICS}
        answer.update(instances=len(pairs), images=len({a["image_id"] for a, _ in pairs}),
                      both_training_terms_down=mean([b["positive_loss"] < a["positive_loss"] and
                                                     b["negative_loss"] < a["negative_loss"] for a, b in pairs]),
                      coverage_up_fpr_down=mean([b["coverage"] >= a["coverage"] and
                                                 b["fpr"] < a["fpr"] for a, b in pairs
                                                 if None not in (a["coverage"], b["coverage"], a["fpr"], b["fpr"])]),
                      iou_up=mean([b["iou"] > a["iou"] for a, b in pairs
                                   if None not in (a["iou"], b["iou"])]))
        out[str(point)] = answer
    return out


def bootstrap_pair_deltas(records, xkey, start, end, repetitions=3000):
    grouped = defaultdict(dict)
    for row in records:
        grouped[row["annotation_id"]][row[xkey]] = row
    by_image = defaultdict(list)
    for items in grouped.values():
        if start in items and end in items:
            a, b = items[start], items[end]
            by_image[a["image_id"]].append((a, b))
    ids = list(by_image)
    rng = random.Random(20260925)
    boot = defaultdict(list)
    for _ in range(repetitions):
        pairs = [pair for iid in rng.choices(ids, k=len(ids)) for pair in by_image[iid]]
        for key in ("iou", "coverage", "fpr", "auc", "positive_loss", "negative_loss"):
            value = mean([b[key] - a[key] if None not in (a[key], b[key]) else None for a, b in pairs])
            if value is not None:
                boot[f"{key}_delta"].append(value)
    return {key: [percentile(values, .025), percentile(values, .975)] for key, values in boot.items()}


def trajectory_summary(rows):
    by_instance = defaultdict(list)
    for row in rows:
        by_instance[row["annotation_id"]].append(row)
    final = []
    for group in by_instance.values():
        group.sort(key=lambda row: row["step"])
        final.extend((group[0], group[-1]))
    result = paired(final, "step", (0, max(row["step"] for row in rows)))
    # Some optimizers stop before max steps, so include each instance's actual end.
    endpoints = []
    for group in by_instance.values():
        group.sort(key=lambda row: row["step"])
        endpoints.extend((dict(group[0], phase="start"), dict(group[-1], phase="end")))
    result["final_actual"] = paired(endpoints, "phase", ("start", "end"))["end"]
    result["final_actual_ci"] = bootstrap_pair_deltas(endpoints, "phase", "start", "end")
    result["steps"] = paired(rows, "step", (0, 1, 2, 5, 10, 20, 40, 60))
    result["step_1_ci"] = bootstrap_pair_deltas(rows, "step", 0, 1)
    result["mean_final_step"] = mean([group[-1]["step"] for group in by_instance.values()])
    return result


def bootstrap_difference(rows, repetitions=3000):
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    ids = list(by_image)
    rng = random.Random(20260925)
    estimates = []
    for _ in range(repetitions):
        drawn = [row for iid in rng.choices(ids, k=len(ids)) for row in by_image[iid]]
        fail = [row["pos_neg_cos"] for row in drawn if row["initial_iou"] < .75 and row["pos_neg_cos"] is not None]
        success = [row["pos_neg_cos"] for row in drawn if row["initial_iou"] >= .75 and row["pos_neg_cos"] is not None]
        if fail and success:
            estimates.append(mean(fail) - mean(success))
    return dict(failure_minus_success=mean([r["pos_neg_cos"] for r in rows if r["initial_iou"] < .75]) -
                mean([r["pos_neg_cos"] for r in rows if r["initial_iou"] >= .75]),
                image_cluster_ci=[percentile(estimates, .025), percentile(estimates, .975)])


def main(args):
    run = args.run
    gradients = json.loads((run / "GRADIENTS.json").read_text())
    interpolation = json.loads((run / "INTERPOLATION.json").read_text())
    trajectories = json.loads((run / "TRAJECTORIES.json").read_text())
    result = dict(gradient_difference=bootstrap_difference(gradients),
                  interpolation=paired(interpolation, "alpha", (0, .001, .005, .01, .02, .05, .1, .2, .4, .6, .8, 1)),
                  trajectory=trajectory_summary(trajectories))
    result["interpolation_0_001_ci"] = bootstrap_pair_deltas(interpolation, "alpha", 0, .001)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "PATH_SUMMARY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"gradient_difference": result["gradient_difference"],
                      "interpolation_0_001": result["interpolation"]["0.001"],
                      "trajectory_step_1": result["trajectory"]["steps"]["1"],
                      "trajectory_final": result["trajectory"]["final_actual"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
