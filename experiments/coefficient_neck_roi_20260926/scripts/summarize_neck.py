"""Image-clustered validation summary for 7J-N neck descriptors."""
import argparse
import json
from pathlib import Path

import numpy as np


ARMS = ("h_only", "true_local", "wrong_instance", "wrong_image")
METRICS = ("effect_cos", "coefficient_cos", "radius_log_abs", "image_iou_delta",
           "auc_delta", "grid_iou_delta", "coverage_delta", "fpr_delta", "mask75_net")


def values(row, arm):
    result = row["arms"][arm]
    output = {name: result[name] for name in ("effect_cos", "coefficient_cos", "radius_log_abs")}
    output["image_iou_delta"] = result["image_iou"] - row["original_image_iou"]
    output["mask75_net"] = int(result["image_iou"] >= .75) - int(row["original_image_iou"] >= .75)
    for name in ("auc", "iou", "coverage", "fpr"):
        before, after = row["original_grid"][name], result["grid"][name]
        key = "grid_iou_delta" if name == "iou" else f"{name}_delta"
        output[key] = after - before if before is not None and after is not None else None
    return output


def summarize(rows, arm, seed, repetitions=2000):
    images = sorted({r["image_id"] for r in rows})
    indices = {iid: j for j, iid in enumerate(images)}
    samples = np.random.default_rng(seed).integers(0, len(images), size=(repetitions, len(images)))
    result = dict(images=len(images), instances=len(rows), metrics={})
    for metric in METRICS:
        sums = np.zeros(len(images))
        counts = np.zeros(len(images))
        for row in rows:
            value = values(row, arm)[metric]
            if value is not None:
                sums[indices[row["image_id"]]] += value
                counts[indices[row["image_id"]]] += 1
        estimates = sums[samples].sum(1) / counts[samples].sum(1)
        result["metrics"][metric] = dict(mean=float(sums.sum() / counts.sum()),
                                          ci95=np.quantile(estimates, [.025, .975]).tolist())
    result["repairs"] = sum(r["original_image_iou"] < .75 <= r["arms"][arm]["image_iou"] for r in rows)
    result["damages"] = sum(r["arms"][arm]["image_iou"] < .75 <= r["original_image_iou"] for r in rows)
    return result


def contrast(rows, other, seed, repetitions=3000):
    images = sorted({r["image_id"] for r in rows})
    indices = {iid: j for j, iid in enumerate(images)}
    samples = np.random.default_rng(seed).integers(0, len(images), size=(repetitions, len(images)))
    result = {}
    for metric in ("effect_cos", "coefficient_cos", "image_iou_delta", "auc_delta", "mask75_net"):
        sums = np.zeros(len(images))
        counts = np.zeros(len(images))
        for row in rows:
            a, b = values(row, "true_local")[metric], values(row, other)[metric]
            if a is not None and b is not None:
                sums[indices[row["image_id"]]] += a - b
                counts[indices[row["image_id"]]] += 1
        estimates = sums[samples].sum(1) / counts[samples].sum(1)
        result[metric] = dict(mean=float(sums.sum() / counts.sum()),
                              ci95=np.quantile(estimates, [.025, .975]).tolist())
    return result


def main(args):
    rows = json.loads((args.evaluation / "ROWS.json").read_text())
    groups = dict(all=rows,
                  original_failure=[r for r in rows if r["original_image_iou"] < .75],
                  original_success=[r for r in rows if r["original_image_iou"] >= .75],
                  failure_good_box=[r for r in rows if r["original_image_iou"] < .75 and r["box_iou"] >= .75])
    result = dict(evaluation_run=args.evaluation.name, groups={}, true_local_contrasts={})
    for gi, (name, group) in enumerate(groups.items()):
        result["groups"][name] = {arm: summarize(group, arm, 20260925 + 10 * gi + k)
                                  for k, arm in enumerate(ARMS)}
        result["true_local_contrasts"][name] = {
            arm: contrast(group, arm, 20300925 + 10 * gi + k)
            for k, arm in enumerate(("h_only", "wrong_instance", "wrong_image"))}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SUMMARY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({arm: {key: result["groups"]["all"][arm]["metrics"][key]["mean"]
                            for key in METRICS} for arm in ARMS}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
