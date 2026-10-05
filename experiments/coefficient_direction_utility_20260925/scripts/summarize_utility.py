"""Image-clustered uncertainty for the GT-assisted 7E.1 utility curves."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


ARMS = ("h_only", "true_local")
RHO = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
METRICS = ("image_iou_delta", "grid_iou_delta", "auc_delta", "coverage_delta", "fpr_delta", "mask75_net", "effect_cos")


def one(row, condition):
    before = row["conditions"]["baseline"]
    after = row["conditions"][condition]
    value = dict(image_iou_delta=after["image_iou"] - before["image_iou"],
                 mask75_net=int(after["image_iou"] >= 0.75) - int(before["image_iou"] >= 0.75),
                 effect_cos=after["effect_cos"])
    for name in ("iou", "auc", "coverage", "fpr"):
        a, b = after["grid"][name], before["grid"][name]
        value[("grid_iou" if name == "iou" else name) + "_delta"] = a - b if a is not None and b is not None else None
    return value


def group_summary(rows, condition, seed, bootstrap_count):
    data = [(row["image_id"], one(row, condition)) for row in rows]
    image_ids = sorted({iid for iid, _ in data})
    mapped = {iid: k for k, iid in enumerate(image_ids)}
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(image_ids), size=(bootstrap_count, len(image_ids)))
    output = dict(images=len(image_ids), instances=len(rows), metrics={})
    for metric in METRICS:
        sums = np.zeros(len(image_ids), dtype=np.float64)
        counts = np.zeros(len(image_ids), dtype=np.int64)
        for iid, values in data:
            v = values[metric]
            if v is not None:
                sums[mapped[iid]] += v
                counts[mapped[iid]] += 1
        center = float(sums.sum() / counts.sum()) if counts.sum() else None
        boot_denominator = counts[samples].sum(axis=1)
        valid = boot_denominator > 0
        replicates = sums[samples].sum(axis=1)[valid] / boot_denominator[valid]
        output["metrics"][metric] = dict(mean=center,
                                          ci95=np.quantile(replicates, [.025, .975]).tolist() if len(replicates) else None,
                                          valid_instances=int(counts.sum()))
    output["repairs"] = sum(row["conditions"]["baseline"]["image_iou"] < .75 <=
                            row["conditions"][condition]["image_iou"] for row in rows)
    output["damages"] = sum(row["conditions"][condition]["image_iou"] < .75 <=
                            row["conditions"]["baseline"]["image_iou"] for row in rows)
    return output


def main(args):
    rows = json.loads((args.evaluation / "ROWS.json").read_text())
    groups = dict(all=rows,
                  original_failure=[r for r in rows if r["conditions"]["baseline"]["image_iou"] < .75],
                  original_success=[r for r in rows if r["conditions"]["baseline"]["image_iou"] >= .75],
                  failure_good_box=[r for r in rows if r["conditions"]["baseline"]["image_iou"] < .75
                                    and r["box_iou"] >= .75])
    output = dict(source_run=args.evaluation.name, bootstrap_count=args.bootstrap, groups={}, thresholds={})
    for gi, (name, subset) in enumerate(groups.items()):
        output["groups"][name] = {}
        for ai, arm in enumerate(ARMS):
            conditions = (f"{arm}:raw", f"{arm}:oracle_radius") + tuple(f"{arm}:rho_{rho:g}" for rho in RHO)
            output["groups"][name][arm] = {
                condition.split(":", 1)[1]: group_summary(subset, condition,
                                                             20260925 + 100 * gi + 10 * ai + k,
                                                             args.bootstrap)
                for k, condition in enumerate(conditions)
            }
    for arm in ARMS:
        conditions = output["groups"]["all"][arm]
        output["thresholds"][arm] = {}
        for metric in ("image_iou_delta", "auc_delta", "mask75_net"):
            positives = [rho for rho in RHO if conditions[f"rho_{rho:g}"]["metrics"][metric]["ci95"][0] > 0]
            output["thresholds"][arm][metric] = min(positives) if positives else None
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SUMMARY.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    compact = {arm: {key: {metric: round(values["metrics"][metric]["mean"], 5)
                           for metric in ("effect_cos", "image_iou_delta", "auc_delta", "mask75_net")}
                     for key, values in output["groups"]["all"][arm].items()} for arm in ARMS}
    print(json.dumps(dict(all=compact, thresholds=output["thresholds"]), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    main(parser.parse_args())
