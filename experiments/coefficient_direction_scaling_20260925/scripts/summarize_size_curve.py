"""Image-clustered 7F-A validation learning curve and paired size contrasts."""
import argparse
import json
from pathlib import Path

import numpy as np


SIZES = (800, 1600, 3200, 6400, 10000)
METRICS = ("effect_cos", "coefficient_cos", "radius_log_abs", "image_iou_delta",
           "auc_delta", "mask75_net", "fraction_cos_ge_0_5", "fraction_cos_ge_0_65")


def values(row, size):
    result = row["sizes"][str(size)]
    cos = result["effect_cos"]
    return dict(effect_cos=cos, coefficient_cos=result["coefficient_cos"],
                radius_log_abs=result["radius_log_abs"],
                image_iou_delta=result["image_iou"] - row["original_image_iou"],
                auc_delta=(result["auc"] - row["original_auc"])
                if result["auc"] is not None and row["original_auc"] is not None else None,
                mask75_net=int(result["image_iou"] >= .75) - int(row["original_image_iou"] >= .75),
                fraction_cos_ge_0_5=int(cos >= .5), fraction_cos_ge_0_65=int(cos >= .65))


def summarize(rows, size, seed, draws=2000):
    images = sorted({r["image_id"] for r in rows})
    index = {iid: j for j, iid in enumerate(images)}
    sample = np.random.default_rng(seed).integers(0, len(images), size=(draws, len(images)))
    output = dict(images=len(images), instances=len(rows), metrics={})
    for metric in METRICS:
        sums = np.zeros(len(images))
        counts = np.zeros(len(images))
        for row in rows:
            v = values(row, size)[metric]
            if v is not None:
                sums[index[row["image_id"]]] += v
                counts[index[row["image_id"]]] += 1
        estimates = sums[sample].sum(1) / counts[sample].sum(1)
        output["metrics"][metric] = dict(mean=float(sums.sum() / counts.sum()),
                                          ci95=np.quantile(estimates, [.025, .975]).tolist())
    output["repairs"] = sum(r["original_image_iou"] < .75 <= r["sizes"][str(size)]["image_iou"] for r in rows)
    output["damages"] = sum(r["sizes"][str(size)]["image_iou"] < .75 <= r["original_image_iou"] for r in rows)
    return output


def paired(rows, size, seed, draws=3000):
    images = sorted({r["image_id"] for r in rows})
    index = {iid: j for j, iid in enumerate(images)}
    sample = np.random.default_rng(seed).integers(0, len(images), size=(draws, len(images)))
    output = {}
    for metric in ("effect_cos", "image_iou_delta", "auc_delta", "mask75_net"):
        sums = np.zeros(len(images))
        counts = np.zeros(len(images))
        for row in rows:
            a, b = values(row, size)[metric], values(row, 800)[metric]
            if a is not None and b is not None:
                sums[index[row["image_id"]]] += a - b
                counts[index[row["image_id"]]] += 1
        estimates = sums[sample].sum(1) / counts[sample].sum(1)
        output[metric] = dict(mean=float(sums.sum() / counts.sum()),
                              ci95=np.quantile(estimates, [.025, .975]).tolist())
    return output


def main(args):
    rows = json.loads((args.evaluation / "ROWS.json").read_text())
    groups = dict(all=rows,
                  original_failure=[r for r in rows if r["original_image_iou"] < .75],
                  original_success=[r for r in rows if r["original_image_iou"] >= .75],
                  failure_good_box=[r for r in rows if r["original_image_iou"] < .75 and r["box_iou"] >= .75])
    result = dict(evaluation_run=args.evaluation.name, groups={}, paired_vs_800={})
    for gi, (name, group) in enumerate(groups.items()):
        result["groups"][name] = {str(n): summarize(group, n, 20260925 + 10 * gi + k)
                                  for k, n in enumerate(SIZES)}
        result["paired_vs_800"][name] = {str(n): paired(group, n, 20300925 + 10 * gi + k)
                                           for k, n in enumerate(SIZES[1:])}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SUMMARY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({n: {k: result["groups"]["all"][n]["metrics"][k]["mean"] for k in METRICS}
                      for n in map(str, SIZES)}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
