"""Paired image-cluster bootstrap for Experiment 7B validation rows."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


FIELDS = ("iou", "auc", "fp_inside_box", "fp_rate", "target_coverage")


def summarize(rows, indices, reference="original"):
    original = rows[reference]
    output = {}
    for arm, arm_rows in rows.items():
        if arm == reference or (reference != "original" and arm == "original"):
            continue
        differences = {field: np.array([
            (arm_rows[i][field] - original[i][field]) if arm_rows[i][field] is not None
            and original[i][field] is not None else np.nan for i in indices], dtype=float)
            for field in FIELDS}
        metrics = {field: float(np.nanmean(values)) for field, values in differences.items()}
        metrics["mask75_count_delta"] = int(sum(arm_rows[i]["iou"] >= .75 for i in indices)
                                           - sum(original[i]["iou"] >= .75 for i in indices))
        metrics["mask75_rate_delta"] = metrics["mask75_count_delta"] / len(indices)
        metrics["n"] = len(indices)
        output[arm] = metrics
    return output


def bootstrap(rows, indices, repetitions, seed, reference="original"):
    original = rows[reference]
    grouped = defaultdict(list)
    for i in indices:
        grouped[original[i]["image_id"]].append(i)
    image_ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    samples = {arm: defaultdict(list) for arm in rows
               if arm != reference and (reference == "original" or arm != "original")}
    for _ in range(repetitions):
        picked = rng.choice(image_ids, size=len(image_ids), replace=True)
        selected = [i for iid in picked for i in grouped[iid]]
        for arm, values in summarize(rows, selected, reference).items():
            for field, value in values.items():
                if field != "n":
                    samples[arm][field].append(value)
    return {arm: {field: list(np.quantile(values, [.025, .975]))
                  for field, values in metrics.items()} for arm, metrics in samples.items()}


def main(a):
    report = {}
    for checkpoint in ("LAST", "SELECTED"):
        rows = json.loads((a.run / f"{checkpoint}_VAL_ROWS.json").read_text())
        baseline = rows["original"]
        keys = [(r["image_id"], r["annotation_id"]) for r in baseline]
        assert len(keys) == len(set(keys))
        for arm, arm_rows in rows.items():
            assert len(arm_rows) == len(baseline)
            assert [(r["image_id"], r["annotation_id"]) for r in arm_rows] == keys
        groups = dict(all=list(range(len(baseline))),
                      original_mask75_failed=[i for i, r in enumerate(baseline) if r["baseline_iou"] < .75],
                      good_box_original_mask75_failed=[i for i, r in enumerate(baseline)
                                                       if r["baseline_iou"] < .75 and r["box_iou"] >= .75])
        report[checkpoint.lower()] = {
            name: {
                "versus_original": dict(paired_delta=summarize(rows, indices),
                    image_cluster_95ci=bootstrap(rows, indices, a.bootstrap, a.seed + j)),
                "versus_official_bce": dict(paired_delta=summarize(rows, indices, "official_bce"),
                    image_cluster_95ci=bootstrap(rows, indices, a.bootstrap, a.seed + j, "official_bce")),
                "versus_negative_weighted": dict(paired_delta=summarize(rows, indices, "negative_weighted"),
                    image_cluster_95ci=bootstrap(rows, indices, a.bootstrap, a.seed + j, "negative_weighted")),
            } for j, (name, indices) in enumerate(groups.items())}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "SUMMARY.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({stage: {group: value["versus_official_bce"]["paired_delta"]
                             for group, value in groups.items()}
                      for stage, groups in report.items()}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=3000)
    p.add_argument("--seed", type=int, default=20260924)
    main(p.parse_args())
