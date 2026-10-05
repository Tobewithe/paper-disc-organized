"""Paired image-clustered limits of the locked 7L result."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from train_benefit_gate import features


def main(a):
    locked = json.loads((a.evaluation / "RESULTS.json").read_text())
    development = json.loads((a.gate / "DEVELOPMENT.json").read_text())
    selected = development["selected_by_development_top10"]
    assert selected == locked["selected_by_development"] == "X3_huber"
    rows = []
    for part in sorted((a.test_scores / "shards").glob("PART_*.pt")):
        rows.extend(torch.load(part, weights_only=True, map_location="cpu"))
    assert len(rows) == locked["candidates"] == 14911
    four = [features(r) for r in rows]
    x = [np.asarray([entry[j] for entry in four], np.float32) for j in range(4)]
    benefit = np.asarray([float(r["benefit_iou"]) for r in rows])
    before = np.asarray([float(r["original_iou"]) for r in rows])
    after = np.asarray([float(r["corrected_iou"]) for r in rows])
    ids = np.asarray([int(r["image_id"]) for r in rows])
    diff75 = (after >= .75).astype(float) - (before >= .75).astype(float)
    models = ("X1_huber", "X2_huber", "X3_huber", "X4_huber")
    scores = {name:joblib.load(a.gate / f"{name}.joblib").predict(x[int(name[1])-1])
              for name in models}
    threshold = development["groups"]["X3"]["huber"]["threshold_top10"]
    gate = scores[selected] > threshold
    assert abs(gate.mean() - locked["models"][selected]["fixed_dev_threshold"]["coverage"]) < 1e-12
    unique, inverse = np.unique(ids, return_inverse=True)
    groups = [np.where(inverse == j)[0] for j in range(len(unique))]
    rng = np.random.default_rng(20260927)
    estimates = []
    for _ in range(a.bootstrap):
        sample = np.concatenate([groups[j] for j in rng.integers(len(unique), size=len(unique))])
        n = max(1, int(np.ceil(.1 * len(sample))))
        means = {}
        for name in models:
            take = sample[np.argsort(scores[name][sample], kind="stable")[-n:]]
            means[name] = benefit[take].mean()
        estimates.append(dict(
            x3_minus_x1=means["X3_huber"]-means["X1_huber"],
            x3_minus_x2=means["X3_huber"]-means["X2_huber"],
            x3_minus_x4=means["X3_huber"]-means["X4_huber"],
            gated_population_benefit=(benefit[sample]*gate[sample]).mean(),
            gated_minus_all_population_benefit=(benefit[sample]*(gate[sample]-1)).mean(),
            gated_population_mask75=(diff75[sample]*gate[sample]).mean(),
            gated_minus_all_population_mask75=(diff75[sample]*(gate[sample]-1)).mean()))
    keys = estimates[0]
    ci = {key:np.quantile([row[key] for row in estimates], [.025,.975]).tolist()
          for key in keys}
    point_means = {}
    n = max(1, int(np.ceil(.1*len(rows))))
    for name in models:
        take = np.argsort(scores[name], kind="stable")[-n:]
        point_means[name] = float(benefit[take].mean())
    results = dict(selected=selected, test_images=len(unique), candidates=len(rows),
        bootstrap=a.bootstrap, top10_mean_benefit=point_means,
        top10_x3_minus_x1=point_means["X3_huber"]-point_means["X1_huber"],
        top10_x3_minus_x2=point_means["X3_huber"]-point_means["X2_huber"],
        top10_x3_minus_x4=point_means["X3_huber"]-point_means["X4_huber"],
        fixed_dev_threshold=float(threshold), fixed_gate_coverage=float(gate.mean()),
        gated_population_benefit=float((benefit*gate).mean()),
        all_on_population_benefit=float(benefit.mean()),
        gated_minus_all_population_benefit=float((benefit*(gate-1)).mean()),
        gated_population_mask75=float((diff75*gate).mean()),
        all_on_population_mask75=float(diff75.mean()),
        gated_minus_all_population_mask75=float((diff75*(gate-1)).mean()),
        intervals95=ci, note="X1–X4 are prespecified feature groups; X3 was selected on inner train2017 development only. These paired comparisons are secondary; final method AP is untested.")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "ANALYSIS.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (a.out / "COMPLETE.json").write_text(json.dumps(dict(images=len(unique),
        candidates=len(rows), bootstrap=a.bootstrap), indent=2), encoding="utf-8")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("gate", "test_scores", "evaluation", "out"):
        p.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    main(p.parse_args())
