"""One-pass locked evaluation of 7L benefit predictors on independent COCO val."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (average_precision_score, brier_score_loss, r2_score,
                             roc_auc_score)
import torch

from train_benefit_gate import ece, features, top_summary


def load_rows(path):
    complete = json.loads((path / "COMPLETE.json").read_text())
    assert complete["images"] == 2000
    rows, seen = [], set()
    for file in sorted((path / "shards").glob("PART_*.pt")):
        rows.extend(torch.load(file, weights_only=True, map_location="cpu"))
        meta = json.loads(file.with_suffix(".json").read_text())
        seen.update(int(r["image_id"]) for r in meta["images"])
    assert len(seen) == 2000 and len(rows) == complete["instances"]
    keys = [(r["image_id"], r["annotation_id"], r["original_raw_id"]) for r in rows]
    assert len(keys) == len(set(keys))
    return rows, seen


def cluster_ci(scores, benefit, before, after, ids, fraction, repeats=2000, seed=20260927):
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(ids, return_inverse=True)
    grouped = [np.where(inverse == j)[0] for j in range(len(unique))]
    values = np.empty((repeats, 2), dtype=float)
    for k in range(repeats):
        sample = np.concatenate([grouped[j] for j in rng.integers(0, len(unique), len(unique))])
        n = max(1, int(np.ceil(fraction * len(sample))))
        picked = sample[np.argsort(scores[sample], kind="stable")[-n:]]
        values[k, 0] = benefit[picked].mean()
        values[k, 1] = ((after[picked] >= .75).astype(float) -
                         (before[picked] >= .75).astype(float)).mean()
    return dict(mean_benefit_95ci=np.quantile(values[:, 0], [.025, .975]).tolist(),
        mask75_net_95ci=np.quantile(values[:, 1], [.025, .975]).tolist(),
        image_cluster_repeats=repeats)


def main(a):
    gate = json.loads((a.gate / "COMPLETE.json").read_text())
    development = json.loads((a.gate / "DEVELOPMENT.json").read_text())
    selected_name = gate["selected"]
    assert selected_name == development["selected_by_development_top10"]
    manifest = json.loads(a.manifest.read_text())
    rows, images = load_rows(a.test_scores)
    assert images == set(map(int, manifest["independent_test_images"]))
    xlists = [[], [], [], []]
    for row in rows:
        for j, value in enumerate(features(row)):
            xlists[j].append(value)
    x = [np.asarray(values, np.float32) for values in xlists]
    benefit = np.asarray([float(r["benefit_iou"]) for r in rows])
    original = np.asarray([float(r["original_iou"]) for r in rows])
    corrected = np.asarray([float(r["corrected_iou"]) for r in rows])
    ids = np.asarray([int(r["image_id"]) for r in rows])
    label = benefit > .01
    damage = benefit < -.01
    auc_benefit = np.asarray([float(r["benefit_auc"]) for r in rows])
    result = dict(test_images=len(images), candidates=len(rows),
        prevalence=float(label.mean()), all_on_mean_benefit=float(benefit.mean()),
        all_on_mask75_net=float(((corrected>=.75).astype(float)-(original>=.75).astype(float)).mean()),
        all_on_repair=int(np.sum((original<.75) & (corrected>=.75))),
        all_on_damage=int(np.sum((original>=.75) & (corrected<.75))),
        auc_valid=int(np.isfinite(auc_benefit).sum()),
        all_on_mean_auc_benefit=float(np.nanmean(auc_benefit)),
        correction_radius=dict(mean=float(np.mean([float(r["correction"].norm()) for r in rows])),
            median=float(np.median([float(r["correction"].norm()) for r in rows]))),
        selected_by_development=selected_name, models={})
    for j, data in enumerate(x, 1):
        for kind in ("ridge", "huber", "logistic"):
            key = f"X{j}_{kind}"
            model = joblib.load(a.gate / f"{key}.joblib")
            raw = model.predict_proba(data)[:, 1] if kind == "logistic" else model.predict(data)
            item = dict(top10=top_summary(raw, benefit, label, damage),
                        risk_coverage={})
            for q in (.01, .03, .05, .10, .20, .50, 1.):
                subset = top_summary(raw, benefit, label, damage, top=q)
                n = subset["selected"]
                picked = np.argsort(raw, kind="stable")[-n:]
                subset["mask75_net"] = float(((corrected[picked]>=.75).astype(float)-
                                              (original[picked]>=.75).astype(float)).mean())
                item["risk_coverage"][str(q)] = subset
            if kind == "logistic":
                calibrated = joblib.load(a.gate / f"{key}_calibration.joblib").predict(raw)
                item.update(auroc=float(roc_auc_score(label, raw)),
                    auprc=float(average_precision_score(label, raw)),
                    brier=float(brier_score_loss(label, calibrated)),
                    ece=ece(label, calibrated))
            else:
                item.update(spearman=float(spearmanr(benefit, raw).statistic),
                    r2=float(r2_score(benefit, raw)))
            dev_threshold = development["groups"][f"X{j}"][kind]["threshold_top10"]
            selected = raw > dev_threshold
            item["fixed_dev_threshold"] = dict(threshold=float(dev_threshold),
                coverage=float(selected.mean()), selected=int(selected.sum()),
                mean_benefit=float(benefit[selected].mean()) if selected.any() else None,
                mask75_net=float(((corrected[selected]>=.75).astype(float)-
                                  (original[selected]>=.75).astype(float)).mean()) if selected.any() else None)
            if key == selected_name:
                item["top10_cluster_ci"] = cluster_ci(raw, benefit, original, corrected, ids, .1)
                n = item["top10"]["selected"]
                picked = np.argsort(raw, kind="stable")[-n:]
                failed = original[picked] < .75
                item["top10_failure_strata"] = dict(
                    original_failure_fraction=float(failed.mean()),
                    failure_candidates=int(failed.sum()),
                    original_success_candidates=int((~failed).sum()),
                    failure_mean_benefit=float(benefit[picked][failed].mean()) if failed.any() else None,
                    success_mean_benefit=float(benefit[picked][~failed].mean()) if (~failed).any() else None,
                    failure_mask75_repair=int(np.sum((original[picked]<.75)&(corrected[picked]>=.75))),
                    success_mask75_damage=int(np.sum((original[picked]>=.75)&(corrected[picked]<.75))))
                item["top10_mean_auc_benefit"] = float(np.nanmean(auc_benefit[picked]))
            result["models"][key] = item
            print(json.dumps(dict(model=key, top10=item["top10"])), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (a.out / "COMPLETE.json").write_text(json.dumps(dict(images=2000,
        candidates=len(rows), selected=selected_name, test_passes=1), indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("manifest", "gate", "test_scores", "out"):
        p.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    main(p.parse_args())
