"""Fit simple GT-free expected-benefit predictors on cross-fitted correction outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge, SGDRegressor
from sklearn.metrics import (average_precision_score, brier_score_loss, r2_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch


def rank_id(iid):
    return int(hashlib.sha256(f"7L-20260927:gate-dev:{iid}".encode()).hexdigest(), 16)


def load_scores(paths):
    all_rows = []
    all_images = set()
    for k, path in enumerate(paths):
        complete = json.loads((path / "COMPLETE.json").read_text())
        assert complete["fold"] == k and complete["images"] == 2000
        parts = sorted((path / "shards").glob("PART_*.pt"))
        assert parts
        rows = []
        images = set()
        for part in parts:
            rows.extend(torch.load(part, weights_only=True, map_location="cpu"))
            meta = json.loads(part.with_suffix(".json").read_text())
            images.update(int(r["image_id"]) for r in meta["images"])
        assert len(images) == 2000 and not (images & all_images)
        assert len(rows) == complete["instances"]
        assert all(int(r["fold"]) == k for r in rows)
        all_images.update(images)
        all_rows.extend(rows)
    assert len(all_images) == 10000
    keys = [(int(r["image_id"]), int(r["annotation_id"]), int(r["original_raw_id"]))
            for r in all_rows]
    assert len(keys) == len(set(keys))
    return all_rows, all_images


def features(row):
    basic = row["basic"].numpy()
    onehot = np.zeros(80, np.float32)
    onehot[int(np.argmax(basic[15:]))] = 1
    x1 = np.r_[basic[:15], onehot]
    x2 = np.r_[x1, basic[15:], row["h"].numpy(), row["raw_detection"].numpy()]
    x3 = np.r_[x2, row["correction_stats"].numpy()]
    x4 = np.r_[x3, row["local_summary"].numpy()]
    return x1, x2, x3, x4


def top_summary(scores, benefit, repair, damage, top=.1):
    count = max(1, int(np.ceil(top * len(scores))))
    idx = np.argsort(scores, kind="stable")[-count:]
    n_repair = int(repair[idx].sum())
    n_damage = int(damage[idx].sum())
    return dict(selected=count, fraction=float(count / len(scores)),
        mean_benefit=float(benefit[idx].mean()),
        repair=n_repair, damage=n_damage,
        repair_damage_ratio=float(n_repair / max(n_damage, 1)))


def ece(y, probability, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    result = 0.
    for j in range(bins):
        selected = (probability >= edges[j]) & (
            (probability < edges[j + 1]) if j < bins - 1 else (probability <= edges[j + 1]))
        if selected.any():
            result += selected.mean() * abs(y[selected].mean() - probability[selected].mean())
    return float(result)


def main(a):
    torch.set_num_threads(6)
    rows, images = load_scores(a.scores)
    manifest = json.loads(a.manifest.read_text())
    assert images == set(map(int, manifest["train_images"]))
    xlists = [[], [], [], []]
    for row in rows:
        values = features(row)
        for j, value in enumerate(values):
            xlists[j].append(value)
    x = [np.asarray(values, dtype=np.float32) for values in xlists]
    benefit = np.asarray([float(r["benefit_iou"]) for r in rows])
    label = benefit > .01
    damage = benefit < -.01
    image_ids = np.asarray([int(r["image_id"]) for r in rows])
    ordered = sorted(images, key=rank_id)
    dev_ids = set(ordered[:2000])
    dev = np.fromiter((iid in dev_ids for iid in image_ids), bool, count=len(rows))
    fit = ~dev
    assert len(dev_ids) == 2000 and len(images - dev_ids) == 8000
    assert set(image_ids[dev]) <= dev_ids and set(image_ids[fit]) <= images - dev_ids
    assert len(set(image_ids[dev])) > 1900 and len(set(image_ids[fit])) > 7900
    assert fit.sum() > 0 and dev.sum() > 0 and label[fit].sum() > 0 and label[dev].sum() > 0
    a.out.mkdir(parents=True, exist_ok=True)
    results = dict(instances=len(rows), images=10000, gate_fit_images=8000,
        gate_dev_images=2000, positives=int(label.sum()), damages=int(damage.sum()),
        gate_fit_images_with_candidates=len(set(image_ids[fit])),
        gate_dev_images_with_candidates=len(set(image_ids[dev])),
        sklearn_version=sklearn.__version__,
        prevalence=float(label.mean()),
        benefit_overall=float(benefit.mean()),
        correction_radius=dict(mean=float(np.mean([float(r["correction"].norm()) for r in rows])),
            median=float(np.median([float(r["correction"].norm()) for r in rows]))),
        auc_valid=int(np.isfinite([r["benefit_auc"] for r in rows]).sum()),
        feature_dimensions={f"X{j+1}":int(z.shape[1]) for j,z in enumerate(x)},
        groups={})
    best_name, best_value = None, -float("inf")
    for j, data in enumerate(x, 1):
        group = {}
        for kind in ("ridge", "huber", "logistic"):
            if kind == "ridge":
                model = make_pipeline(StandardScaler(), Ridge(alpha=10., solver="lsqr"))
                target = benefit
            elif kind == "huber":
                model = make_pipeline(StandardScaler(), SGDRegressor(loss="huber",
                    epsilon=.01, alpha=1e-4, max_iter=1000, tol=1e-4,
                    learning_rate="adaptive", eta0=.001, random_state=20260927))
                target = benefit
            else:
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1., max_iter=400, solver="lbfgs"))
                target = label.astype(int)
            model.fit(data[fit], target[fit])
            raw = model.predict_proba(data[dev])[:, 1] if kind == "logistic" else model.predict(data[dev])
            item = dict(top10=top_summary(raw, benefit[dev], label[dev], damage[dev]),
                threshold_top10=float(np.quantile(raw, .9)))
            if kind == "logistic":
                calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                calibrator.fit(raw, label[dev].astype(int))
                item.update(auroc=float(roc_auc_score(label[dev], raw)),
                            auprc=float(average_precision_score(label[dev], raw)),
                            brier_uncalibrated=float(brier_score_loss(label[dev], raw)),
                            ece_uncalibrated=ece(label[dev], raw))
                joblib.dump(calibrator, a.out / f"X{j}_{kind}_calibration.joblib")
            else:
                item.update(spearman=float(spearmanr(benefit[dev], raw).statistic),
                            r2=float(r2_score(benefit[dev], raw)))
            joblib.dump(model, a.out / f"X{j}_{kind}.joblib")
            group[kind] = item
            if item["top10"]["mean_benefit"] > best_value:
                best_value = item["top10"]["mean_benefit"]
                best_name = f"X{j}_{kind}"
            print(json.dumps(dict(group=j, model=kind, result=item)), flush=True)
        results["groups"][f"X{j}"] = group
        (a.out / "DEVELOPMENT.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    results["selected_by_development_top10"] = best_name
    results["selected_dev_top10_mean_benefit"] = best_value
    (a.out / "DEVELOPMENT.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (a.out / "COMPLETE.json").write_text(json.dumps(dict(
        selected=best_name, instances=len(rows), images=10000,
        test_labels_read=False), indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--scores", type=Path, nargs=5, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
