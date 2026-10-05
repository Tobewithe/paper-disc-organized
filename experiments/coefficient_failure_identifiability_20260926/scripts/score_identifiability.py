"""Simple fixed-regularization GT-free failure/gap prediction on fixed candidates."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr
import torch
import torch.nn.functional as F


ARMS = ("box_score", "h", "h_raw_detection", "h_local_space")


def load(path):
    obj = torch.load(path / "FEATURES.pt", weights_only=True, map_location="cpu")
    rows = obj["rows"]
    h = torch.stack([r["h"] for r in rows]).float().numpy()
    basic = torch.stack([r["basic"] for r in rows]).float().numpy()
    raw = torch.stack([r["raw_detection"] for r in rows]).float().numpy()
    local = torch.stack([r["local_hmap"] for r in rows]).float()
    pooled = F.adaptive_avg_pool2d(local, (2, 2)).flatten(1).numpy()
    std = local.flatten(2).std(2).numpy()
    features = dict(box_score=basic, h=h,
        h_raw_detection=np.concatenate((h, basic, raw), axis=1),
        h_local_space=np.concatenate((h, pooled, std), axis=1))
    assert all(np.isfinite(v).all() for v in features.values())
    return rows, features


def scale_fit(x):
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.maximum(std, .01)
    return mean, std


def scale(x, mean, std):
    return np.clip((x - mean) / std, -15, 15).astype(np.float64)


def logistic_fit(x, y, regularization=.01):
    n, d = x.shape
    design = np.column_stack((x, np.ones(n)))
    def objective(w):
        z = design @ w
        loss = float(np.logaddexp(0, z).mean() - np.mean(y * z) +
                     regularization / 2 * np.dot(w[:-1], w[:-1]))
        grad = design.T @ (1 / (1 + np.exp(-np.clip(z, -50, 50))) - y) / n
        grad[:-1] += regularization * w[:-1]
        return loss, grad
    result = minimize(objective, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                      options=dict(maxiter=350, ftol=1e-11))
    assert np.isfinite(result.x).all() and result.success, result.message
    return result.x, float(result.fun)


def predict_logistic(x, weights):
    z = np.column_stack((x, np.ones(len(x)))) @ weights
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def ridge_fit(x, y, regularization=10.):
    n = len(x)
    design = np.column_stack((x, np.ones(n)))
    gram = design.T @ design
    gram.flat[::len(gram) + 1] += regularization
    gram[-1, -1] -= regularization  # Do not shrink intercept.
    return np.linalg.solve(gram, design.T @ y)


def binary_metrics(y, score, threshold):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positive, negative = int(y.sum()), int((1 - y).sum())
    assert positive and negative
    ranks = rankdata(score)
    auc = (ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)
    order = np.argsort(-score, kind="stable")
    sorted_y = y[order]
    precision = np.cumsum(sorted_y) / np.arange(1, len(y) + 1)
    ap = float((precision * sorted_y).sum() / positive)
    bins = np.minimum((score * 10).astype(int), 9)
    ece = sum((bins == i).mean() * abs(float(score[bins == i].mean()) - float(y[bins == i].mean()))
              for i in range(10) if np.any(bins == i))
    selected = score > threshold
    return dict(n=len(y), prevalence=positive / len(y), auroc=float(auc), auprc=ap,
        brier=float(np.mean((score - y) ** 2)), ece10=float(ece),
        threshold=float(threshold), threshold_fpr=float(selected[y == 0].mean()),
        threshold_recall=float(selected[y == 1].mean()))


def cluster_contrast(rows, y, scores_a, scores_b, threshold_a, threshold_b):
    ids = np.array([r["image_id"] for r in rows])
    images = np.unique(ids)
    members = [np.flatnonzero(ids == iid) for iid in images]
    rng = np.random.default_rng(20260926)
    trials = []
    for _ in range(2000):
        chosen = np.concatenate([members[k] for k in rng.integers(0, len(images), len(images))])
        if len(np.unique(y[chosen])) < 2:
            continue
        a = binary_metrics(y[chosen], scores_a[chosen], threshold_a)
        b = binary_metrics(y[chosen], scores_b[chosen], threshold_b)
        trials.append([a["auroc"] - b["auroc"], a["auprc"] - b["auprc"],
                       a["threshold_recall"] - b["threshold_recall"]])
    return dict(auroc=np.quantile(trials, [.025, .975], axis=0)[:, 0].tolist(),
                auprc=np.quantile(trials, [.025, .975], axis=0)[:, 1].tolist(),
                recall=np.quantile(trials, [.025, .975], axis=0)[:, 2].tolist())


def main(a):
    fit_rows, fit = load(a.fit)
    dev_rows, dev = load(a.dev)
    val_rows, val = load(a.val)
    assert {r["image_id"] for r in fit_rows}.isdisjoint({r["image_id"] for r in dev_rows})
    assert {r["image_id"] for r in val_rows}.isdisjoint({r["image_id"] for r in fit_rows + dev_rows})
    gap_threshold = float(np.quantile([r["oracle_gap"] for r in fit_rows], .75))
    labels = {}
    for split, rows in (("fit", fit_rows), ("dev", dev_rows), ("val", val_rows)):
        labels[split] = dict(failure=np.array([r["failure"] for r in rows], dtype=int),
            high_gap=np.array([r["oracle_gap"] >= gap_threshold for r in rows], dtype=int),
            log_gap=np.log1p(np.maximum([r["oracle_gap"] for r in rows], 0)))
    results, scores = {}, {}
    for arm in ARMS:
        mean, std = scale_fit(fit[arm])
        xfit, xdev, xval = (scale(z[arm], mean, std) for z in (fit, dev, val))
        results[arm], scores[arm] = dict(dim=xfit.shape[1]), {}
        for task in ("failure", "high_gap"):
            weights, fit_loss = logistic_fit(xfit, labels["fit"][task])
            sd, sv = predict_logistic(xdev, weights), predict_logistic(xval, weights)
            threshold = float(np.quantile(sd[labels["dev"][task] == 0], .95))
            results[arm][task] = dict(fit_loss=fit_loss,
                dev=binary_metrics(labels["dev"][task], sd, threshold),
                val=binary_metrics(labels["val"][task], sv, threshold))
            scores[arm][task] = sv
        weights = ridge_fit(xfit, labels["fit"]["log_gap"])
        prediction = np.column_stack((xval, np.ones(len(xval)))) @ weights
        y = labels["val"]["log_gap"]
        results[arm]["continuous_gap"] = dict(
            spearman=float(spearmanr(prediction, y).statistic),
            r2=float(1 - np.sum((prediction - y) ** 2) / np.sum((y - y.mean()) ** 2)))
    contrasts = {}
    for task in ("failure", "high_gap"):
        contrasts[task] = {}
        for other in ("h", "box_score", "h_raw_detection"):
            contrasts[task]["h_local_space_minus_" + other] = cluster_contrast(
                val_rows, labels["val"][task], scores["h_local_space"][task], scores[other][task],
                results["h_local_space"][task]["val"]["threshold"],
                results[other][task]["val"]["threshold"])
    a.out.mkdir(parents=True, exist_ok=True)
    summary = dict(gap_threshold_fit_p75=gap_threshold,
        instances={"fit": len(fit_rows), "dev": len(dev_rows), "val": len(val_rows)},
        results=results, contrasts=contrasts)
    (a.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (a.out / "SCORES.json").write_text(json.dumps([
        dict(image_id=r["image_id"], annotation_id=r["annotation_id"],
             failure=int(labels["val"]["failure"][i]),
             high_gap=int(labels["val"]["high_gap"][i]),
             oracle_gap=float(r["oracle_gap"]),
             scores={arm: {task: float(scores[arm][task][i]) for task in ("failure", "high_gap")}
                     for arm in ARMS}) for i, r in enumerate(val_rows)], indent=2), encoding="utf-8")
    print(json.dumps({arm: {task: results[arm][task]["val"] for task in ("failure", "high_gap")}
                      for arm in ARMS}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("fit", "dev", "val", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
