"""Compose frozen 7I failure scores and frozen 7J/7J-N mask corrections."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


GATES = (("h_raw_detection", "failure"), ("box_score", "failure"),
         ("h_raw_detection", "high_gap"))
RATES = (0.01, 0.03, 0.05, 0.10)
ARMS = (("7JN_h_only", "h_only"), ("7JN_true_local", "true_local"),
        ("7J_h_only", "h_only"))


def key(row):
    return (int(row["image_id"]), int(row["annotation_id"]))


def read_features(directory):
    payload = torch.load(directory / "FEATURES.pt", weights_only=True, map_location="cpu")
    rows = payload["rows"]
    keys = [key(row) for row in rows]
    x_h = torch.stack([row["h"] for row in rows]).float().numpy()
    x_basic = torch.stack([row["basic"] for row in rows]).float().numpy()
    x_raw = torch.stack([row["raw_detection"] for row in rows]).float().numpy()
    label = np.array([int(row["failure"]) for row in rows], dtype=np.int8)
    gap = np.array([float(row["oracle_gap"]) for row in rows])
    del payload, rows
    gc.collect()
    return dict(keys=keys, label=label, gap=gap,
                box_score=x_basic,
                h_raw_detection=np.concatenate((x_h, x_basic, x_raw), axis=1))


def threshold_for_good_fpr(score, failure, rate):
    good = np.sort(score[failure == 0])[::-1]
    assert len(good) > 0
    n_allowed = math.floor(rate * len(good))
    threshold = float(good[n_allowed]) if n_allowed < len(good) else -float("inf")
    actual = float(np.mean(good > threshold))
    assert actual <= rate + 1e-12
    return threshold, actual


def as_arr(rows, arm, field):
    return np.array([r["arms"][arm][field] for r in rows], dtype=np.float64)


def clustered_ci(values, valid, image_index, samples):
    images = int(image_index.max()) + 1
    sums = np.bincount(image_index[valid], weights=values[valid], minlength=images)
    counts = np.bincount(image_index[valid], minlength=images)
    estimates = sums[samples].sum(axis=1) / np.maximum(counts[samples].sum(axis=1), 1)
    return [float(q) for q in np.quantile(estimates, [0.025, 0.975])]


def evaluate_selection(selected, baseline, corrected, failure, image_index, samples,
                       reference=None):
    original_good = ~failure
    repaired = selected & failure & (corrected["iou"] >= .75)
    damaged = selected & original_good & (corrected["iou"] < .75)
    iou_delta = np.where(selected, corrected["iou"] - baseline["iou"], 0.)
    auc_valid = np.isfinite(baseline["auc"]) & np.isfinite(corrected["auc"])
    auc_delta = np.where(selected & auc_valid, corrected["auc"] - baseline["auc"], 0.)
    net = repaired.astype(float) - damaged.astype(float)
    all_valid = np.ones(len(selected), dtype=bool)
    result = dict(
        selected=int(selected.sum()), coverage=float(selected.mean()),
        good_selected=int((selected & original_good).sum()),
        good_fpr=float(selected[original_good].mean()),
        failure_selected=int((selected & failure).sum()),
        failure_recall=float(selected[failure].mean()),
        failure_precision=float(failure[selected].mean()) if selected.any() else None,
        repairs=int(repaired.sum()), damages=int(damaged.sum()), net_repair=int(net.sum()),
        mask75_delta=float(net.mean()), mask75_delta_ci95=clustered_ci(net, all_valid, image_index, samples),
        image_iou_delta=float(iou_delta.mean()),
        image_iou_delta_ci95=clustered_ci(iou_delta, all_valid, image_index, samples),
        grid_auc_delta=float(auc_delta[auc_valid].mean()),
        grid_auc_delta_ci95=clustered_ci(auc_delta, auc_valid, image_index, samples),
        selected_failure_repair_rate=float(repaired.sum() / max((selected & failure).sum(), 1)),
        selected_good_damage_rate=float(damaged.sum() / max((selected & original_good).sum(), 1)),
    )
    if reference is not None:
        rselected = reference
        riou = np.where(rselected, corrected["iou"] - baseline["iou"], 0.)
        raug = np.where(rselected & auc_valid, corrected["auc"] - baseline["auc"], 0.)
        rnet = (rselected & failure & (corrected["iou"] >= .75)).astype(float) - (
            rselected & original_good & (corrected["iou"] < .75)).astype(float)
        result["versus_all_on"] = dict(
            mask75_delta=float((net - rnet).mean()),
            mask75_delta_ci95=clustered_ci(net - rnet, all_valid, image_index, samples),
            image_iou_delta=float((iou_delta - riou).mean()),
            image_iou_delta_ci95=clustered_ci(iou_delta - riou, all_valid, image_index, samples),
            grid_auc_delta=float((auc_delta[auc_valid] - raug[auc_valid]).mean()),
            grid_auc_delta_ci95=clustered_ci(auc_delta - raug, auc_valid, image_index, samples))
    return result


def main(args):
    sys.path.insert(0, str(args.score_source / "scripts"))
    from score_identifiability import logistic_fit, predict_logistic, scale, scale_fit

    fit = read_features(args.fit)
    dev = read_features(args.dev)
    val = read_features(args.val)
    assert len(set(fit["keys"])) == len(fit["keys"])
    assert len(set(dev["keys"])) == len(dev["keys"])
    assert len(set(val["keys"])) == len(val["keys"])
    assert set(fit["keys"]).isdisjoint(dev["keys"])
    assert set(val["keys"]).isdisjoint(fit["keys"] + dev["keys"])
    saved = json.loads((args.saved_scores / "SCORES.json").read_text())
    assert [key(row) for row in saved] == val["keys"]
    gap_threshold = float(np.quantile(fit["gap"], .75))
    scores, model_checks = {}, {}
    for feature, task in GATES:
        mean, std = scale_fit(fit[feature])
        xfit = scale(fit[feature], mean, std)
        xdev = scale(dev[feature], mean, std)
        xval = scale(val[feature], mean, std)
        labels = fit["label"] if task == "failure" else (fit["gap"] >= gap_threshold).astype(int)
        weights, _ = logistic_fit(xfit, labels)
        sd = predict_logistic(xdev, weights)
        sv = predict_logistic(xval, weights)
        old = np.array([r["scores"][feature][task] for r in saved])
        maxdiff = float(np.max(np.abs(sv - old)))
        assert maxdiff < 1e-8, (feature, task, maxdiff)
        scores[(feature, task)] = dict(dev=sd, val=sv)
        model_checks[f"{feature}_{task}"] = maxdiff
    del fit
    gc.collect()

    rows_jn = json.loads((args.jn_evaluation / "ROWS.json").read_text())
    rows_j = json.loads((args.j_evaluation / "ROWS.json").read_text())
    jn = {key(row): row for row in rows_jn}
    j = {key(row): row for row in rows_j}
    assert len(jn) == len(rows_jn) == len(val["keys"])
    assert len(j) == len(rows_j) == len(val["keys"])
    assert set(jn) == set(j) == set(val["keys"])
    ordered_jn = [jn[k] for k in val["keys"]]
    ordered_j = [j[k] for k in val["keys"]]
    b_iou = np.array([r["original_image_iou"] for r in ordered_jn])
    assert np.allclose(b_iou, [r["original_image_iou"] for r in ordered_j], atol=1e-10)
    assert np.array_equal(b_iou < .75, val["label"].astype(bool))
    b_auc = np.array([r["original_grid"]["auc"] if r["original_grid"]["auc"] is not None
                      else np.nan for r in ordered_jn])
    baseline = dict(iou=b_iou, auc=b_auc)
    corrected = {}
    for name, arm in ARMS:
        source = ordered_j if name.startswith("7J_") else ordered_jn
        corrected[name] = dict(iou=as_arr(source, arm, "image_iou"),
            auc=np.array([r["arms"][arm]["grid"]["auc"] if r["arms"][arm]["grid"]["auc"] is not None
                          else np.nan for r in source]))
    failure = val["label"].astype(bool)
    image_ids = np.array([k[0] for k in val["keys"]])
    _, image_index = np.unique(image_ids, return_inverse=True)
    image_count = int(image_index.max()) + 1
    samples = np.random.default_rng(20260927).integers(0, image_count, size=(3000, image_count))

    result = dict(protocol="PROTOCOL.md", n=len(failure), images=image_count,
                  failure_count=int(failure.sum()), original_success_count=int((~failure).sum()),
                  original_mask75=float((~failure).mean()), model_reproduction_maxdiff=model_checks,
                  all_on={}, gates={})
    all_on = np.ones(len(failure), dtype=bool)
    for name in corrected:
        result["all_on"][name] = evaluate_selection(all_on, baseline, corrected[name],
            failure, image_index, samples)
    for feature, task in GATES:
        gate_name = f"{feature}_{task}"
        sd, sv = scores[(feature, task)]["dev"], scores[(feature, task)]["val"]
        result["gates"][gate_name] = {}
        for rate in RATES:
            threshold, dev_fpr = threshold_for_good_fpr(sd, dev["label"].astype(bool), rate)
            selected = sv > threshold
            op = dict(target_dev_good_fpr=rate, threshold=threshold,
                      realized_dev_good_fpr=dev_fpr, arms={})
            for name in corrected:
                op["arms"][name] = evaluate_selection(selected, baseline, corrected[name],
                    failure, image_index, samples, reference=all_on)
            result["gates"][gate_name][str(int(rate * 100))] = op
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(all_on=result["all_on"], primary=result["gates"]["h_raw_detection_failure"]),
                     indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("score_source", "fit", "dev", "val", "saved_scores", "jn_evaluation", "j_evaluation", "out"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    main(p.parse_args())
