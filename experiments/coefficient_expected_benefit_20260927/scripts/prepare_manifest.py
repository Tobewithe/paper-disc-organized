"""Freeze 10k five-fold train IDs, untouched 2k val IDs and cluster power estimate."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm


def rank_id(prefix, image_id):
    return hashlib.sha256(f"7L-20260927:{prefix}:{image_id}".encode()).hexdigest()


def required_images(rows, selection, arm, metric, delta, alpha=.05, power=.9):
    ids = np.array([r["image_id"] for r in rows])
    images, group = np.unique(ids, return_inverse=True)
    n = np.bincount(group)
    before = np.array([r["original_image_iou"] for r in rows])
    after = np.array([r["arms"][arm]["image_iou"] for r in rows])
    if metric == "mask75":
        effect = ((after >= .75).astype(float) - (before >= .75).astype(float)) * selection
    else:
        effect = (after - before) * selection
    sums = np.bincount(group, weights=effect)
    mean = effect.mean()
    variance = np.var(sums - mean * n, ddof=1) / n.mean() ** 2
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    required = math.ceil(z * z * variance / delta ** 2)
    return dict(observed_images=len(images), observed_candidates=len(rows), observed_effect=float(mean),
                influence_variance=float(variance), min_effect=delta, alpha=alpha, power=power,
                required_images=required)


def main(a):
    fit_index = json.loads((a.tenk_bank / "INDEX.json").read_text())["fit"]
    train_ids = [int(r["image_id"]) for r in fit_index]
    assert len(train_ids) == len(set(train_ids)) == 10000
    old_index = json.loads((a.old_bank / "INDEX.json").read_text())
    dev_ids = {int(r["image_id"]) for r in old_index["dev"]}
    old_val = {int(r["image_id"]) for r in old_index["val"]}
    assert len(dev_ids) == len(old_val) == 200
    assert not (set(train_ids) & dev_ids)
    val_ids = {int(p.stem) for p in a.val_images.glob("*.jpg")}
    assert len(val_ids) == 5000 and old_val <= val_ids
    unseen = sorted(val_ids - old_val, key=lambda iid: rank_id("test", iid))
    assert len(unseen) == 4800
    test_ids = unseen[:2000]
    ordered_train = sorted(train_ids, key=lambda iid: rank_id("fold", iid))
    folds = [ordered_train[k::5] for k in range(5)]
    assert all(len(group) == 2000 for group in folds)
    assert set().union(*map(set, folds)) == set(train_ids)
    assert not (set(test_ids) & old_val)

    rows = json.loads((a.jn_evaluation / "ROWS.json").read_text())
    score_rows = json.loads((a.saved_scores / "SCORES.json").read_text())
    score_by_key = {(int(r["image_id"]), int(r["annotation_id"])): r for r in score_rows}
    gate = json.loads((a.gate_result / "RESULTS.json").read_text())
    threshold = gate["gates"]["h_raw_detection_failure"]["5"]["threshold"]
    selected = np.array([score_by_key[(r["image_id"], r["annotation_id"])]
                         ["scores"]["h_raw_detection"]["failure"] > threshold for r in rows])
    power = {}
    for arm in ("h_only", "true_local"):
        for name, mask in (("all", np.ones(len(rows), dtype=bool)), ("gate5", selected)):
            for metric, delta in (("mask75", .01), ("iou", .005)):
                power[f"{arm}_{name}_{metric}"] = required_images(rows, mask, arm, metric, delta)
    n_required = max(item["required_images"] for item in power.values())
    assert n_required <= len(unseen), "COCO val capacity below requested power; revise before test labels"
    manifest = dict(source_train_bank=str(a.tenk_bank), old_development_images=sorted(dev_ids),
                    excluded_reviewed_val_images=sorted(old_val), train_images=train_ids,
                    folds=folds, independent_test_images=test_ids,
                    rule="SHA256 7L-20260927 prefix, image-level partition; fixed before test label access",
                    power=power, max_required_images=n_required,
                    chosen_test_images=len(test_ids),
                    scope="Candidate-level effect only; no full COCO AP power claim")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(train=len(train_ids), fold_sizes=list(map(len, folds)),
                          test=len(test_ids), max_required_images=n_required,
                          old_val_excluded=len(old_val))))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("tenk_bank", "old_bank", "val_images", "jn_evaluation", "saved_scores", "gate_result", "out"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    main(p.parse_args())
