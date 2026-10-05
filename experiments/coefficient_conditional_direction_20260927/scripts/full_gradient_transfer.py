"""Full-population crossfold gradient agreement for nearest GT-controlled h."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_conditional_direction import load_bank, matching_groups, retrieve


def row_cos(a, b):
    return np.sum(a * b, axis=1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def cluster_ci(ids, values, selected, reps=2000, seed=20260927):
    images, inverse = np.unique(ids[selected], return_inverse=True)
    sums = np.bincount(inverse, weights=values[selected], minlength=len(images))
    counts = np.bincount(inverse, minlength=len(images))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(images), size=(reps, len(images)), dtype=np.int32)
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    return {"mean": float(values[selected].mean()),
            "ci95": np.quantile(boot, [.025, .975]).tolist()}


def main(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    bank = load_bank(args.targets / "shards", args.annotations, manifest)
    moment = np.load(args.moments, allow_pickle=False)
    n = len(bank["image_id"])
    assert np.array_equal(bank["image_id"], moment["image_id"])
    assert np.array_equal(bank["annotation_id"], moment["annotation_id"])
    top = np.full(n, -1, np.int32)
    random = np.full(n, -1, np.int32)
    h_cos = np.full(n, np.nan, np.float32)
    rng = np.random.default_rng(20260927)
    for k in range(5):
        fit = np.flatnonzero(bank["fold"] != k).astype(np.int32)
        query = np.flatnonzero(bank["fold"] == k).astype(np.int32)
        h = bank["h"]
        mu, sd = h[fit].mean(0), h[fit].std(0)
        x = (h - mu) / np.maximum(sd, .01)
        x = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        pools, groups, fallback = matching_groups(bank, fit, query)
        retrieved, similarity = retrieve(x, pools, groups, n)
        top[query] = retrieved[query, 0]
        h_cos[query] = similarity[query]
        for group, rows in groups.items():
            eligible = pools[group]
            random[np.asarray(rows, np.int32)] = rng.choice(eligible, size=len(rows))
        print(json.dumps({"fold": k, "queries": len(query),
                          "fallback_counts": np.bincount(fallback[query], minlength=4).tolist()}),
              flush=True)
    assert (top >= 0).all() and (random >= 0).all()
    assert np.all(bank["image_id"] != bank["image_id"][top])
    assert np.all(bank["image_id"] != bank["image_id"][random])
    fields = {}
    for key in ("s", "t", "g"):
        a = moment[key].astype(np.float64)
        fields[key + "_top"] = row_cos(a, a[top])
        fields[key + "_random"] = row_cos(a, a[random])
    failure = bank["original_iou"] < .75
    goodbox = bank["box_iou"] >= .75
    masks = {"all": np.ones(n, bool), "failure": failure,
             "success": ~failure, "failure_good_box": failure & goodbox,
             "success_good_box": ~failure & goodbox,
             "severe_failure": bank["original_iou"] < .5,
             "failure_good_box_h90": failure & goodbox & (h_cos >= .9)}
    result = {"audit": {"instances": n, "images": len(set(bank["image_id"])),
               "image_disjoint": True, "same_category_area_level_pool": True,
               "random_control": "same eligible pool, one fixed draw per query",
               "gt_used_for_pool_matching_only": True}, "groups": {}}
    for group, mask in masks.items():
        d = {"instances": int(mask.sum()), "images": int(len(set(bank["image_id"][mask]))),
             "h_nearest_similarity_mean": float(np.mean(h_cos[mask]))}
        for key in ("s", "t", "g"):
            nearest, control = fields[key + "_top"], fields[key + "_random"]
            d[key + "_nearest_cos_mean"] = float(np.mean(nearest[mask]))
            d[key + "_random_cos_mean"] = float(np.mean(control[mask]))
            d[key + "_nearest_nonpositive"] = float(np.mean(nearest[mask] <= 0))
            d[key + "_nearest_minus_random"] = cluster_ci(
                bank["image_id"], nearest - control, mask)
        result["groups"][group] = d
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "NEIGHBORS.npz", image_id=bank["image_id"],
             annotation_id=bank["annotation_id"], fold=bank["fold"],
             top=top, random=random, h_cos=h_cos, **fields)
    (args.out / "SUMMARY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"instances": n, "folds": 5,
           "matched_random_control": True}, indent=2), encoding="utf-8")
    print(json.dumps({"complete": n, "good_box_failure":
        result["groups"]["failure_good_box"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("manifest", "targets", "annotations", "moments", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    main(parser.parse_args())
