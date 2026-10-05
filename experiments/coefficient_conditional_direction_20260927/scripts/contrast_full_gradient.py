"""Image-clustered group contrasts for the completed full gradient retrieval."""
import argparse
import json
from pathlib import Path

import numpy as np


def contrast(ids, values, a, b, seed=20260927, repetitions=2000):
    images, inv = np.unique(ids, return_inverse=True)
    ac = np.bincount(inv, weights=a.astype(float), minlength=len(images))
    bc = np.bincount(inv, weights=b.astype(float), minlength=len(images))
    av = np.bincount(inv, weights=values * a, minlength=len(images))
    bv = np.bincount(inv, weights=values * b, minlength=len(images))
    rng = np.random.default_rng(seed)
    boot = np.empty(repetitions)
    for k in range(repetitions):
        draw = rng.integers(0, len(images), len(images))
        boot[k] = av[draw].sum() / ac[draw].sum() - bv[draw].sum() / bc[draw].sum()
    return {"a_count": int(a.sum()), "b_count": int(b.sum()),
            "a_mean": float(values[a].mean()), "b_mean": float(values[b].mean()),
            "a_minus_b": float(values[a].mean() - values[b].mean()),
            "ci95": np.quantile(boot, [.025, .975]).tolist()}


def main(args):
    m = np.load(args.moments, allow_pickle=False)
    n = np.load(args.neighbors, allow_pickle=False)
    assert np.array_equal(m["image_id"], n["image_id"])
    assert np.array_equal(m["annotation_id"], n["annotation_id"])
    good_box = m["box_iou"] >= .75
    failure = m["original_iou"] < .75
    high_h = n["h_cos"] >= .9
    output = {"posthoc_exploratory": True,
              "metric": "crossfold nearest h neighbor cosine of official GT-box coefficient gradient",
              "good_box_failure_minus_success": contrast(
                 m["image_id"], n["g_top"], good_box & failure,
                 good_box & ~failure),
              "good_box_h90_failure_minus_success": contrast(
                 m["image_id"], n["g_top"], good_box & failure & high_h,
                 good_box & ~failure & high_h)}
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--moments", type=Path, required=True)
    parser.add_argument("--neighbors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
