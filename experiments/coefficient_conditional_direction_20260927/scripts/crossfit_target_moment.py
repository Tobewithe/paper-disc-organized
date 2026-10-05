"""Image-disjoint fivefold prediction of official GT prototype-ownership moments."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


MODES = {
    "A_h": ("h",),
    "B_h_s": ("h", "s"),
    "C_h_s_gram": ("h", "s", "gram_diag"),
    "D_h_s_gram_local": ("h", "s", "gram_diag", "local_summary"),
}


def cosine(a, b):
    return np.sum(a * b, axis=1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def describe(m, t, g, s_pred, pred):
    e = pred - t
    tnorm = np.linalg.norm(t, axis=1)
    gnorm = np.linalg.norm(g, axis=1)
    et = np.linalg.norm(e, axis=1) / np.maximum(tnorm, 1e-8)
    eg = np.linalg.norm(e, axis=1) / np.maximum(gnorm, 1e-8)
    cos = cosine(g - e, g)
    model_proxy = s_pred - pred
    proxy_cos = cosine(model_proxy, g)
    proxy_error = np.linalg.norm(model_proxy - g, axis=1) / np.maximum(gnorm, 1e-8)
    denominator = np.sum((t - t.mean(0)) ** 2)
    return {"instances": int(m.sum()), "target_r2": float(1 - np.sum(e * e) / denominator),
            "target_relative_error_median": float(np.median(et)),
            "target_relative_error_p90": float(np.quantile(et, .9)),
            "gradient_relative_error_median": float(np.median(eg)),
            "gradient_relative_error_p90": float(np.quantile(eg, .9)),
            "gradient_cos_mean": float(np.mean(cos)),
            "gradient_cos_median": float(np.median(cos)),
            "gradient_cos_nonpositive": float(np.mean(cos <= 0)),
            "gtfree_proxy_gradient_cos_mean": float(np.mean(proxy_cos)),
            "gtfree_proxy_gradient_cos_nonpositive": float(np.mean(proxy_cos <= 0)),
            "gtfree_proxy_gradient_relative_error_median": float(np.median(proxy_error)),
            "response_support_error_to_gradient_median": float(np.median(
                np.linalg.norm(s_pred - (g + t), axis=1) / np.maximum(gnorm, 1e-8))),
            "target_norm_median": float(np.median(tnorm)),
            "gradient_norm_median": float(np.median(gnorm))}


def cluster_ci(image_id, values, selected, reps=2000, seed=20260927):
    ids, inv = np.unique(image_id[selected], return_inverse=True)
    v = values[selected]
    sums = np.bincount(inv, weights=v, minlength=len(ids))
    counts = np.bincount(inv, minlength=len(ids))
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for k in range(reps):
        draw = rng.integers(0, len(ids), size=len(ids))
        out[k] = sums[draw].sum() / counts[draw].sum()
    return {"mean": float(np.mean(v)), "ci95": np.quantile(out, [.025, .975]).tolist()}


def main(args):
    source = np.load(args.moments, allow_pickle=False)
    n = len(source["image_id"])
    assert n == 71269 and len(set(zip(source["image_id"], source["annotation_id"]))) == n
    fold, image = source["fold"], source["image_id"]
    assert all(len(set(image[fold == k]) & set(image[fold != k])) == 0 for k in range(5))
    t, g, s = (source[k].astype(np.float64) for k in ("t", "g", "s"))
    s_predbox = source["s_predbox"].astype(np.float64)
    assert np.max(np.abs(s - t - g)) < 2e-5
    success = source["original_iou"] >= .75
    goodbox = source["box_iou"] >= .75
    masks = {"all": np.ones(n, dtype=bool), "failure": ~success,
             "success": success, "failure_good_box": ~success & goodbox,
             "success_good_box": success & goodbox,
             "severe_failure": source["original_iou"] < .5}
    oof = {mode: np.empty_like(t, dtype=np.float32) for mode in MODES}
    fit_details = {}
    for mode, blocks in MODES.items():
        x = np.concatenate([s_predbox if b == "s" else source[b] for b in blocks],
                           axis=1).astype(np.float64)
        assert np.isfinite(x).all()
        fit_details[mode] = []
        for k in range(5):
            train, test = fold != k, fold == k
            scaler_x, scaler_y = StandardScaler(), StandardScaler()
            xtrain = scaler_x.fit_transform(x[train])
            ytrain = scaler_y.fit_transform(t[train])
            model = Ridge(alpha=100., solver="lsqr", tol=1e-5)
            model.fit(xtrain, ytrain)
            oof[mode][test] = scaler_y.inverse_transform(
                model.predict(scaler_x.transform(x[test]))).astype(np.float32)
            fit_details[mode].append({"fold": k, "fit_images": int(len(set(image[train]))),
                "test_images": int(len(set(image[test]))), "fit_instances": int(train.sum()),
                "test_instances": int(test.sum()), "features": int(x.shape[1])})
            print(json.dumps({"mode": mode, "fold": k, "test_instances": int(test.sum())}), flush=True)
    summary = {"audit": {"instances": n, "images": len(set(image)),
               "fivefold_image_disjoint": True, "alpha": 100.,
               "no_gt_in_features": True, "fit": fit_details}, "groups": {}, "paired": {}}
    for group, m in masks.items():
        summary["groups"][group] = {}
        for mode, p in oof.items():
            summary["groups"][group][mode] = describe(m, t[m], g[m], s_predbox[m], p[m])
        gradient_cos = {mode: cosine(s - p, g) for mode, p in oof.items()}
        proxy_cos = {mode: cosine(s_predbox - p, g) for mode, p in oof.items()}
        summary["paired"][group] = {}
        for mode in list(MODES)[1:]:
            diff = gradient_cos[mode] - gradient_cos["A_h"]
            summary["paired"][group][mode + "_minus_A_h_gradient_cos"] = cluster_ci(image, diff, m)
            proxy_diff = proxy_cos[mode] - proxy_cos["A_h"]
            summary["paired"][group][mode + "_minus_A_h_gtfree_proxy_cos"] = cluster_ci(image, proxy_diff, m)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "OOF_TARGET_PREDICTIONS.npz", image_id=image,
             annotation_id=source["annotation_id"], fold=fold, **oof)
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"instances": n, "images": len(set(image)),
          "folds": 5, "modes": list(MODES), "image_disjoint": True}, indent=2), encoding="utf-8")
    print(json.dumps({"completed": n, "groups": {name: {mode: summary["groups"][name][mode]["gradient_cos_mean"]
              for mode in MODES} for name in ("all", "failure_good_box")}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--moments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
