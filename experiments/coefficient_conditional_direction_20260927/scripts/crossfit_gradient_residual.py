"""Exploratory matched-capacity direct prediction of the official residual gradient."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from crossfit_target_moment import MODES, cluster_ci, cosine


def main(args):
    data = np.load(args.moments, allow_pickle=False)
    indirect = np.load(args.indirect, allow_pickle=False)
    image, ann, fold = (data[k] for k in ("image_id", "annotation_id", "fold"))
    assert np.array_equal(image, indirect["image_id"])
    assert np.array_equal(ann, indirect["annotation_id"])
    assert np.array_equal(fold, indirect["fold"])
    g = data["g"].astype(np.float64)
    s = data["s"].astype(np.float64)
    s_pred = data["s_predbox"].astype(np.float64)
    goodbox = data["box_iou"] >= .75
    success = data["original_iou"] >= .75
    groups = {"all": np.ones(len(image), dtype=bool), "failure": ~success,
              "success": success, "failure_good_box": ~success & goodbox,
              "success_good_box": success & goodbox,
              "severe_failure": data["original_iou"] < .5}
    direct = {mode: np.empty_like(g, dtype=np.float32) for mode in MODES}
    for mode, blocks in MODES.items():
        x = np.concatenate([s_pred if b == "s" else data[b] for b in blocks],
                           axis=1).astype(np.float64)
        for k in range(5):
            fit, test = fold != k, fold == k
            xs, ys = StandardScaler(), StandardScaler()
            train_x, train_y = xs.fit_transform(x[fit]), ys.fit_transform(g[fit])
            model = Ridge(alpha=100., solver="lsqr", tol=1e-5)
            model.fit(train_x, train_y)
            direct[mode][test] = ys.inverse_transform(
                model.predict(xs.transform(x[test]))).astype(np.float32)
            print(json.dumps({"mode": mode, "fold": k, "test_instances": int(test.sum())}), flush=True)
    summary = {"audit": {"instances": len(image), "images": len(set(image)),
               "fivefold_image_disjoint": True, "alpha": 100., "direct_gradient_target": True,
               "posthoc_exploratory": True}, "groups": {}}
    norms = np.maximum(np.linalg.norm(g, axis=1), 1e-8)
    for name, mask in groups.items():
        summary["groups"][name] = {}
        for mode in MODES:
            d = direct[mode].astype(np.float64)
            indirect_g = s - indirect[mode].astype(np.float64)
            direct_cos, indirect_cos = cosine(d, g), cosine(indirect_g, g)
            dnorm = np.linalg.norm(d - g, axis=1) / norms
            summary["groups"][name][mode] = {
                "instances": int(mask.sum()),
                "direct_cos_mean": float(direct_cos[mask].mean()),
                "direct_cos_nonpositive": float(np.mean(direct_cos[mask] <= 0)),
                "direct_relative_error_median": float(np.median(dnorm[mask])),
                "indirect_cos_mean": float(indirect_cos[mask].mean()),
                "direct_minus_indirect_cos": cluster_ci(image, direct_cos-indirect_cos, mask),
            }
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "OOF_DIRECT_GRADIENT.npz", image_id=image, annotation_id=ann,
             fold=fold, **direct)
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"instances": len(image),
          "images": len(set(image)), "folds": 5, "modes": list(MODES)}, indent=2), encoding="utf-8")
    print(json.dumps({"completed": len(image),
        "good_box_failure": {m: summary["groups"]["failure_good_box"][m]
                            for m in MODES}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--moments", type=Path, required=True)
    parser.add_argument("--indirect", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
