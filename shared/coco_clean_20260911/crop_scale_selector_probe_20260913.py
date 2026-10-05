"""S079: inference-visible crop-scale selector probe.

Fit one regressor per candidate crop scale using only features available at
inference. The target (IoU at each scale) is used only on the disjoint fit
split; transfer performance is evaluated from frozen decoded masks. This is a
diagnostic upper-bound probe, not an end-to-end AP result or a trained method.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


SCALES = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)
FEATURES = (
    "ring_positive_fraction",
    "inside_positive_fraction",
    "ring_inside_ratio",
    "candidate_score",
    "candidate_class_index",
)


def mean_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "n": int(values.size)}


def evaluate(part, predictions):
    base = part["iou_1.0"].to_numpy(dtype=np.float64)
    selected = np.asarray(predictions, dtype=np.float64)
    out = {"n": int(len(part)), "base_iou": float(base.mean()),
           "selected_iou": float(selected.mean()),
           "selected_delta": float((selected - base).mean()),
           "oracle_delta": float((part[[f"iou_{s:.1f}" for s in SCALES]].max(axis=1).to_numpy() - base).mean())}
    for key, mask in {
        "high": part["group"].eq("high"),
        "low": part["group"].eq("low"),
        "support_sufficient_mask_bad": part["state"].eq("box_good_support_sufficient_mask_bad"),
        "support_low_mask_bad": part["state"].eq("box_good_support_low_mask_bad"),
        "mask_good": part["state"].eq("box_good_mask_good"),
    }.items():
        q = np.asarray(mask)
        if q.any():
            out[key] = {
                "n": int(q.sum()),
                "selected_delta": float((selected[q] - base[q]).mean()),
                "oracle_delta": float((part.loc[q, [f"iou_{s:.1f}" for s in SCALES]].max(axis=1).to_numpy() - base[q]).mean()),
            }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "diagnostics/crop_support_failure_cross_20260913/per_target.csv")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "diagnostics/crop_scale_selector_probe_20260913")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    fit = df[df["split"].eq("fit")].copy()
    transfer = df[df["split"].eq("transfer")].copy()
    X = fit[list(FEATURES)].fillna(0).to_numpy(dtype=np.float64)
    Xt = transfer[list(FEATURES)].fillna(0).to_numpy(dtype=np.float64)
    y_transfer = np.stack([transfer[f"iou_{s:.1f}"].to_numpy(dtype=np.float64) for s in SCALES], axis=1)
    predicted = []
    for k, scale in enumerate(SCALES):
        model = ExtraTreesRegressor(
            n_estimators=100,
            min_samples_leaf=20,
            max_features=1.0,
            random_state=20260913 + k,
            n_jobs=1,
        )
        model.fit(X, fit[f"iou_{scale:.1f}"].to_numpy(dtype=np.float64))
        predicted.append(model.predict(Xt))
    predicted = np.stack(predicted, axis=1)
    selected_index = np.argmax(predicted, axis=1)
    selected_iou = y_transfer[np.arange(len(transfer)), selected_index]
    result = {
        "status": "COMPLETE",
        "input": str(args.input),
        "fit_targets": int(len(fit)),
        "transfer_targets": int(len(transfer)),
        "features": list(FEATURES),
        "scales": list(SCALES),
        "model": "ExtraTreesRegressor(n_estimators=100,min_samples_leaf=20,max_features=1.0)",
        "selected_scale_counts": {str(s): int((selected_index == k).sum()) for k, s in enumerate(SCALES)},
        "transfer": evaluate(transfer.reset_index(drop=True), selected_iou),
        "limitations": "Fit labels use GT-derived IoU; transfer selection uses only inference-visible features. Frozen paired-candidate diagnostic; no end-to-end AP or training.",
    }
    (args.out / "SUMMARY.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Inference-visible crop-scale selector probe (S079)", "",
             "The selector is calibrated on the fit split with GT-derived per-scale IoU labels and evaluated on disjoint transfer targets. At transfer time it uses only raw border/inside response, candidate score, and predicted class.", "",
             f"- Fit targets: {len(fit)}; transfer targets: {len(transfer)}",
             f"- Transfer baseline IoU@1.0: {result['transfer']['base_iou']:.6f}",
             f"- Transfer selector IoU: {result['transfer']['selected_iou']:.6f} (delta {result['transfer']['selected_delta']:+.6f})",
             f"- Transfer oracle delta: {result['transfer']['oracle_delta']:+.6f}", "",
             "## Transfer subgroup deltas"]
    for key in ("high", "low", "support_sufficient_mask_bad", "support_low_mask_bad", "mask_good"):
        if key in result["transfer"]:
            v = result["transfer"][key]
            lines.append(f"- {key}: n={v['n']}, selector {v['selected_delta']:+.6f}, oracle {v['oracle_delta']:+.6f}")
    lines += ["", "Interpretation: a learned inference-visible selector is compared against the unchanged 1.0 crop. This result is not an AP claim and cannot use the GT-derived state at deployment.", ""]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
