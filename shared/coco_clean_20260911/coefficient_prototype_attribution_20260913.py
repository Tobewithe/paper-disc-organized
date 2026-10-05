"""S080: Re-analyze S074 to separate coefficient opportunity from fixed-P limits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "experiments/coco_clean_20260911/diagnostics/fixed_prototype_split_20260913_v5/instances.csv"
OUT = ROOT / "experiments/coco_clean_20260911/diagnostics/coefficient_prototype_attribution_20260913"
OUT.mkdir(parents=True, exist_ok=True)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": None, "ci95": [None, None]}
    draws = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def summarize(frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    baseline_bad = frame["baseline_iou"] < 0.75
    frame = frame.loc[baseline_bad].copy()
    frame["coeff_reaches_75"] = frame["allfit_iou"] >= 0.75
    frame["split_reaches_75"] = frame["split_iou"] >= 0.75
    frame["fixed_p_below_75"] = ~frame["coeff_reaches_75"]
    frame["coeff_gain_points"] = (frame["allfit_iou"] - frame["baseline_iou"]) * 100.0
    frame["fixed_p_gap_points"] = np.maximum(0.0, 0.75 - frame["allfit_iou"]) * 100.0
    out = {
        "n": int(len(frame)),
        "coeff_reaches_75_n": int(frame["coeff_reaches_75"].sum()),
        "coeff_reaches_75_fraction": float(frame["coeff_reaches_75"].mean()) if len(frame) else None,
        "split_reaches_75_n": int(frame["split_reaches_75"].sum()),
        "split_reaches_75_fraction": float(frame["split_reaches_75"].mean()) if len(frame) else None,
        "fixed_p_below_75_n": int(frame["fixed_p_below_75"].sum()),
        "coeff_gain_points": bootstrap_mean(frame["coeff_gain_points"].to_numpy(), rng),
        "fixed_p_gap_points": bootstrap_mean(frame["fixed_p_gap_points"].to_numpy(), rng),
    }
    return out


def main():
    df = pd.read_csv(INPUT)
    rng = np.random.default_rng(20260913)
    rows = []
    for density in ["all", "high", "low"]:
        for residual in ["all", *sorted(df["residual"].unique())]:
            sub = df.copy()
            if density != "all":
                sub = sub[sub["density"] == density]
            if residual != "all":
                sub = sub[sub["residual"] == residual]
            item = {"density": density, "residual": residual, **summarize(sub, rng)}
            rows.append(item)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "SUMMARY.csv", index=False)
    df2 = df.copy()
    df2["baseline_bad"] = df2["baseline_iou"] < 0.75
    df2["coefficient_recoverable"] = df2["baseline_bad"] & (df2["allfit_iou"] >= 0.75)
    df2["fixed_prototype_limited"] = df2["baseline_bad"] & (df2["allfit_iou"] < 0.75)
    df2["coefficient_gain_points"] = (df2["allfit_iou"] - df2["baseline_iou"]) * 100.0
    df2["fixed_p_gap_points"] = np.maximum(0.0, 0.75 - df2["allfit_iou"]) * 100.0
    df2.to_csv(OUT / "instances_attributed.csv", index=False)
    payload = {
        "experiment": "S080_COEFFICIENT_PROTOTYPE_ATTRIBUTION",
        "status": "COMPLETE",
        "input": str(INPUT),
        "n_total": int(len(df)),
        "n_baseline_bad": int(df2["baseline_bad"].sum()),
        "definition": {
            "coefficient_recoverable": "baseline_iou < 0.75 and allfit_iou >= 0.75",
            "fixed_prototype_limited": "baseline_iou < 0.75 and allfit_iou < 0.75",
            "allfit": "GT-assisted coefficient fit on all sampled pixels with prototype, box and decoder fixed",
        },
        "summary": rows,
        "limitations": [
            "This is a re-analysis of S074, not a new forward pass, training run, or AP result.",
            "The all-fit coefficient is GT-assisted and is an upper-bound opportunity, not a deployable predictor.",
            "fixed_prototype_limited means the fixed-P conditional fit remains below threshold; it does not uniquely prove prototype causality.",
            "The queue was outcome-stratified and contains only 140 selected instances.",
        ],
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S080：系数机会与固定原型限制归因",
        "",
        "状态：COMPLETE / S074 复算。无新增前向、训练或 COCO AP。",
        "",
        "## 判定",
        "",
        "对原始 IoU<0.75 的实例，若固定原型下的 GT-assisted all-fit IoU 达到 0.75，记为“系数可修复”；否则记为“固定原型条件下仍受限”。后者不能单独证明原型因果，只表示即使给出全像素最优系数，当前固定 P、框和支持域仍不足以达到阈值。",
        "",
        "## 结果",
        "",
        "| 密度 | 失败子型 | n | 系数可修复 | 固定原型仍低于 .75 | all-fit 增益(点) | 固定原型余 gap(点) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        if r["residual"] == "all" or int(r["n"]) == 0:
            continue
        gain = r["coeff_gain_points"]["mean"] if isinstance(r["coeff_gain_points"], dict) else None
        gap = r["fixed_p_gap_points"]["mean"] if isinstance(r["fixed_p_gap_points"], dict) else None
        lines.append(
            f"| {r['density']} | {r['residual']} | {int(r['n'])} | "
            f"{int(r['coeff_reaches_75_n'])} ({100*r['coeff_reaches_75_fraction']:.1f}%) | "
            f"{int(r['fixed_p_below_75_n'])} ({100*r['fixed_p_below_75_n']/r['n']:.1f}%) | "
            f"{gain:.2f} | {gap:.2f} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "两类失败同时存在。`coefficient_recoverable` 是系数读出方法的条件机会；`fixed_prototype_limited` 说明只修系数不够，但其剩余瓶颈可能来自原型表达、目标支持、低分辨率和模型外的误差，不能直接命名为原型失败。高低密度均出现相同结构，因此 S080 不支持把该归因写成密集场景独有机制。下一步若做方法，应在独立图像上学习共享系数修正，并同时报告固定原型仍受限的残差，避免把 GT oracle 增益写成可部署收益。",
        "",
        "详见 `SUMMARY.json`、`SUMMARY.csv`、`instances_attributed.csv`；原始输入为 S074 `instances.csv`。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "out": str(OUT), "n": int(len(df))}, indent=2))


if __name__ == "__main__":
    main()
