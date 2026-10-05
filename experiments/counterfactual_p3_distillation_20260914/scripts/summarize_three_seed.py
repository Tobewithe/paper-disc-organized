from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "three_seed_evidence"
RUNS = EVIDENCE / "runs"

STANDARD_COLUMNS = {
    "box_ap": "metrics/mAP50-95(B)",
    "mask_ap": "metrics/mAP50-95(M)",
    "box_ap50": "metrics/mAP50(B)",
    "mask_ap50": "metrics/mAP50(M)",
    "box_recall": "metrics/recall(B)",
    "mask_recall": "metrics/recall(M)",
}

TARGET_COLUMNS = [
    "raw_best_box_iou_delta",
    "raw_box50_delta",
    "final_box_iou_delta",
    "final_box50_delta",
    "final_mask_iou_delta",
    "final_mask75_delta",
]


def sample_stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "sample_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
    }


def main() -> None:
    standard_rows: list[dict[str, object]] = []
    aggregate: dict[str, object] = {"standard": {}, "targeted": {}}

    for seed in range(3):
        base = pd.read_csv(RUNS / f"baseline_s{seed}" / "results.csv")
        method = pd.read_csv(RUNS / f"cfp3r_s{seed}" / "results.csv")
        if len(base) != 3 or len(method) != 3:
            raise RuntimeError(f"seed {seed}: expected three validation rows per arm")

        for policy, base_row, method_row in (
            ("fixed_final_epoch", base.iloc[-1], method.iloc[-1]),
            (
                "best_mask_ap_per_arm",
                base.loc[base[STANDARD_COLUMNS["mask_ap"]].idxmax()],
                method.loc[method[STANDARD_COLUMNS["mask_ap"]].idxmax()],
            ),
        ):
            row: dict[str, object] = {
                "seed": seed,
                "checkpoint_policy": policy,
                "baseline_epoch": int(base_row["epoch"]),
                "method_epoch": int(method_row["epoch"]),
            }
            for short, column in STANDARD_COLUMNS.items():
                row[f"baseline_{short}"] = float(base_row[column])
                row[f"method_{short}"] = float(method_row[column])
                row[f"delta_{short}_points"] = 100.0 * float(method_row[column] - base_row[column])
            standard_rows.append(row)

    standard = pd.DataFrame(standard_rows)
    standard.to_csv(ROOT / "three_seed_standard_metrics.csv", index=False)

    for policy in standard["checkpoint_policy"].unique():
        part = standard[standard["checkpoint_policy"] == policy]
        aggregate["standard"][policy] = {
            short: sample_stats(part[f"delta_{short}_points"].tolist())
            for short in STANDARD_COLUMNS
        }

    targeted_rows: list[dict[str, object]] = []
    for seed in range(3):
        df = pd.read_csv(EVIDENCE / f"targeted_summary_s{seed}.csv")
        for _, row in df[df["arm"] == "cfp3r"].iterrows():
            out: dict[str, object] = {"seed": seed, "cohort": row["cohort"], "n": int(row["n"])}
            for column in TARGET_COLUMNS:
                out[column] = float(row[column])
            targeted_rows.append(out)
    targeted = pd.DataFrame(targeted_rows)
    targeted.to_csv(ROOT / "three_seed_targeted_metrics.csv", index=False)

    for cohort in targeted["cohort"].unique():
        part = targeted[targeted["cohort"] == cohort]
        aggregate["targeted"][cohort] = {
            column: sample_stats(part[column].tolist()) for column in TARGET_COLUMNS
        }

    (ROOT / "three_seed_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    final = standard[standard["checkpoint_policy"] == "fixed_final_epoch"]
    best = standard[standard["checkpoint_policy"] == "best_mask_ap_per_arm"]
    failure = targeted[targeted["cohort"] == "raw_geometry_small"]
    control = targeted[targeted["cohort"] == "matched_small_control"]

    lines = [
        "# Three-seed short-run confirmation",
        "",
        "All values below are method minus the matched Baseline. Standard AP values are percentage points. "
        "Targeted evaluations use each arm's `best.pt` and the same 239 frozen raw-geometry-small failures plus 199 matched controls.",
        "",
        "## Fixed final epoch (epoch 3)",
        "",
        "| Seed | Box AP | Mask AP | Box AP50 | Mask AP50 | Box Recall | Mask Recall |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in final.iterrows():
        lines.append(
            f"| {int(row.seed)} | {row.delta_box_ap_points:+.3f} | {row.delta_mask_ap_points:+.3f} | "
            f"{row.delta_box_ap50_points:+.3f} | {row.delta_mask_ap50_points:+.3f} | "
            f"{row.delta_box_recall_points:+.3f} | {row.delta_mask_recall_points:+.3f} |"
        )
    mean = final[[f"delta_{x}_points" for x in STANDARD_COLUMNS]].mean()
    sd = final[[f"delta_{x}_points" for x in STANDARD_COLUMNS]].std(ddof=1)
    lines.extend(
        [
            f"| Mean | {mean.delta_box_ap_points:+.3f} | {mean.delta_mask_ap_points:+.3f} | "
            f"{mean.delta_box_ap50_points:+.3f} | {mean.delta_mask_ap50_points:+.3f} | "
            f"{mean.delta_box_recall_points:+.3f} | {mean.delta_mask_recall_points:+.3f} |",
            f"| Sample SD | {sd.delta_box_ap_points:.3f} | {sd.delta_mask_ap_points:.3f} | "
            f"{sd.delta_box_ap50_points:.3f} | {sd.delta_mask_ap50_points:.3f} | "
            f"{sd.delta_box_recall_points:.3f} | {sd.delta_mask_recall_points:.3f} |",
            "",
            "## Best Mask-AP checkpoint selected separately per arm",
            "",
            "| Seed | Baseline epoch | Method epoch | Mask AP delta |",
            "|---:|---:|---:|---:|",
        ]
    )
    for _, row in best.iterrows():
        lines.append(
            f"| {int(row.seed)} | {int(row.baseline_epoch)} | {int(row.method_epoch)} | {row.delta_mask_ap_points:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Frozen targeted cohorts (`best.pt`)",
            "",
            "| Cohort | Metric | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    labels = {
        "raw_best_box_iou_delta": "Raw best Box IoU",
        "raw_box50_delta": "Raw Box50 rate",
        "final_box_iou_delta": "Final Box IoU",
        "final_box50_delta": "Final Box50 rate",
        "final_mask_iou_delta": "Final Mask IoU",
        "final_mask75_delta": "Final Mask75 rate",
    }
    for cohort_name, cohort_df in (("Failure", failure), ("Control", control)):
        for column in TARGET_COLUMNS:
            values = cohort_df.sort_values("seed")[column].to_numpy() * 100.0
            lines.append(
                f"| {cohort_name} | {labels[column]} | {values[0]:+.3f} | {values[1]:+.3f} | "
                f"{values[2]:+.3f} | {values.mean():+.3f} ± {values.std(ddof=1):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The diagnosed failure cohort improves in continuous raw Box IoU, final Box IoU and final Mask IoU in all three seeds; raw Box50 is also positive in every seed. "
            "This is reproducible mechanism-to-training evidence. It is not yet a final COCO performance claim: training used a 1,000-image pilot for only three epochs, controls also gain on some final metrics, no target crossed Mask75, and independently selected best checkpoints do not improve Mask AP in seeds 1 and 2. "
            "The failure cohort is defined and evaluated on the one-to-many + NMS branch, while training `results.csv` uses default one-to-one validation, so final confirmation must make branch-specific results explicit. "
            "The next confirmatory experiment must use a fixed full training budget and a predeclared checkpoint rule.",
            "",
        ]
    )
    (ROOT / "THREE_SEED_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
