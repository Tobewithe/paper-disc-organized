"""Aggregate official and rich target-level results across paired seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_METRICS = [
    "raw_best_box_iou", "raw_p3_best_box_iou", "raw_box50", "raw_box60", "raw_box70", "raw_box75",
    "raw_true_score_at_best_geometry", "raw_box_iou_at_max_true_score", "raw_center_error_norm",
    "final_candidate_exists", "final_box_iou", "final_box50", "final_box60", "final_box70", "final_box75",
    "mask_iou", "final_mask50", "final_mask60", "final_mask70", "final_mask75",
    "target_coverage", "prediction_purity", "false_positive_over_gt", "predicted_area_over_gt",
    "same_neighbor_leak_pred", "other_neighbor_leak_pred", "background_leak_pred",
    "same_neighbor_leak_gt", "other_neighbor_leak_gt", "background_leak_gt", "boundary_f1",
]

OFFICIAL_METRICS = [
    "ap", "ap50", "ap75", "aps", "apm", "apl", "ar_max1", "ar_max10", "ar_max100",
    "ar_small", "ar_medium", "ar_large", "recall_at_p80_iou50", "recall_at_p90_iou50",
    "recall_at_p80_iou75", "recall_at_p90_iou75",
]


def stats(values) -> tuple[float, float]:
    a = np.asarray(values, float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def boot(values, seed=1414, reps=5000):
    a = np.asarray(values, float); rng = np.random.default_rng(seed)
    means = np.array([rng.choice(a, len(a), replace=True).mean() for _ in range(reps)])
    lo, hi = np.quantile(means, [.025, .975])
    return float(a.mean()), float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); root = args.root
    official = pd.read_csv(root / "rich_eval_saved" / "official_metrics.csv")
    official_rows = []
    for seed in range(3):
        for task in ("bbox", "segm"):
            baseline = official[(official.arm == f"baseline_s{seed}") & (official.task == task)].iloc[0]
            method = official[(official.arm == f"cfp3r_s{seed}") & (official.task == task)].iloc[0]
            row = {"seed": seed, "task": task}
            row.update({metric: 100 * (float(method[metric]) - float(baseline[metric])) for metric in OFFICIAL_METRICS})
            official_rows.append(row)
    official_delta = pd.DataFrame(official_rows)
    official_delta.to_csv(root / "rich_official_deltas.csv", index=False)
    per_category = pd.read_csv(root / "rich_eval_saved" / "per_category_metrics.csv")
    category_rows = []
    for seed in range(3):
        for task in ("bbox", "segm"):
            baseline = per_category[(per_category.arm == f"baseline_s{seed}") & (per_category.task == task)]
            method = per_category[(per_category.arm == f"cfp3r_s{seed}") & (per_category.task == task)]
            joined = baseline.merge(method, on=["task", "category_id", "category_name"], suffixes=("_baseline", "_method"))
            for metric in ("ap", "ap50", "ap75", "r50", "r75"):
                delta = 100 * (joined[f"{metric}_method"] - joined[f"{metric}_baseline"])
                category_rows.append({"seed": seed, "task": task, "metric": metric, "categories": int(delta.notna().sum()), "macro_delta": float(delta.mean()), "improved": int((delta > 0).sum()), "degraded": int((delta < 0).sum())})
    pd.DataFrame(category_rows).to_csv(root / "rich_per_category_deltas.csv", index=False)

    paired_all = []
    target_seed_rows = []
    transition_rows = []
    for seed in range(3):
        data = pd.read_csv(root / f"rich_targeted_s{seed}" / "per_target.csv")
        baseline = data[data.arm == "baseline"]
        method = data[data.arm == "method"]
        paired = baseline.merge(method, on=["cohort", "pair_id", "image_id", "annotation_id"], suffixes=("_baseline", "_method"))
        paired["seed"] = seed
        for metric in TARGET_METRICS:
            paired[f"delta_{metric}"] = paired[f"{metric}_method"] - paired[f"{metric}_baseline"]
        paired_all.append(paired)
        for cohort, group in paired.groupby("cohort"):
            for metric in TARGET_METRICS:
                mean, low, high = boot(group[f"delta_{metric}"], seed=900+seed)
                target_seed_rows.append({"seed": seed, "cohort": cohort, "metric": metric, "n": len(group), "delta": mean, "ci_low": low, "ci_high": high})
            for metric in [m for m in TARGET_METRICS if m.endswith(("50", "60", "70", "75")) or m == "final_candidate_exists"]:
                before = group[f"{metric}_baseline"].astype(int); after = group[f"{metric}_method"].astype(int)
                transition_rows.append({"seed": seed, "cohort": cohort, "metric": metric, "n": len(group), "recovered": int(((before == 0) & (after == 1)).sum()), "degraded": int(((before == 1) & (after == 0)).sum()), "net": int((after-before).sum())})
    paired = pd.concat(paired_all, ignore_index=True)
    pd.DataFrame(target_seed_rows).to_csv(root / "rich_targeted_seed_metrics.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(root / "rich_threshold_transitions.csv", index=False)
    state_transitions = (paired.groupby(["seed", "cohort", "final_failure_state_baseline", "final_failure_state_method"])
                         .size().reset_index(name="n"))
    state_transitions.to_csv(root / "rich_failure_state_transitions.csv", index=False)

    micro_rows = []
    for seed in range(3):
        data = pd.read_csv(root / f"rich_targeted_s{seed}" / "per_target.csv")
        for cohort, cohort_data in data.groupby("cohort"):
            arm_values = {}
            for arm, arm_data in cohort_data.groupby("arm"):
                gt_pixels = float(arm_data.gt_pixels.sum())
                pred_pixels = float(arm_data.predicted_pixels.sum())
                arm_values[arm] = {
                    "coverage_micro": float(arm_data.true_positive_pixels.sum()) / max(gt_pixels, 1),
                    "purity_micro": float(arm_data.true_positive_pixels.sum()) / max(pred_pixels, 1),
                    "same_neighbor_leak_pred_micro": float(arm_data.same_neighbor_fp_pixels.sum()) / max(pred_pixels, 1),
                    "other_neighbor_leak_pred_micro": float(arm_data.other_neighbor_fp_pixels.sum()) / max(pred_pixels, 1),
                    "background_leak_pred_micro": float(arm_data.background_fp_pixels.sum()) / max(pred_pixels, 1),
                    "same_neighbor_leak_gt_micro": float(arm_data.same_neighbor_fp_pixels.sum()) / max(gt_pixels, 1),
                    "other_neighbor_leak_gt_micro": float(arm_data.other_neighbor_fp_pixels.sum()) / max(gt_pixels, 1),
                    "background_leak_gt_micro": float(arm_data.background_fp_pixels.sum()) / max(gt_pixels, 1),
                }
            for metric, method_value in arm_values["method"].items():
                micro_rows.append({"seed": seed, "cohort": cohort, "metric": metric, "baseline": arm_values["baseline"][metric], "method": method_value, "delta": method_value-arm_values["baseline"][metric]})
    micro = pd.DataFrame(micro_rows)
    micro.to_csv(root / "rich_spatial_micro_metrics.csv", index=False)

    interaction_rows = []
    for seed, seed_data in paired.groupby("seed"):
        failure = seed_data[seed_data.cohort == "raw_geometry_small"]
        control = seed_data[seed_data.cohort == "matched_small_control"]
        for metric in TARGET_METRICS:
            joined = failure[["pair_id", f"delta_{metric}"]].merge(control[["pair_id", f"delta_{metric}"]], on="pair_id", suffixes=("_failure", "_control"))
            values = joined[f"delta_{metric}_failure"] - joined[f"delta_{metric}_control"]
            mean, low, high = boot(values, seed=1200+int(seed))
            interaction_rows.append({"seed": int(seed), "metric": metric, "pairs": len(joined), "interaction": mean, "ci_low": low, "ci_high": high})
    interactions = pd.DataFrame(interaction_rows)
    interactions.to_csv(root / "rich_failure_control_interactions.csv", index=False)

    # Exploratory strata use fixed, interpretable bins; contrast and within-small area use pooled seed-0 tertiles.
    reference = paired[(paired.seed == 0) & (paired.cohort == "raw_geometry_small")]
    area_q = np.unique(reference["coco_area_baseline"].quantile([0, 1/3, 2/3, 1]).to_numpy())
    contrast_q = np.unique(reference["target_ring_lab_contrast_baseline"].quantile([0, 1/3, 2/3, 1]).to_numpy())
    strata_rows = []
    for seed, data in paired.groupby("seed"):
        data = data.copy()
        data["ici_bin"] = pd.cut(data.same_class_box_ici_baseline, [-np.inf, 0, .1, .5, np.inf], labels=["zero", "low", "middle", "high"])
        data["e4_bin"] = pd.cut(data.same_class_boundary_exposure_e4_baseline, [-np.inf, 0, .2, np.inf], labels=["zero", "middle", "high"])
        data["baseline_severity"] = pd.cut(data.raw_best_box_iou_baseline, [-np.inf, .25, .4, .5, np.inf], labels=["severe", "moderate", "near50", "already50"])
        if len(area_q) == 4:
            data["area_tertile"] = pd.cut(data.coco_area_baseline, area_q, labels=["smallest", "middle", "largest"], include_lowest=True)
        if len(contrast_q) == 4:
            data["contrast_tertile"] = pd.cut(data.target_ring_lab_contrast_baseline, contrast_q, labels=["low", "middle", "high"], include_lowest=True)
        for variable in ["ici_bin", "e4_bin", "baseline_severity", "area_tertile", "contrast_tertile"]:
            if variable not in data:
                continue
            for (cohort, level), group in data.groupby(["cohort", variable], observed=True):
                for metric in ["raw_best_box_iou", "raw_box50", "final_box_iou", "mask_iou", "target_coverage", "prediction_purity", "same_neighbor_leak_gt", "background_leak_gt", "boundary_f1"]:
                    strata_rows.append({"seed": int(seed), "cohort": cohort, "stratum": variable, "level": str(level), "metric": metric, "n": len(group), "delta": float(group[f"delta_{metric}"].mean())})
    pd.DataFrame(strata_rows).to_csv(root / "rich_stratified_effects.csv", index=False)
    (root / "rich_evaluation_bins.json").write_text(json.dumps({"area_tertile_edges": area_q.tolist(), "contrast_tertile_edges": contrast_q.tolist()}, indent=2), encoding="utf-8")

    lines = ["# Rich three-seed evaluation", "", "All deltas are method minus the matched Baseline. Official values are AP/recall percentage points.", "", "## Official original-annotation COCOeval on saved post-training best.pt predictions", "", "| Task | Metric | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |", "|---|---|---:|---:|---:|---:|"]
    display_official = ["ap", "ap50", "ap75", "aps", "ar_max100", "ar_small", "recall_at_p90_iou75"]
    for task in ("bbox", "segm"):
        part = official_delta[official_delta.task == task].sort_values("seed")
        for metric in display_official:
            values = part[metric].to_numpy(); mean, sd = stats(values)
            lines.append(f"| {task} | {metric} | {values[0]:+.3f} | {values[1]:+.3f} | {values[2]:+.3f} | {mean:+.3f} ± {sd:.3f} |")
    lines += ["", "## Frozen raw-geometry-small failure cohort", "", "| Metric | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |", "|---|---:|---:|---:|---:|"]
    target_table = pd.DataFrame(target_seed_rows)
    display_target = ["raw_p3_best_box_iou", "raw_best_box_iou", "raw_box50", "raw_center_error_norm", "final_box_iou", "mask_iou", "target_coverage", "prediction_purity", "same_neighbor_leak_pred", "background_leak_pred", "boundary_f1", "final_mask75"]
    for metric in display_target:
        part = target_table[(target_table.cohort == "raw_geometry_small") & (target_table.metric == metric)].sort_values("seed")
        values = 100 * part.delta.to_numpy(); mean, sd = stats(values)
        lines.append(f"| {metric} | {values[0]:+.3f} | {values[1]:+.3f} | {values[2]:+.3f} | {mean:+.3f} ± {sd:.3f} |")
    lines += ["", "## Pixel-count micro averages in the failure cohort", "", "| Metric | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |", "|---|---:|---:|---:|---:|"]
    for metric in ["coverage_micro", "purity_micro", "same_neighbor_leak_pred_micro", "other_neighbor_leak_pred_micro", "background_leak_pred_micro"]:
        part = micro[(micro.cohort == "raw_geometry_small") & (micro.metric == metric)].sort_values("seed")
        values = 100 * part.delta.to_numpy(); mean, sd = stats(values)
        lines.append(f"| {metric} | {values[0]:+.3f} | {values[1]:+.3f} | {values[2]:+.3f} | {mean:+.3f} ± {sd:.3f} |")
    lines += ["", "## Failure-minus-control interaction", "", "| Metric | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |", "|---|---:|---:|---:|---:|"]
    for metric in display_target:
        part = interactions[interactions.metric == metric].sort_values("seed")
        values = 100 * part.interaction.to_numpy(); mean, sd = stats(values)
        lines.append(f"| {metric} | {values[0]:+.3f} | {values[1]:+.3f} | {values[2]:+.3f} | {mean:+.3f} ± {sd:.3f} |")
    lines += ["", "The official table uses original COCO annotations and default one-to-one post-training best.pt predictions. Training logs confirm that Ultralytics explicitly ran final_eval on best.pt before writing predictions.json. The target table uses one-to-many + NMS best checkpoints. They answer different branch-specific questions and are not combined into a single effect estimate.", ""]
    (root / "RICH_EVALUATION_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
