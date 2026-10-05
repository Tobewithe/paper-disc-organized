"""Build manuscript tables and vector/raster figures from recorded experiments.

Run in the project's conda pytorch environment. No model inference or training.
Existing experiment records are read only. Outputs stay beside this script.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DIAG = ROOT / "experiments"
PILOT = ROOT / "experiments/counterfactual_p3_distillation_20260914"


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def select(rows, **keys):
    matches = [r for r in rows if all(r[k] == str(v) for k, v in keys.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for {keys}, got {len(matches)}")
    return matches[0]


def ci(row, value="delta", lo="ci_low", hi="ci_high"):
    return f"{100*float(row[value]):+.3f} [{100*float(row[lo]):+.3f}, {100*float(row[hi]):+.3f}]"


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def savefig(fig, name):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / "figures" / f"{name}.{suffix}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def forest(ax, labels, rows, value="delta", low="ci_low", high="ci_high", color="#176b87"):
    vals = np.array([float(r[value]) for r in rows])*100
    los = np.array([float(r[low]) for r in rows])*100
    his = np.array([float(r[high]) for r in rows])*100
    y = np.arange(len(rows))
    ax.errorbar(vals, y, xerr=np.stack([vals-los, his-vals]), fmt="o", color=color, ecolor=color, capsize=3, markersize=5, lw=1.4)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#9aa5ae", ls="--", lw=1)
    ax.grid(axis="x", color="#e7ebef", lw=.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Change in metric × 100 (95% CI)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main():
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    lineage = read(DIAG / "no_final_slot_lineage_20260914/summary.csv")
    interventions = read(DIAG / "neighbor_specific_recovery_20260915/paired_contrasts.csv")
    transplants = read(DIAG / "small_object_causal_interventions_20260914/feature_localization_summary.csv")
    effects = read(PILOT / "rich_across_seed_target_effects.csv")
    official = read(PILOT / "rich_eval_saved/official_metrics.csv")
    assert len(official) == 12
    stages = [
        ("raw_geometry", "Raw geometry", "原始框几何不合格"),
        ("correct_class", "Correct class", "类别筛选后不再可用"),
        ("score_0001", "Confidence 0.001", "置信度筛选后不再可用"),
        ("nms", "NMS", "NMS 后不再可用"),
        ("top300", "Top 300", "每图前 300 个后不再可用"),
        ("eval100", "Evaluation limit 100/category", "每类评估上限 100 后不再可用"),
        ("assignment_competition", "Final matching competition", "最终匹配竞争"),
    ]
    lr = [select(lineage, scope="current_unmatched", first_loss=k) for k, _, _ in stages]
    assert sum(int(r["count"]) for r in lr) == 3699
    edit_specs = [
        ("neighbor_flat", "background_flat", "Neighbor fill − matched-shape background fill", "邻居填色 − 等形背景填色"),
        ("neighbor_blur", "background_blur", "Neighbor blur − matched-shape background blur", "邻居模糊 − 等形背景模糊"),
        ("neighbor_flat", "distant_instance_flat", "Neighbor fill − distant-object fill", "邻居填色 − 远端实例填色"),
    ]
    # The CSV is authoritative about the arm spelling.
    distant_names = sorted({r["control"] for r in interventions if "distant" in r["control"]})
    if len(distant_names) != 1:
        raise ValueError(distant_names)
    edit_specs[-1] = ("neighbor_flat", distant_names[0], edit_specs[-1][2], edit_specs[-1][3])
    er = [select(interventions, cohort="raw_geometry_small", treatment=t, control=c, metric="raw_p3_best_box_iou") for t, c, _, _ in edit_specs]
    transplant_specs = [
        ("edited", "Full joint edited view", "联合编辑整图"),
        ("p3_full", "P3, full tensor", "替换完整 P3"),
        ("p3_local", "P3, local support", "替换局部 P3"),
        ("p4_full", "P4, full tensor", "替换完整 P4"),
        ("p5_full", "P5, full tensor", "替换完整 P5"),
    ]
    tr = [select(transplants, cohort="raw_geometry_small", arm=k) for k, _, _ in transplant_specs]
    metric_specs = [
        ("raw_p3_best_box_iou", "Raw P3 box IoU ↑", "原始 P3 框 IoU ↑"),
        ("raw_best_box_iou", "Raw all-level box IoU ↑", "全部层级原始框 IoU ↑"),
        ("raw_center_error_norm", "Raw all-level center error ↓", "全部层级最佳框中心误差 ↓"),
        ("raw_box50", "Raw all-level Box50 rate ↑", "原始 Box50 可用率 ↑"),
        ("final_box_iou", "Selected final box IoU ↑", "所选最终框 IoU ↑"),
        ("mask_iou", "Associated final mask IoU ↑", "关联最终掩码 IoU ↑"),
        ("target_coverage", "Target coverage ↑", "目标覆盖率 ↑"),
        ("prediction_purity", "Prediction purity ↑", "预测纯度 ↑"),
        ("boundary_f1", "Boundary F1 ↑", "边界 F1 ↑"),
        ("same_neighbor_leak_pred", "Same-category neighbor leakage ↓", "同类邻居泄漏率 ↓"),
        ("background_leak_pred", "Background leakage ↓", "背景泄漏率 ↓"),
        ("final_mask75", "Associated Mask75 rate ↑", "关联掩码 Mask75 比例 ↑"),
    ]
    pr = [select(effects, analysis="cohort", cohort="raw_geometry_small", metric=k) for k, _, _ in metric_specs]
    assert all(int(r["n"]) == 239 for r in pr)
    official_specs = [("bbox", "ap", "Box AP"), ("segm", "ap", "Mask AP"), ("segm", "ap50", "Mask AP50"), ("segm", "ap75", "Mask AP75"), ("segm", "aps", "Mask AP small"), ("bbox", "ar_max100", "Box AR100"), ("segm", "ar_max100", "Mask AR100")]
    of_rows = []
    for task, metric, label in official_specs:
        b = np.array([float(select(official, arm=f"baseline_s{s}", task=task)[metric]) for s in range(3)])*100
        m = np.array([float(select(official, arm=f"cfp3r_s{s}", task=task)[metric]) for s in range(3)])*100
        of_rows.append([label, f"{b.mean():.3f}", f"{m.mean():.3f}", f"{(m-b).mean():+.3f} ± {(m-b).std(ddof=1):.3f}"])

    for lang, filename in (("en", "PAPER_DRAFT.md"), ("zh", "PAPER_DRAFT_ZH.md")):
        label_index = 1 if lang == "en" else 2
        texts = {
            "lineage": md_table(["First unavailable stage", "GT count", "% of unmatched"] if lang == "en" else ["首次不再有合格候选的阶段", "GT 数量", "占未匹配 GT"], [[s[label_index], r["count"], f"{float(r['fraction'])*100:.2f}%"] for s, r in zip(stages, lr)]),
            "intervention": md_table(["Comparison", "n", "Δ P3 IoU [95% CI]"] if lang == "en" else ["比较", "n", "P3 IoU 差值 [95% CI]"], [[s[2 if lang == "en" else 3], r["n"], ci(r, "paired_net")] for s, r in zip(edit_specs, er)]),
            "transplant": md_table(["Donor / replacement", "Δ raw IoU [95% CI]", "Recovered / 384"] if lang == "en" else ["输入/特征替换方式", "原始 IoU 增益 [95% CI]", "恢复数 / 384"], [[s[label_index], ci(r, "best_any_box_iou_delta", "best_any_box_iou_ci_low", "best_any_box_iou_ci_high"), r["box50_recovered"]] for s, r in zip(transplant_specs, tr)]),
            "pilot": md_table(["Metric", "Δ [95% CI]"] if lang == "en" else ["指标", "方法 − 基线 [95% CI]"], [[s[label_index], ci(r)] for s, r in zip(metric_specs, pr)]),
            "official": md_table(["Metric", "Baseline mean", "IG-P3 mean", "Paired Δ ± seed SD"] if lang == "en" else ["指标", "基线均值", "IG-P3 均值", "配对差值 ± 种子标准差"], of_rows),
        }
        path = OUT / filename
        if not path.exists():
            raise FileNotFoundError(path)
        body = path.read_text(encoding="utf-8")
        for key, table in texts.items():
            pattern = rf"<!-- generated:{key}:start -->.*?<!-- generated:{key}:end -->"
            replacement = f"<!-- generated:{key}:start -->\n{table}\n<!-- generated:{key}:end -->"
            body, n = re.subn(pattern, lambda _: replacement, body, flags=re.S)
            if n != 1:
                raise ValueError(f"{filename}: expected one {key} block, got {n}")
            (OUT / "tables" / f"{key}_{lang}.md").write_text(table + "\n", encoding="utf-8")
        path.write_text(body, encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 3, figsize=(15.8, 4.1), gridspec_kw={"width_ratios": [1.0, 1.15, 1.2]})
    y = np.arange(len(lr))
    vals = [int(r["count"]) for r in lr]
    axs[0].barh(y, vals, color=["#176b87"]*3+["#aab7c4"]*4)
    axs[0].set_yticks(y, ["Raw geometry", "Correct class", "Confidence", "NMS", "Top 300", "Eval limit", "Competition"])
    axs[0].invert_yaxis()
    for yy, val in zip(y, vals):
        axs[0].text(val+20, yy, f"{val:,}", va="center", fontsize=9)
    axs[0].set_xlim(0, 1800)
    axs[0].set_xlabel("GT instances (3,699 unmatched)")
    axs[0].set_title("(a) Candidate availability")
    axs[0].spines[["top", "right"]].set_visible(False)
    forest(axs[1], ["Neighbor − background\nfill, n = 302", "Neighbor − background\nblur, n = 302", "Neighbor − distant\nfill, n = 299"], er, "paired_net")
    axs[1].set_title("(b) Single-neighbor controls")
    axs[1].set_xlabel("Raw P3 IoU change × 100")
    forest(axs[2], ["Joint edited view", "Full P3", "Local P3", "Full P4", "Full P5"], tr, "best_any_box_iou_delta", "best_any_box_iou_ci_low", "best_any_box_iou_ci_high", "#ba6b21")
    axs[2].set_title("(c) Joint-edit feature donor")
    axs[2].set_xlabel("All-level raw IoU change × 100")
    fig.tight_layout(w_pad=2.4)
    savefig(fig, "diagnostic_evidence")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    show = [0, 1, 4, 5, 6, 7, 8]
    forest(ax, [metric_specs[i][1].replace(" ↑", "") for i in show], [pr[i] for i in show])
    ax.set_title("Three-seed pilot: frozen localization-failure cohort (n = 239)", pad=16)
    ax.set_xlabel("IG-P3 − baseline, metric × 100\n95% bootstrap CI after per-instance averaging across seeds")
    fig.tight_layout()
    savefig(fig, "pilot_effects")

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.5); ax.axis("off")
    def box(x, y, w, h, text, color="#e7f1f5"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10", facecolor=color, edgecolor="#537181", linewidth=1.1))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=10)
    def arrow(a, b, label="", dashed=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.1, color="#537181", linestyle="--" if dashed else "-"))
        if label:
            ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+.13, label, ha="center", fontsize=9)
    box(.2, 3.4, 2.1, .9, "Ordinary augmented\ntraining image x")
    box(3.1, 3.4, 2.2, .9, "Student\noriginal view")
    box(.2, .65, 2.1, 1.15, "Training GT\nboxes + masks", "#f6eee3")
    box(3.1, .65, 2.2, 1.15, "Edited teacher view\nneighbor weakening\n+ target contrast")
    box(6.1, .65, 2.1, 1.15, "Frozen teacher\nnominate P3\nlocation k* ")
    box(6.1, 3.4, 2.1, .9, "Student P3 box\nand class at k*")
    box(9.05, 2.45, 3.4, 1.4, "Location gate\nstudent IoU < 0.5\nteacher IoU ≥ 0.5\nquality gap ≥ 0.1", "#f6eee3")
    box(9.05, .5, 3.4, 1.2, "Additional GT box loss\n+ small true-class loss\n(student gradients only)")
    arrow((2.4, 3.85), (3, 3.85)); arrow((5.4, 3.85), (6, 3.85))
    arrow((1.25, 3.3), (3.2, 1.9)); arrow((2.4, 1.25), (3, 1.25))
    arrow((5.4, 1.25), (6, 1.25)); arrow((8.3, 1.45), (8.95, 2.8))
    arrow((8.3, 3.85), (8.95, 3.5)); arrow((10.7, 2.35), (10.7, 1.8))
    arrow((7.15, 1.9), (7.15, 3.3), dashed=True)
    ax.text(7.32, 2.55, "select k*", fontsize=9, va="center")
    ax.plot([1.25, 1.25, 10.7], [.55, .16, .16], ls="--", color="#537181", lw=1.1)
    arrow((10.7, .16), (10.7, .4), dashed=True)
    ax.text(5.0, .24, "GT regression target", fontsize=9)
    ax.text(.2, 5.05, "IG-P3: edited-view teacher selects where and when to add supervision", fontsize=14, weight="bold")
    ax.text(.2, -.02, "Inference: original student only. GT supplies the regression target; the teacher box is not the target.", fontsize=10)
    savefig(fig, "method_overview")

    sources = ["diagnostics/no_final_slot_lineage_20260914/summary.csv", "diagnostics/neighbor_specific_recovery_20260915/paired_contrasts.csv", "diagnostics/small_object_causal_interventions_20260914/feature_localization_summary.csv", "experiments/counterfactual_p3_distillation_20260914/rich_across_seed_target_effects.csv", "experiments/counterfactual_p3_distillation_20260914/rich_eval_saved/official_metrics.csv"]
    (OUT / "tables/source_manifest.json").write_text(json.dumps({"purpose": "Traceable paper tables; no new inference", "sources": sources, "tables_per_language": 5, "figures": ["diagnostic_evidence", "method_overview", "pilot_effects"], "scale": "All plotted IoU and normalized-ratio changes are multiplied by 100"}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("Built 5 tables per manuscript, 3 figures in PNG/PDF/SVG, and source manifest.")


if __name__ == "__main__":
    main()
