"""Summarize the frozen OGPS Gate-0 replay; never train or run a model.

Input is PER_CANDIDATE.jsonl with permanent candidate identity, box_iou,
oracle_matched, and iou/coverage/auc/fpr/mask75_<ARM> fields. Null O values
remain missing. Statistics resample images, not individual candidates.

Usage: python summarize_ogps.py --run <run-directory> [--out <directory>]
       python summarize_ogps.py --input <jsonl> --out <directory>

API: summarize(rows, out, seed=0, bootstrap=5000, source_path=None) -> dict
Only NumPy and the standard library are used. The source JSONL is not rewritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ARMS = ("A", "O", "G_GTsolver", "S_BASE", "T_TRUE", "N_BASEH", "W_INST", "W_IMAGE",
        "D_TRUE", "D_BASEH", "D_INST", "D_IMAGE", "D_GT")
ARM_LABELS = {
    "A": "原模型系数与原型", "O": "已有有限GT oracle（身份匹配者）",
    "G_GTsolver": "GT 8×8约束经固定ridge求解", "S_BASE": "原模型自伪标签经相同求解",
    "T_TRUE": "旧7O真实ownership输出经求解", "N_BASEH": "旧7O空ROI/base-h输出经求解",
    "W_INST": "旧7O wrong-instance独立训练组输出经求解",
    "W_IMAGE": "旧7O wrong-image独立训练组输出经求解",
    "D_TRUE": "真实ownership直接上采样", "D_BASEH": "base-h输出直接上采样",
    "D_INST": "wrong-instance输出直接上采样", "D_IMAGE": "wrong-image输出直接上采样",
    "D_GT": "GT soft grid直接上采样",
}
METRICS = ("iou", "mask75", "coverage", "auc", "fpr")
PRIMARY_PAIRS = (("T_TRUE", "A"), ("T_TRUE", "D_TRUE"), ("T_TRUE", "W_INST"),
                 ("T_TRUE", "W_IMAGE"), ("T_TRUE", "S_BASE"))
ALL_PAIRS = tuple(dict.fromkeys(PRIMARY_PAIRS + tuple((a, "A") for a in ARMS if a != "A")
                               + (("G_GTsolver", "D_GT"), ("N_BASEH", "D_BASEH"),
                                  ("W_INST", "D_INST"), ("W_IMAGE", "D_IMAGE"))))
EXPECTED_IMAGES, EXPECTED_CANDIDATES = 196, 1346


def finite(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def mean(values):
    values = [float(v) for v in values if finite(v)]
    return sum(values) / len(values) if values else None


def arr(values):
    return np.asarray([float(v) if finite(v) else np.nan for v in values], dtype=np.float64)


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(values)
    return np.quantile(values[ok], [0.025, 0.975]).tolist() if ok.any() else [None, None]


def image_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row.get("split", "val")), int(row["image_id"]))].append(row)
    return [groups[key] for key in sorted(groups)]


def value(row, metric, arm):
    if arm == "O" and row.get("oracle_matched") is not True:
        return None
    # Mask75 comes from the same continuous original-image IoU, including empties.
    if metric == "mask75":
        iou = row.get(f"iou_{arm}")
        return int(float(iou) >= 0.75) if finite(iou) else None
    return row.get(f"{metric}_{arm}")


def paired_stats(groups, arm, ref, metric, seed, bootstrap):
    """Each pair uses the same finite candidates within every sampled image."""
    sums_a, sums_b, counts = [], [], []
    for one in groups:
        a = arr([value(r, metric, arm) for r in one])
        b = arr([value(r, metric, ref) for r in one])
        valid = np.isfinite(a) & np.isfinite(b)
        sums_a.append(float(a[valid].sum()))
        sums_b.append(float(b[valid].sum()))
        counts.append(int(valid.sum()))
    sa, sb, counts = np.asarray(sums_a), np.asarray(sums_b), np.asarray(counts)
    good = counts > 0
    ma = np.divide(sa, counts, out=np.zeros_like(sa), where=good)
    mb = np.divide(sb, counts, out=np.zeros_like(sb), where=good)
    delta = ma - mb
    macro_draws, candidate_draws = [], []
    if len(groups):
        rng = np.random.default_rng(seed)
        for start in range(0, bootstrap, 256):
            draw = rng.integers(0, len(groups), size=(min(256, bootstrap-start), len(groups)))
            nm, nc = good[draw].sum(1), counts[draw].sum(1)
            macro_draws.extend(np.divide(delta[draw].sum(1), nm, out=np.full(len(draw), np.nan), where=nm > 0))
            candidate_draws.extend(np.divide((sa-sb)[draw].sum(1), nc, out=np.full(len(draw), np.nan), where=nc > 0))
    total = sum(len(one) for one in groups)
    return {
        "status": "complete_pairs" if int(counts.sum()) == total and total else "missing_or_undefined_pairs",
        "image_macro": {"arm_mean": mean(ma[good]), "reference_mean": mean(mb[good]),
                        "delta": mean(delta[good]), "ci95": interval(macro_draws), "valid_images": int(good.sum())},
        "candidate": {"arm_mean": float(sa.sum()/counts.sum()) if counts.sum() else None,
                      "reference_mean": float(sb.sum()/counts.sum()) if counts.sum() else None,
                      "delta": float((sa-sb).sum()/counts.sum()) if counts.sum() else None,
                      "ci95": interval(candidate_draws), "valid_candidates": int(counts.sum())},
        "missing_or_undefined_candidates": total-int(counts.sum()),
        "empty_bootstrap_draws": int(np.isnan(macro_draws).sum()),
    }


def table(rows, seed, bootstrap, role):
    groups = image_groups(rows)
    result = {"n_images": len(groups), "n_candidates": len(rows), "role": role,
              "candidate": {}, "image_macro": {}, "undefined": {}, "comparisons": {}, "mask75_counts": {}}
    for metric in METRICS:
        result["candidate"][metric] = {a: mean([value(r, metric, a) for r in rows]) for a in ARMS}
        result["image_macro"][metric] = {a: mean([mean([value(r, metric, a) for r in one]) for one in groups]) for a in ARMS}
        result["undefined"][metric] = {a: sum(not finite(value(r, metric, a)) for r in rows) for a in ARMS}
    for arm, ref in ALL_PAIRS:
        comparison = {metric: paired_stats(groups, arm, ref, metric, seed, bootstrap) for metric in METRICS}
        paired = [r for r in rows if finite(value(r, "mask75", arm)) and finite(value(r, "mask75", ref))]
        repair = sum(value(r, "mask75", arm) == 1 and value(r, "mask75", ref) == 0 for r in paired)
        damage = sum(value(r, "mask75", arm) == 0 and value(r, "mask75", ref) == 1 for r in paired)
        comparison["mask75_transition"] = {
            "repair": repair, "damage": damage, "net": repair-damage,
            "valid_candidates": len(paired), "missing_candidates": len(rows)-len(paired),
            "reference_success": sum(value(r, "mask75", ref) == 1 for r in paired),
            "reference_failure": sum(value(r, "mask75", ref) == 0 for r in paired),
            "net_candidate_fraction": (repair-damage)/len(paired) if paired else None,
            "net_ci95_candidate_fraction": comparison["mask75"]["candidate"]["ci95"],
            "net_ci95_image_macro_fraction": comparison["mask75"]["image_macro"]["ci95"],
        }
        result["comparisons"][f"{arm}_minus_{ref}"] = comparison
    for arm in ARMS:
        valid = [value(r, "mask75", arm) for r in rows if finite(value(r, "mask75", arm))]
        result["mask75_counts"][arm] = {"success": int(sum(valid)), "defined": len(valid), "undefined": len(rows)-len(valid)}
    return result


def ratio_of_gains(rows, seed, bootstrap, require_complete=True):
    """Ratio of IMAGE-MACRO gains on ONE shared subgroup, never mean of ratios."""
    missing = [r for r in rows if r.get("oracle_matched") is not True or
               not all(finite(r.get(f"iou_{a}")) for a in ("A", "O", "G_GTsolver"))]
    result = {"definition": "macro(IoU_G_GTsolver-IoU_A) / macro(IoU_O-IoU_A) on the same fixed candidates and images",
              "n_candidates": len(rows), "n_images": len(image_groups(rows)), "missing_or_unmatched_candidates": len(missing),
              "status": "unknown", "ratio": None, "ci95": [None, None], "numerator": None, "denominator": None,
              "bootstrap_nonpositive_denominators": None, "complete_population_required": require_complete}
    if not rows:
        result["reason"] = "No box-good original-failure candidates"
        return result
    if missing and require_complete:
        result["reason"] = "Required subgroup contains missing or identity-unmatched oracle/solver records; no silent complete-case gate"
        return result
    missing_ids = {id(r) for r in missing}
    selected = [r for r in rows if id(r) not in missing_ids]
    if not selected:
        result["reason"] = "No complete matched candidates"
        return result
    groups = image_groups(selected)
    numerator = np.asarray([mean([r["iou_G_GTsolver"]-r["iou_A"] for r in one]) for one in groups])
    denominator = np.asarray([mean([r["iou_O"]-r["iou_A"] for r in one]) for one in groups])
    num, den = float(numerator.mean()), float(denominator.mean())
    result.update(numerator=num, denominator=den, matched_candidates=len(selected), matched_images=len(groups))
    if not math.isfinite(den) or den <= 0:
        result["reason"] = "Oracle macro gain is nonpositive or undefined"
        return result
    rng = np.random.default_rng(seed)
    ratios, num_draws, den_draws = [], [], []
    for start in range(0, bootstrap, 256):
        draw = rng.integers(0, len(groups), size=(min(256, bootstrap-start), len(groups)))
        a, b = numerator[draw].mean(1), denominator[draw].mean(1)
        num_draws.extend(a)
        den_draws.extend(b)
        ratios.extend(np.divide(a, b, out=np.full(len(draw), np.nan), where=b > 0))
    invalid = int(np.sum(np.asarray(den_draws) <= 0))
    result.update(ratio=num/den, numerator_ci95=interval(num_draws), denominator_ci95=interval(den_draws),
                  bootstrap_nonpositive_denominators=invalid)
    if invalid:
        # Do not condition the CI on a favorable positive-denominator bootstrap.
        result["reason"] = "At least one bootstrap oracle denominator is nonpositive; ratio CI/gate unresolved, no invalid draws discarded"
        return result
    result.update(status="defined", ci95=interval(ratios), reason="Same-subset image-cluster paired ratio bootstrap")
    return result


def positive_rule(stat, threshold=0.0, inclusive=False, require_ci=True):
    point, ci = stat.get("delta"), stat.get("ci95", [None, None])
    if not finite(point) or (require_ci and not all(finite(v) for v in ci)):
        return None
    point_pass = point >= threshold if inclusive else point > threshold
    return bool(point_pass and (not require_ci or ci[0] > 0))


def decide(results, issues):
    ga = results["gate0A"]
    if ga["status"] != "defined":
        a_state = "UNKNOWN"
    elif ga["ratio"] < 0.5:
        a_state = "STOP"
    elif ga["ratio"] < 0.7:
        a_state = "GRAY"
    else:
        a_state = "PROCEED_TO_GATE0B"
    comparisons = results["tables"]["val:all"]["comparisons"]
    stats = {f"{a}_minus_{b}": comparisons[f"{a}_minus_{b}"]["iou"]["image_macro"] for a, b in PRIMARY_PAIRS}
    rules = {
        "true_minus_A_positive_point": positive_rule(stats["T_TRUE_minus_A"], require_ci=False),
        "true_minus_direct_at_least_0_002_and_positive_ci": positive_rule(stats["T_TRUE_minus_D_TRUE"], threshold=0.002, inclusive=True),
        "true_minus_wrong_instance_positive_ci": positive_rule(stats["T_TRUE_minus_W_INST"]),
        "true_minus_wrong_image_positive_ci": positive_rule(stats["T_TRUE_minus_W_IMAGE"]),
        "true_minus_self_pseudo_positive_ci": positive_rule(stats["T_TRUE_minus_S_BASE"]),
    }
    transition = comparisons["T_TRUE_minus_A"]["mask75_transition"]
    rules["mask75_repair_greater_than_damage_vs_A"] = transition["repair"] > transition["damage"] if transition["missing_candidates"] == 0 else None
    for arm, ref in PRIMARY_PAIRS:
        if comparisons[f"{arm}_minus_{ref}"]["iou"]["status"] != "complete_pairs":
            issues.append(f"Incomplete primary IoU pairs: {arm}-{ref}")
    if issues:
        decision, reason = "INCOMPLETE_EVIDENCE", "候选范围或必要指标不完整；保留结果，不能放行。"
    elif a_state == "UNKNOWN":
        decision, reason = "STOP_GATE0A_UNKNOWN", "0A的同子集oracle比值未定义或不稳定，不能作可行性判断。"
    elif a_state == "STOP":
        decision, reason = "STOP_GATE0A", "本固定8×8 ridge版本回收的oracle增益比例低于0.5，按预设条件停止。"
    elif a_state == "GRAY":
        decision, reason = "STOP_GATE0A_GRAY", "本固定8×8 ridge版本回收比例处于0.5–0.7灰区，不放行训练。"
    elif any(v is False for v in rules.values()):
        decision, reason = "STOP_GATE0B", "至少一项固定GT-free效用对照未通过；零训练回放在此结案。"
    elif any(v is None for v in rules.values()):
        decision, reason = "STOP_GATE0B_UNKNOWN", "GT-free效用对照存在未知项，不能放行。"
    else:
        decision, reason = "GATE0_REPLAY_SUPPORTED", "固定零训练回放通过预设判别条件；仅形成后续方案讨论依据，不自动训练。"
    return {
        "decision": decision, "reason": reason, "gate0A_state": a_state, "gate0B_rules": rules,
        "failed_rules": [k for k, v in rules.items() if v is False],
        "unknown_rules": [k for k, v in rules.items() if v is None],
        "training_executed": False, "training_authorized_by_this_summary": False,
        "gate0B_primary_comparisons": stats, "T_TRUE_vs_A_mask75": transition,
        "integrity_issues": issues,
        "decision_definitions": {
            "gate0A": "ratio point <0.5 STOP; [0.5,0.7) GRAY; >=0.7 evaluates 0B; ratio CI reported, not substituted for point rule",
            "gate0B_baseline": "T_TRUE-A macro IoU point >0; CI reported without adding an unregistered baseline-CI gate",
            "gate0B_direct": "T_TRUE-D_TRUE macro IoU >=0.002 and paired CI lower>0",
            "gate0B_wrong_and_self": "T_TRUE-W_INST, T_TRUE-W_IMAGE, T_TRUE-S_BASE: paired macro IoU CI lower>0",
            "gate0B_mask75": "T_TRUE repairs>damage relative to A on all candidates; image-cluster net intervals also reported",
        },
    }


def fmt(x, pp=False, signed=False):
    if not finite(x):
        return "未知"
    value = float(x)*(100 if pp else 1)
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def fmt_effect(stat, pp=True):
    ci = stat.get("ci95", [None, None])
    return f"{fmt(stat.get('delta'),pp,True)} [{fmt(ci[0],pp,True)}, {fmt(ci[1],pp,True)}]"


def summarize(rows, out, seed=0, bootstrap=5000, source_path=None):
    rows, out = list(rows), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    issues = []
    val = [r for r in rows if r.get("split") == "val"]
    if len(val) != len(rows):
        issues.append("Input includes non-val rows; only declared val is summarized")
    required = ("split", "image_id", "annotation_id", "raw_id", "branch", "pyramid_level", "target_gt_idx")
    for j, row in enumerate(val):
        missing = [k for k in required if k not in row]
        if missing:
            raise ValueError(f"Row {j} lacks permanent identity fields: {missing}")
    keys = [tuple(r[k] for k in required) for r in val]
    if len(keys) != len(set(keys)):
        issues.append("Duplicate permanent candidate identities; no duplicate is silently removed")
    groups = image_groups(val)
    if (len(groups), len(val)) != (EXPECTED_IMAGES, EXPECTED_CANDIDATES):
        issues.append(f"Frozen val scope is 196/1346; observed {len(groups)}/{len(val)}")
    for arm in ARMS:
        if arm != "O" and any(not finite(r.get(f"iou_{arm}")) for r in val):
            issues.append(f"Non-oracle arm {arm} has missing/nonfinite IoU; candidates retained")
    missing_box = sum(not finite(r.get("box_iou")) for r in val)
    if missing_box:
        issues.append(f"{missing_box} candidates lack box_iou; Gate0A membership unresolved")
    flag_inconsistency = []
    for arm in ARMS:
        bad = sum(finite(r.get(f"mask75_{arm}")) and finite(r.get(f"iou_{arm}")) and
                  bool(r[f"mask75_{arm}"]) != bool(r[f"iou_{arm}"] >= 0.75) for r in val)
        if bad:
            flag_inconsistency.append({"arm": arm, "count": bad})
            issues.append(f"Mask75 flag inconsistent with original IoU: {arm}, {bad}")
    box_fail = [r for r in val if finite(r.get("box_iou")) and r["box_iou"] >= 0.75 and
                finite(r.get("iou_A")) and r["iou_A"] < 0.75]
    selected = {"val:all": val, "val:box_good_original_failure": box_fail}
    selected.update({f"val:P{lev+3}": [r for r in val if r["pyramid_level"] == lev] for lev in range(3)})
    selected.update({f"val:original_{name}": [r for r in val if finite(r.get("iou_A")) and
                     (r["iou_A"] >= 0.75) == flag] for name, flag in (("success", True), ("failure", False))})
    if any("size_group" in r for r in val):
        selected.update({f"val:size_{name}": [r for r in val if r.get("size_group") == name] for name in ("small", "medium", "large")})
    tables = {key: table(one, seed, bootstrap, "primary" if key == "val:all" else "descriptive_fixed_stratum") for key, one in selected.items()}
    ratio = ratio_of_gains(box_fail, seed, bootstrap)
    if missing_box:
        ratio.update(status="unknown", reason="Missing box_iou prevents complete prespecified subgroup identification")
    per_image = []
    for one in groups:
        rr = {"split": "val", "image_id": one[0]["image_id"], "n_candidates": len(one)}
        for arm in ARMS:
            for metric in METRICS:
                rr[f"{metric}_{arm}"] = mean([value(r, metric, arm) for r in one])
                rr[f"{metric}_{arm}_defined_candidates"] = sum(finite(value(r, metric, arm)) for r in one)
        per_image.append(rr)
    source = {"path": str(Path(source_path).resolve()), "sha256": hashlib.sha256(Path(source_path).read_bytes()).hexdigest()} if source_path else None
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "arms": list(ARMS), "arm_labels": ARM_LABELS,
        "seed": seed, "bootstrap": bootstrap, "bootstrap_unit": "paired whole image clusters",
        "main_metric": "val all-candidate original-image Mask IoU, image macro; candidate mean secondary",
        "ci_definition": "pointwise percentile 95% paired image bootstrap; not familywise-adjusted",
        "units": "fractions; multiply deltas by100 for percentage points",
        "fixed_subgroup": "A MaskIoU<0.75 and frozen box_iou>=0.75; no post-correction filtering",
        "gate0A": ratio, "gate0A_matched_only_descriptive": ratio_of_gains(box_fail, seed, bootstrap, require_complete=False),
        "tables": tables, "source": source, "integrity_issues": issues,
        "mask75_flag_inconsistencies": flag_inconsistency,
        "limitations": [
            "本轮零训练，仅固定官方TAL候选回放；不是完整推理输出或COCO AP。",
            "0A只检验当前固定8×8 grid和ridge求解版本，不代表所有ownership投影或系数求解均不可行。",
            "0A强而0B弱不证明neck特征在信息论上缺少实例归属信息。",
            "W_INST/W_IMAGE来自旧7O独立训练组，不是同一head在线交换输入的纯因果对照。",
            "GT solver和finite oracle是GT辅助参照；GT-free arms推理不使用评价GT。",
            "finite oracle是已有有限正则目标的数值参照，不是IoU严格上界；0A增益比不截断为[0,1]，不解释为认证可恢复比例。",
            "AUC使用原评价定义的连续logit和相同支持；AUC未定义时保留该候选的IoU等指标。",
            "FPR是固定预测框内非目标像素误报率，可能包含邻居，不能单独称纯背景泄漏。",
            "全部固定replay可在0A停止时仍输出供记账；这不构成后续训练授权。",
        ],
    }
    decision = decide(result, issues)
    write_json(out/"RESULTS.json", result)
    write_json(out/"DECISION.json", decision)
    with (out/"PER_IMAGE.jsonl").open("w", encoding="utf-8") as stream:
        for row in per_image:
            stream.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False)+"\n")
    fields = ("group", "arm", "n_images", "n_candidates", "macro_iou", "candidate_iou", "macro_delta_vs_A",
              "macro_delta_ci_low", "macro_delta_ci_high", "repair_vs_A", "damage_vs_A", "net_vs_A",
              "macro_coverage", "macro_auc", "macro_fpr", "undefined_iou", "undefined_auc")
    with (out/"ABLATION_TABLE.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, item in tables.items():
            for arm in ARMS:
                rr = {"group": name, "arm": arm, "n_images": item["n_images"], "n_candidates": item["n_candidates"],
                      "macro_iou": item["image_macro"]["iou"][arm], "candidate_iou": item["candidate"]["iou"][arm],
                      "undefined_iou": item["undefined"]["iou"][arm], "undefined_auc": item["undefined"]["auc"][arm]}
                rr.update({f"macro_{m}": item["image_macro"][m][arm] for m in ("coverage", "auc", "fpr")})
                if arm != "A":
                    pair = item["comparisons"][f"{arm}_minus_A"]
                    stat = pair["iou"]["image_macro"]
                    rr.update(macro_delta_vs_A=stat["delta"], macro_delta_ci_low=stat["ci95"][0], macro_delta_ci_high=stat["ci95"][1])
                    rr.update({f"{key}_vs_A": pair["mask75_transition"][key] for key in ("repair", "damage", "net")})
                writer.writerow(rr)
    lines = ["# OGPS Gate 0 零训练回放", "", f"**{decision['decision']}：{decision['reason']}**", "",
             f"范围：{len(groups)}张有效val图、{len(val)}个候选；预设为196图/1346候选。主指标为图片macro原图IoU。", "",
             "## 组别", ""] + [f"- {a}：{ARM_LABELS[a]}。" for a in ARMS]
    lines += ["", "## Gate 0A：GT约束求解的可行性", "",
              f"固定框好原失败子集：{ratio['n_images']}图/{ratio['n_candidates']}候选；缺失或不匹配oracle/solver为{ratio['missing_or_unmatched_candidates']}个。",
              f"GT solver的macro增益：{fmt(ratio['numerator'],True,True)} pp；finite oracle同子集macro增益：{fmt(ratio['denominator'],True,True)} pp。",
              f"两项macro增益之比：{fmt(ratio['ratio'])}，95% CI [{fmt(ratio['ci95'][0])}, {fmt(ratio['ci95'][1])}]；状态{ratio['status']}，判定{decision['gate0A_state']}。",
              "先在图内对共同候选求增益均值，再跨图平均、相除；绝不平均逐实例比率。任一必需oracle身份未匹配，则0A为unknown；matched-only结果仅描述。",
              "比值点估计<0.5停止；0.5–0.7为灰区；≥0.7才进入0B判别。区间如实报告，不将有利区间或子组替换原门槛。", "",
              "## Gate 0B：固定GT-free对照", "", "所有效应及区间均为百分点。", "",
              "| 对比 | 图片macro IoU差[95% CI] | 候选IoU差[95% CI] | 修复/损伤/净数 | 净Mask75候选比例差[95% CI] |",
              "|---|---:|---:|---:|---:|"]
    for a, b in PRIMARY_PAIRS:
        item = tables["val:all"]["comparisons"][f"{a}_minus_{b}"]
        tr = item["mask75_transition"]
        lines.append(f"| {a}−{b} | {fmt_effect(item['iou']['image_macro'])} | {fmt_effect(item['iou']['candidate'])} | {tr['repair']}/{tr['damage']}/{tr['net']:+d} | {fmt_effect(item['mask75']['candidate'])} |")
    lines += ["", "T_TRUE−A须点估计为正，区间单列；T_TRUE−D_TRUE须≥0.2 pp且区间下界>0；对W_INST、W_IMAGE和S_BASE均须区间下界>0；相对A须修复数>损伤数。",
              "T_TRUE未优于S_BASE时，不能证明新增ownership输出提供了额外作用。W_INST/W_IMAGE为独立训练组，因此其比较不单独识别纯错配因果。", "",
              "| 条件 | 通过状态 |", "|---|---|"]
    lines += [f"| {k} | {'未知' if v is None else '通过' if v else '未通过'} |" for k, v in decision["gate0B_rules"].items()]
    lines += ["", "## 全体掩码质量", "", "| 组 | macro IoU | 候选IoU | Mask75达标/有效 | Coverage | AUC | 框内非目标FPR |", "|---|---:|---:|---:|---:|---:|---:|"]
    overall = tables["val:all"]
    for a in ARMS:
        n = overall["mask75_counts"][a]
        lines.append(f"| {a} | {fmt(overall['image_macro']['iou'][a],True)} | {fmt(overall['candidate']['iou'][a],True)} | {n['success']}/{n['defined']} | {fmt(overall['image_macro']['coverage'][a],True)} | {fmt(overall['image_macro']['auc'][a],True)} | {fmt(overall['image_macro']['fpr'][a],True)} |")
    lines += ["", "所有O的绝对值基于实际匹配且有效者；其有效数量与其他组不同时不得直接比较未配对绝对均值，配对结果另列。P3/P4/P5、原成功/失败及可用面积分层只作描述，见RESULTS.json与ABLATION_TABLE.csv。",
              "", "## 范围与结束", ""] + ["- "+text for text in result["limitations"]]
    lines += ["", f"95%区间为{bootstrap}次整图配对percentile bootstrap，未进行多比较校正。通过与否不触发训练或新实验。"]
    if issues:
        lines += ["", "## 未完整证据", ""] + ["- "+issue for issue in issues]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if args.input is None and args.run is None:
        parser.error("Provide --run or --input")
    source = args.input or args.run/"PER_CANDIDATE.jsonl"
    out = args.out or args.run or source.parent
    with source.open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    summarize(rows, out, args.seed, args.bootstrap, source)
    decision = json.loads((out/"DECISION.json").read_text(encoding="utf-8"))
    print(json.dumps({"decision": decision["decision"], "reason": decision["reason"], "training_executed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
