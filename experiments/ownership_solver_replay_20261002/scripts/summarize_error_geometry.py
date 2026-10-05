"""Bounded, zero-training error-propagation diagnostic summary.

API: summarize(rows, out, seed=20261003, bootstrap=5000) -> results dict.
CLI: python summarize_error_geometry.py --run RUN [--out OUT]
     python summarize_error_geometry.py --input PER_CANDIDATE.jsonl --out OUT

Only NumPy and the standard library are used. This never modifies candidates,
fits a model, chooses intervention cells, or reruns inference. All candidates,
including empty masks and energy-matching failures, remain in the primary table.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ARMS = ("A", "T", "G", "H8", "E8", "R8", "EM8")
METRICS = ("iou", "coverage", "auc", "fpr", "mask75")
LABELS = {
    "A": "原模型", "T": "旧7O真实ownership经固定求解器",
    "G": "全部8×8 GT约束经同一求解器",
    "H8": "按连续logit传播能量贪心选择8格，用GT替换",
    "E8": "按最大绝对输入误差选择8格，用GT替换",
    "R8": "固定随机8格，用GT替换",
    "EM8": "4096个固定随机8格子集中匹配H8所移除输入误差能量者",
}
PAIRS = tuple(dict.fromkeys((("H8", "EM8"), ("H8", "E8"), ("H8", "T"))
                          + tuple((a, "A") for a in ARMS if a != "A")))
GEOMETRY = ("input_energy", "input_energy_removed_H8", "input_energy_removed_E8",
            "input_energy_removed_R8", "input_energy_removed_EM8", "full_mse_T",
            "full_mse_H8", "full_mse_E8", "full_mse_R8", "full_mse_EM8",
            "energy_match_relative_error", "perm_mse_mean",
            "actual_to_permutation_mse_ratio", "AR_spectral_norm", "n_support")
EXPECTED_IMAGES, EXPECTED_CANDIDATES = 196, 1346


def finite(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and math.isfinite(float(x))


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def write_json(path, x):
    Path(path).write_text(json.dumps(clean(x), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def avg(xs):
    xs = [float(x) for x in xs if finite(x)]
    return float(np.mean(xs)) if xs else None


def ci(xs):
    xs = np.asarray(xs, dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    return np.quantile(xs, [0.025, 0.975]).tolist() if len(xs) else [None, None]


def groups(rows):
    d = defaultdict(list)
    for r in rows:
        d[(str(r.get("split", "val")), str(r.get("image_id")))].append(r)
    return [d[k] for k in sorted(d)]


def val(row, metric, arm):
    # Derive Mask75 from the same original-image IoU. Empty-mask IoU=0 is valid.
    if metric == "mask75":
        x = row.get(f"iou_{arm}")
        return int(float(x) >= 0.75) if finite(x) else None
    return row.get(f"{metric}_{arm}")


def sample_indices(n, seed, bootstrap):
    rng = np.random.default_rng(seed)
    if n:
        for start in range(0, bootstrap, 256):
            yield rng.integers(0, n, (min(256, bootstrap-start), n))


def pair(one_groups, arm, ref, metric, seed, bootstrap):
    counts, sums_a, sums_b = [], [], []
    for one in one_groups:
        pairs = [(val(r, metric, arm), val(r, metric, ref)) for r in one]
        pairs = [(a, b) for a, b in pairs if finite(a) and finite(b)]
        counts.append(len(pairs))
        sums_a.append(sum(float(a) for a, _ in pairs))
        sums_b.append(sum(float(b) for _, b in pairs))
    n = np.asarray(counts, dtype=np.int64)
    a, b = np.asarray(sums_a), np.asarray(sums_b)
    defined = n > 0
    ma = np.divide(a, n, out=np.zeros(len(n)), where=defined)
    mb = np.divide(b, n, out=np.zeros(len(n)), where=defined)
    macro_draws, candidate_draws = [], []
    for draw in sample_indices(len(n), seed, bootstrap):
        ni, nc = defined[draw].sum(1), n[draw].sum(1)
        macro_draws.extend(np.divide((ma-mb)[draw].sum(1), ni, out=np.full(len(draw), np.nan), where=ni > 0))
        candidate_draws.extend(np.divide((a-b)[draw].sum(1), nc, out=np.full(len(draw), np.nan), where=nc > 0))
    total = sum(map(len, one_groups))
    return {
        "complete_pairs": bool(total and n.sum() == total),
        "missing_or_undefined_candidates": int(total-n.sum()),
        "image_macro": {"arm_mean": avg(ma[defined]), "reference_mean": avg(mb[defined]),
                        "delta": avg((ma-mb)[defined]), "ci95": ci(macro_draws),
                        "valid_images": int(defined.sum())},
        "candidate": {"arm_mean": float(a.sum()/n.sum()) if n.sum() else None,
                      "reference_mean": float(b.sum()/n.sum()) if n.sum() else None,
                      "delta": float((a-b).sum()/n.sum()) if n.sum() else None,
                      "ci95": ci(candidate_draws), "valid_candidates": int(n.sum())},
        "undefined_bootstrap_draws": int(np.sum(~np.isfinite(macro_draws))),
    }


def table(rows, seed, bootstrap, description):
    gs = groups(rows)
    t = {"description": description, "n_images": len(gs), "n_candidates": len(rows),
         "image_macro": {}, "candidate": {}, "undefined": {}, "comparisons": {}}
    for metric in METRICS:
        t["image_macro"][metric] = {a: avg([avg([val(r, metric, a) for r in g]) for g in gs]) for a in ARMS}
        t["candidate"][metric] = {a: avg([val(r, metric, a) for r in rows]) for a in ARMS}
        t["undefined"][metric] = {a: sum(not finite(val(r, metric, a)) for r in rows) for a in ARMS}
    for a, b in PAIRS:
        p = {metric: pair(gs, a, b, metric, seed, bootstrap) for metric in METRICS}
        paired = [r for r in rows if finite(val(r, "mask75", a)) and finite(val(r, "mask75", b))]
        repaired = sum(val(r, "mask75", a) == 1 and val(r, "mask75", b) == 0 for r in paired)
        damaged = sum(val(r, "mask75", a) == 0 and val(r, "mask75", b) == 1 for r in paired)
        p["mask75_transition"] = {
            "repair": repaired, "damage": damaged, "net": repaired-damaged,
            "valid_candidates": len(paired), "missing_candidates": len(rows)-len(paired),
            "reference_success": sum(val(r, "mask75", b) == 1 for r in paired),
            "reference_failure": sum(val(r, "mask75", b) == 0 for r in paired),
            "net_candidate_fraction": (repaired-damaged)/len(paired) if paired else None,
            "net_candidate_fraction_ci95": p["mask75"]["candidate"]["ci95"],
            "net_image_macro_fraction_ci95": p["mask75"]["image_macro"]["ci95"],
        }
        t["comparisons"][f"{a}_minus_{b}"] = p
    return t


def distribution(xs):
    xs = list(xs)
    a = np.asarray([float(x) for x in xs if finite(x)], dtype=np.float64)
    return {"defined": len(a), "undefined": len(xs)-len(a), "mean": avg(a),
            "median": float(np.median(a)) if len(a) else None,
            "p10": float(np.quantile(a, .1)) if len(a) else None,
            "p90": float(np.quantile(a, .9)) if len(a) else None,
            "min": float(a.min()) if len(a) else None, "max": float(a.max()) if len(a) else None}


def match_contributions(rows, seed, bootstrap):
    """Keep original image denominators, so matched+unmatched=all exactly."""
    per_image = []
    for one in groups(rows):
        valid = all(finite(r.get("iou_H8")) and finite(r.get("iou_EM8"))
                    and isinstance(r.get("energy_match_pass"), (bool, np.bool_)) for r in one)
        item = {"split": one[0].get("split", "val"), "image_id": one[0].get("image_id"), "n_candidates": len(one)}
        if valid:
            item["matched"] = sum((r["iou_H8"]-r["iou_EM8"]) for r in one if r["energy_match_pass"])/len(one)
            item["unmatched"] = sum((r["iou_H8"]-r["iou_EM8"]) for r in one if not r["energy_match_pass"])/len(one)
            item["all"] = sum(r["iou_H8"]-r["iou_EM8"] for r in one)/len(one)
        else:
            item.update(matched=None, unmatched=None, all=None)
        per_image.append(item)
    out = {"definition": "mean_j[sum_i_in_group(IoU_H8-IoU_EM8)/n_j], with original full-image n_j",
           "n_images": len(per_image), "valid_images": sum(finite(r["all"]) for r in per_image),
           "matched": {}, "unmatched": {}, "all": {}, "additivity_error": None}
    if not per_image or out["valid_images"] != len(per_image):
        out.update(status="undefined", reason="Missing IoU or matching flag; no partial-case full-cohort contribution")
        return out, per_image
    for key in ("matched", "unmatched", "all"):
        a = np.asarray([r[key] for r in per_image])
        draws = [v for ix in sample_indices(len(a), seed, bootstrap) for v in a[ix].mean(1)]
        out[key] = {"delta": float(a.mean()), "ci95": ci(draws)}
    out["additivity_error"] = out["matched"]["delta"]+out["unmatched"]["delta"]-out["all"]["delta"]
    total = out["all"]["delta"]
    out["unmatched_share_of_total"] = out["unmatched"]["delta"]/total if total > 0 else None
    out["share_note"] = "Descriptive only; may be outside [0,1] when contributions have opposite signs. Undefined if total <=0."
    out["status"] = "defined"
    return out, per_image


def recovery_ratio(rows, seed, bootstrap):
    out = {"definition": "macro(IoU_H8-IoU_T)/macro(IoU_G-IoU_T) on identical candidates",
           "status": "undefined", "ratio": None, "ci95": [None, None], "used_for_gate": False,
           "n_candidates": len(rows), "n_images": len(groups(rows)),
           "numerator": None, "denominator": None, "bootstrap_nonpositive_denominators": None}
    if not rows or any(not all(finite(r.get(f"iou_{a}")) for a in ("H8", "T", "G")) for r in rows):
        out["reason"] = "No candidates or missing required metric; no complete-case substitution"
        return out
    gs = groups(rows)
    a = np.asarray([avg([r["iou_H8"]-r["iou_T"] for r in one]) for one in gs])
    b = np.asarray([avg([r["iou_G"]-r["iou_T"] for r in one]) for one in gs])
    ad, bd = [], []
    for ix in sample_indices(len(gs), seed, bootstrap):
        ad.extend(a[ix].mean(1)); bd.extend(b[ix].mean(1))
    out.update(numerator=float(a.mean()), denominator=float(b.mean()), numerator_ci95=ci(ad), denominator_ci95=ci(bd))
    if b.mean() <= 0:
        out["reason"] = "GT8-solver reference macro gain is nonpositive; ratio undefined"
        return out
    out["ratio"] = float(a.mean()/b.mean())
    bd = np.asarray(bd)
    invalid = int(np.sum(bd <= 0))
    out["bootstrap_nonpositive_denominators"] = invalid
    if invalid:
        out["reason"] = "Point ratio defined; ratio interval undefined because nonpositive denominator bootstrap draws are not discarded"
        return out
    out.update(status="defined", ci95=ci(np.asarray(ad)/bd), reason="Paired image bootstrap of ratio of macro gains, not mean of candidate ratios")
    return out


def integrity(rows):
    issues = []
    if len(rows) != EXPECTED_CANDIDATES or len(groups(rows)) != EXPECTED_IMAGES:
        issues.append(f"Frozen population count differs: {len(rows)} candidates / {len(groups(rows))} images, expected 1346 / 196")
    fields = ("split", "image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx")
    ids = [tuple(r.get(k) for k in fields) for r in rows]
    duplicates = sum(n-1 for n in Counter(ids).values())
    if duplicates:
        issues.append(f"Duplicate full candidate identities: {duplicates}")
    for k in fields:
        n = sum(r.get(k) is None for r in rows)
        if n:
            issues.append(f"Missing identity field {k}: {n}")
    if any(r.get("split") != "val" for r in rows):
        issues.append("Input contains non-val split; no filtering was performed")
    for a in ARMS:
        n = sum(not finite(r.get(f"iou_{a}")) or not 0 <= float(r[f"iou_{a}"]) <= 1 for r in rows)
        if n:
            issues.append(f"Missing/nonfinite/out-of-range IoU_{a}: {n}; these are not empty-mask exclusions")
        inconsistent = sum(finite(r.get(f"mask75_{a}")) and finite(r.get(f"iou_{a}"))
                           and int(r[f"mask75_{a}"]) != int(r[f"iou_{a}"] >= .75) for r in rows)
        if inconsistent:
            issues.append(f"Mask75_{a} inconsistent with IoU threshold: {inconsistent}")
    badflag = sum(not isinstance(r.get("energy_match_pass"), (bool, np.bool_)) for r in rows)
    badrel = sum(not finite(r.get("energy_match_relative_error")) or float(r["energy_match_relative_error"]) < 0 for r in rows)
    disagree = sum(isinstance(r.get("energy_match_pass"), (bool, np.bool_))
                   and finite(r.get("energy_match_relative_error"))
                   and bool(r["energy_match_pass"]) != (r["energy_match_relative_error"] <= .05) for r in rows)
    if badflag or badrel or disagree:
        issues.append(f"Energy match integrity: missing flag={badflag}, invalid relative error={badrel}, flag/<=0.05 disagreement={disagree}")
    return {"status": "passed" if not issues else "incomplete_or_inconsistent", "issues": issues,
            "expected_images": EXPECTED_IMAGES, "expected_candidates": EXPECTED_CANDIDATES,
            "actual_images": len(groups(rows)), "actual_candidates": len(rows),
            "identity_fields": fields, "duplicate_identities": duplicates,
            "candidate_rows_dropped": 0, "source_mask75_recomputed_from_iou": True}


def rule(pair_result, minimum=None):
    stat = pair_result["image_macro"]
    if not pair_result["complete_pairs"] or not finite(stat["delta"]) or not all(finite(x) for x in stat["ci95"]):
        return None
    return bool(stat["ci95"][0] > 0 and (minimum is None or stat["delta"] >= minimum))


def decide(result):
    cmp = result["tables"]["val:all"]["comparisons"]
    matched = result["tables"]["val:energy_match_pass"]["comparisons"]["H8_minus_EM8"]["iou"]
    rate = result["energy_matching"]["pass_fraction_all_candidates"]
    rules = {
        "energy_matching_pass_fraction_at_least_0_90": rate >= .9 if finite(rate) else None,
        "all_H8_minus_EM8_delta_at_least_0_002_and_ci_lower_positive": rule(cmp["H8_minus_EM8"]["iou"], .002),
        "all_H8_minus_E8_ci_lower_positive": rule(cmp["H8_minus_E8"]["iou"]),
        "all_H8_minus_T_ci_lower_positive": rule(cmp["H8_minus_T"]["iou"]),
        "matched_H8_minus_EM8_ci_lower_positive": rule(matched),
    }
    if result["integrity"]["issues"]:
        state, summary = "INCOMPLETE", "输入身份、全体口径或匹配字段仍有缺失/不一致，本轮机制判断不完整。"
    elif all(v is True for v in rules.values()):
        state, summary = "SIGNAL_ESTABLISHED_IN_THIS_DIAGNOSTIC", "本轮预设机制信号通过：在当前GT辅助8格替换及固定求解器范围，错误位置的传播代价具有超出所移除输入误差能量的作用。"
    elif any(v is False for v in rules.values()):
        state, summary = "SIGNAL_NOT_ESTABLISHED", "至少一项预设条件未通过，本轮未建立所要求的误差位置代价机制信号；这不证明该机制不存在。"
    else:
        state, summary = "UNCERTAIN", "有效统计不足以判定预设信号，保留不确定，不通过删样或补训练改变本轮。"
    return {"state": state, "summary": summary, "rules": rules, "stop_this_round": True,
            "automatic_training_authorized": False, "automatic_followup_authorized": False,
            "primary_metric": "val full-cohort image-macro original-image IoU H8-minus-EM8",
            "thresholds": {"minimum_energy_match_pass_fraction": .9, "maximum_energy_match_relative_error": .05,
                           "minimum_primary_iou_delta": .002, "ci_level": .95, "ci_lower_strictly_above": 0},
            "interpretation": "只支持本固定原型、旧7O误差、固定求解器及GT辅助选格/替换的诊断信号；不等于无GT可识别、可部署收益、原生模型唯一根因或信息论限制。",
            "ratio_is_descriptive_not_gate": True, "multiplicity": "预设比较的逐项95%区间；未作额外多重比较校正，不称同时95%覆盖。"}


def fmt(x, pp=False, signed=False):
    if not finite(x):
        return "未定义"
    return format(float(x)*(100 if pp else 1), "+.4f" if signed else ".4f")


def effect(p, kind="image_macro"):
    s = p[kind]
    return f"{fmt(s['delta'], True, True)} [{fmt(s['ci95'][0], True, True)}, {fmt(s['ci95'][1], True, True)}]"


def report(r, d):
    all_t = r["tables"]["val:all"]
    cmp = all_t["comparisons"]
    lines = ["# 一次有界误差传播诊断", "", d["summary"], "",
             f"全体 {all_t['n_images']} 张图片、{all_t['n_candidates']} 个固定官方候选；本轮不训练。所有空掩码及能量匹配未通过候选保留在主表。",
             f"整图配对bootstrap {r['statistics']['bootstrap']}次，seed={r['statistics']['seed']}。下表效应/区间单位均为IoU百分点。", "",
             "## 冻结对照", "", "|组别|操作|", "|---|---|"]
    lines += [f"|{a}|{LABELS[a]}|" for a in ARMS]
    lines += ["", "H8以连续logit传播能量选格；E8、R8、EM8均为冻结对照。EM8只在4096个预先固定随机子集中匹配移除的输入误差平方和，不使用IoU选点，并非全局精确匹配。H8/E8/R8/EM8的替换值来自GT，均不是部署方法。", "",
              "## 预设主比较与强对照", "", "|比较|图片macro IoU差 [95% CI]|候选平均差 [整图bootstrap 95% CI]|相对参考组修复/损伤/净增|", "|---|---:|---:|---:|"]
    for name in ("H8_minus_EM8", "H8_minus_E8", "H8_minus_T"):
        p, tr = cmp[name], cmp[name]["mask75_transition"]
        lines.append(f"|{name}|{effect(p['iou'])}|{effect(p['iou'], 'candidate')}|{tr['repair']} / {tr['damage']} / {tr['net']:+d}|")
    matched = r["tables"]["val:energy_match_pass"]
    m = r["energy_matching"]
    lines += ["", f"能量匹配通过 {m['passed']}/{m['total_candidates']}，比例 {fmt(m['pass_fraction_all_candidates'], True)}%；门槛为至少90%，相对误差不超过5%。未通过 {m['failed']} 个、字段未知 {m['unknown']} 个。匹配子组独立描述并参与预设门槛，未用于替换全体主结果。",
              f"匹配通过子组：{matched['n_images']}张图片、{matched['n_candidates']}个候选，H8−EM8图片macro IoU差 {effect(matched['comparisons']['H8_minus_EM8']['iou'])} pp。", "",
              "### 优势来自匹配者还是未匹配者", "", "逐图使用原全体候选数 n_j 作分母，匹配/未匹配成员之外置零；两项相加为全体macro差。该贡献不是子组自身macro均值。", "",
              "|贡献|macro IoU百分点 [95% CI]|", "|---|---:|"]
    contrib = r["matching_contributions"]
    for k in ("matched", "unmatched", "all"):
        s = contrib.get(k, {})
        bounds = s.get("ci95", [None, None])
        lines.append(f"|{k}|{fmt(s.get('delta'), True, True)} [{fmt(bounds[0], True, True)}, {fmt(bounds[1], True, True)}]|")
    lines += ["", f"未匹配者占全体正收益的比例：{fmt(contrib.get('unmatched_share_of_total'), True)}%。这是描述性比率，分母非正时未定义；有正负抵消时可超出0–100%，不裁截。", "",
              "## 正常掩码效用", "", "各修正组相对A；每格差值均为图片macro百分点，AUC以相同固定支持上的连续logit评价。", "",
              "|组别|IoU原值|IoU变化 [95% CI]|Mask75修复/损伤/净增|Coverage差|AUC差|FPR差|", "|---|---:|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        if a == "A":
            lines.append(f"|A|{fmt(all_t['image_macro']['iou']['A'])}|参照|—|—|—|—|")
            continue
        p = cmp[f"{a}_minus_A"]
        tr = p["mask75_transition"]
        lines.append(f"|{a}|{fmt(all_t['image_macro']['iou'][a])}|{effect(p['iou'])}|{tr['repair']}/{tr['damage']}/{tr['net']:+d}|{fmt(p['coverage']['image_macro']['delta'], True, True)}|{fmt(p['auc']['image_macro']['delta'], True, True)}|{fmt(p['fpr']['image_macro']['delta'], True, True)}|")
    lines += ["", "FPR是固定预测框支持内非目标像素FPR，包括邻居，不单独等于纯背景泄漏。未定义AUC仅影响AUC统计，不移除该候选IoU。全部逐项区间、未定义数量和Mask75净变化区间见JSON。", "",
              "## 少量替换回收了多少GT8参考机会", ""]
    q = r["recovery_ratio"]
    lines += [f"以同一全体候选先作图片macro再求比率：(H8−T)/(G−T)={fmt(q['ratio'])}，95% CI [{fmt(q['ci95'][0])}, {fmt(q['ci95'][1])}]。分子={fmt(q['numerator'], True, True)} pp，分母={fmt(q['denominator'], True, True)} pp。",
              f"状态：{q['status']}；{q['reason']}。",
              "分母不为正时未定义；任何重采样分母非正时不丢弃这些重采样来制造区间。GT8固定求解结果不是任务最优上界，比率不截断到[0,1]，且不作为放行条件。", "",
              "## 连续传播与输入能量", "", "候选级分布；这些描述统计不替代预设效用门槛。", "", "|量|均值|中位数|P10|P90|未定义数|", "|---|---:|---:|---:|---:|---:|"]
    for k, s in r["geometry"].items():
        lines.append(f"|{k}|{fmt(s['mean'])}|{fmt(s['median'])}|{fmt(s['p10'])}|{fmt(s['p90'])}|{s['undefined']}|")
    lines += ["", "actual/permutation MSE比率及AR谱范数只描述本误差与本传播算子的关系，不单独证明原生YOLO根因。", "",
              "## 预先声明的描述性分层", "", "|分层|图片/候选|H8−EM8 macro IoU差 [95% CI]|H8−T macro IoU差 [95% CI]|", "|---|---:|---:|---:|"]
    for name, t in r["tables"].items():
        if name in ("val:all", "val:energy_match_pass"):
            continue
        lines.append(f"|{name}|{t['n_images']}/{t['n_candidates']}|{effect(t['comparisons']['H8_minus_EM8']['iou'])}|{effect(t['comparisons']['H8_minus_T']['iou'])}|")
    lines += ["", "框好失败：Box IoU≥0.75且A原图Mask IoU<0.75；原成功/失败由A固定。金字塔层级不等同目标大小。分层结果不能替代全体主结果。", "",
              "## 最终条件与结束", "", "|预设条件|结果|", "|---|---|"]
    lines += [f"|{k}|{'通过' if v is True else '未通过' if v is False else '未知'}|" for k, v in d["rules"].items()]
    lines += ["", d["summary"], "", d["interpretation"], "",
              "无论结果如何，本轮到此停止，不自动增加训练、扩大数据、调整选格、换阈值或开启后续模型。", "",
              "输入审计："+r["integrity"]["status"]+"。"]
    lines += [f"- {s}" for s in r["integrity"]["issues"]]
    lines += ["", "本批图片此前已被研究查看，属于同域定位性评价，不称新盲测。95%区间为逐项区间，不声称所有比较同时覆盖。"]
    return "\n".join(lines)+"\n"


def summarize(rows, out, seed=20261003, bootstrap=5000):
    rows = list(rows)
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    matched = [r for r in rows if r.get("energy_match_pass") is True]
    failed = sum(r.get("energy_match_pass") is False for r in rows)
    unknown = len(rows)-len(matched)-failed
    subsets = {
        "val:all": rows,
        "val:energy_match_pass": matched,
        "val:energy_match_fail": [r for r in rows if r.get("energy_match_pass") is False],
        "val:box_good_original_failure": [r for r in rows if finite(r.get("box_iou")) and r["box_iou"] >= .75 and finite(r.get("iou_A")) and r["iou_A"] < .75],
        "val:original_success": [r for r in rows if finite(r.get("iou_A")) and r["iou_A"] >= .75],
        "val:original_failure": [r for r in rows if finite(r.get("iou_A")) and r["iou_A"] < .75],
    }
    for level in (0, 1, 2):
        subsets[f"val:level_{level}"] = [r for r in rows if r.get("pyramid_level") == level]
    result = {
        "schema": "bounded_error_geometry_v1", "arms": LABELS, "integrity": integrity(rows),
        "statistics": {"primary": "full-cohort image-macro original-image IoU H8-minus-EM8",
                       "secondary": "candidate mean with whole-image bootstrap", "seed": seed,
                       "bootstrap": bootstrap, "ci": "pointwise percentile 95% paired whole-image bootstrap",
                       "candidate_exclusion": "none; undefined metrics counted per metric; missing required IoU invalidates decision",
                       "strata": "descriptive, original success/failure fixed from A"},
        "energy_matching": {"passed": len(matched), "failed": failed, "unknown": unknown,
                            "total_candidates": len(rows), "pass_fraction_all_candidates": len(matched)/len(rows) if rows and not unknown else None,
                            "relative_error_limit": .05, "minimum_required_pass_fraction": .9,
                            "unknowns_are_not_passes": True},
        "geometry": {k: distribution(r.get(k) for r in rows) for k in GEOMETRY},
        "tables": {name: table(rs, seed, bootstrap, "primary population" if name == "val:all" else "prespecified descriptive subgroup; matched subgroup also has locked gate") for name, rs in subsets.items()},
        "recovery_ratio": recovery_ratio(rows, seed, bootstrap),
        "recovery_ratio_box_good_original_failure": recovery_ratio(subsets["val:box_good_original_failure"], seed, bootstrap),
    }
    result["matching_contributions"], contributions = match_contributions(rows, seed, bootstrap)
    decision = decide(result)
    result["decision"] = decision
    write_json(out/"RESULTS_ERROR_GEOMETRY.json", result)
    write_json(out/"DECISION_ERROR_GEOMETRY.json", decision)
    (out/"ERROR_PROPAGATION_REPORT.md").write_text(report(result, decision), encoding="utf-8")
    cidx = {(str(x["split"]), str(x["image_id"])): x for x in contributions}
    with (out/"PER_IMAGE.jsonl").open("w", encoding="utf-8") as stream:
        for one in groups(rows):
            item = {"split": one[0].get("split", "val"), "image_id": one[0].get("image_id"), "n_candidates": len(one)}
            for a in ARMS:
                for metric in METRICS:
                    xs = [val(row, metric, a) for row in one]
                    item[f"{metric}_{a}"] = avg(xs)
                    item[f"n_defined_{metric}_{a}"] = sum(finite(x) for x in xs)
            item["n_energy_match_pass"] = sum(r.get("energy_match_pass") is True for r in one)
            item["H8_EM8_full_denominator_contributions"] = cidx[(str(item["split"]), str(item["image_id"]))]
            stream.write(json.dumps(clean(item), ensure_ascii=False, allow_nan=False)+"\n")
    with (out/"table.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("stratum", "images", "candidates", "comparison", "metric", "statistic", "arm_mean", "reference_mean", "delta", "ci95_low", "ci95_high", "undefined_candidates"))
        for name, t in result["tables"].items():
            for comparison, p in t["comparisons"].items():
                for metric in METRICS:
                    for kind in ("image_macro", "candidate"):
                        s = p[metric][kind]
                        writer.writerow((name, t["n_images"], t["n_candidates"], comparison, metric, kind,
                                         s["arm_mean"], s["reference_mean"], s["delta"], *s["ci95"], p[metric]["missing_or_undefined_candidates"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    source = args.input or (args.run/"PER_CANDIDATE.jsonl" if args.run else None)
    if source is None:
        parser.error("provide --run or --input")
    out = args.out or args.run or source.parent
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    result = summarize(rows, out, args.seed, args.bootstrap)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
