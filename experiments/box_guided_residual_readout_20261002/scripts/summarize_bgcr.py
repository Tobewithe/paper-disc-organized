"""Summarize frozen-candidate BGCR results; never fit or select checkpoints.

API: summarize(rows, out, seed=20261002, bootstrap=5000) -> dict
CLI: python summarize_bgcr.py --run RUN [--out OUT]
     python summarize_bgcr.py --input PER_CANDIDATE.jsonl --out OUT

Input: frozen_evaluation.evaluate_image rows, arms A/B/C/W, split fit/dev/val,
wrong_roi_available bool. W is evaluated ONLY on the explicitly available subset.
Dependencies: NumPy and the standard library. Source rows are never rewritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ARMS = ("A", "B", "C")
METRICS = ("iou", "mask75", "coverage", "auc", "fpr")
EXPECTED = {"fit": (796, 6058), "dev": (197, 1402), "val": (196, 1346)}
LABELS = {
    "A": "原始冻结模型",
    "B": "c0 + 每层零初始化64→32仿射残差，输入为fit标准化后的原h",
    "C": "c0 + 同形仿射残差，输入为fit标准化后的r=mean(ROIAlign3×3(H,predB))−h",
    "W": "C的同一冻结参数，替换为已冻结错配ROI；只在wrong_roi_available=true子集解释",
}


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


def interval(xs, level=.95):
    xs = np.asarray(xs, dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    tail = (1-level)/2
    return np.quantile(xs, [tail, 1-tail]).tolist() if len(xs) else [None, None]


def image_groups(rows):
    by_id = defaultdict(list)
    for r in rows:
        by_id[(str(r.get("split")), str(r.get("image_id")))].append(r)
    return [by_id[k] for k in sorted(by_id)]


def value(r, metric, arm):
    if metric == "mask75":
        x = r.get(f"iou_{arm}")
        return int(x >= .75) if finite(x) else None
    return r.get(f"{metric}_{arm}")


def resamples(n, seed, bootstrap):
    if n:
        rng = np.random.default_rng(seed)
        for begin in range(0, bootstrap, 256):
            yield rng.integers(0, n, (min(256, bootstrap-begin), n))


def paired_stats(gs, metric, arm, ref, seed, bootstrap, a_success_damage=False):
    counts, left, right = [], [], []
    for one in gs:
        paired = []
        for r in one:
            if a_success_damage:
                if not finite(r.get("iou_A")) or r["iou_A"] < .75:
                    continue
                a, b = value(r, "mask75", arm), value(r, "mask75", ref)
                a = 1-a if finite(a) else None
                b = 1-b if finite(b) else None
            else:
                a, b = value(r, metric, arm), value(r, metric, ref)
            if finite(a) and finite(b):
                paired.append((float(a), float(b)))
        counts.append(len(paired))
        left.append(sum(a for a, _ in paired))
        right.append(sum(b for _, b in paired))
    n = np.asarray(counts, dtype=np.int64)
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    ok = n > 0
    ma, mb = np.divide(a, n, out=np.zeros(len(n)), where=ok), np.divide(b, n, out=np.zeros(len(n)), where=ok)
    macro_draws, candidate_draws = [], []
    for draw in resamples(len(gs), seed, bootstrap):
        ni, nc = ok[draw].sum(1), n[draw].sum(1)
        macro_draws.extend(np.divide((ma-mb)[draw].sum(1), ni, out=np.full(len(draw), np.nan), where=ni > 0))
        candidate_draws.extend(np.divide((a-b)[draw].sum(1), nc, out=np.full(len(draw), np.nan), where=nc > 0))
    population_n = sum(sum(finite(r.get("iou_A")) and r["iou_A"] >= .75 for r in one) for one in gs) if a_success_damage else sum(map(len, gs))
    return {
        "image_macro": {"arm_mean": avg(ma[ok]), "reference_mean": avg(mb[ok]),
                        "delta": avg((ma-mb)[ok]), "ci95": interval(macro_draws), "ci97_5": interval(macro_draws, .975), "valid_images": int(ok.sum())},
        "candidate": {"arm_mean": float(a.sum()/n.sum()) if n.sum() else None,
                      "reference_mean": float(b.sum()/n.sum()) if n.sum() else None,
                      "delta": float((a-b).sum()/n.sum()) if n.sum() else None,
                      "ci95": interval(candidate_draws), "ci97_5": interval(candidate_draws, .975), "valid_candidates": int(n.sum())},
        "population_candidates": population_n, "missing_or_undefined_candidates": int(population_n-n.sum()),
        "complete_pairs": bool(population_n and n.sum() == population_n),
        "undefined_bootstrap_draws": int(np.sum(~np.isfinite(macro_draws))),
    }


def make_table(rows, seed, bootstrap, role, include_wrong=False):
    gs = image_groups(rows)
    arms = ARMS+("W",) if include_wrong else ARMS
    pairs = [("C", "B"), ("C", "A"), ("B", "A")]
    if include_wrong:
        pairs += [("C", "W"), ("W", "A")]
    table = {"role": role, "n_images": len(gs), "n_candidates": len(rows), "arms": arms,
             "candidate": {}, "image_macro": {}, "undefined": {}, "mask75_counts": {},
             "original_success_damage": {}, "comparisons": {}}
    for m in METRICS:
        table["candidate"][m] = {a: avg([value(r, m, a) for r in rows]) for a in arms}
        table["image_macro"][m] = {a: avg([avg([value(r, m, a) for r in one]) for one in gs]) for a in arms}
        table["undefined"][m] = {a: sum(not finite(value(r, m, a)) for r in rows) for a in arms}
    successes = [r for r in rows if finite(r.get("iou_A")) and r["iou_A"] >= .75]
    table["original_success_damage"] = {
        "definition": "A original-image IoU >= .75; damage_X=1[IoU_X<.75]",
        "n_candidates": len(successes), "n_images": len(image_groups(successes)),
        "counts": {a: sum(value(r, "mask75", a) == 0 for r in successes) for a in arms},
        "rates": {a: avg([1-value(r, "mask75", a) for r in successes if finite(value(r, "mask75", a))]) for a in arms},
        "undefined": {a: sum(not finite(value(r, "mask75", a)) for r in successes) for a in arms},
    }
    for a in arms:
        valid = [value(r, "mask75", a) for r in rows if finite(value(r, "mask75", a))]
        table["mask75_counts"][a] = {"success": int(sum(valid)), "defined": len(valid), "undefined": len(rows)-len(valid)}
    for a, b in pairs:
        p = {m: paired_stats(gs, m, a, b, seed, bootstrap) for m in METRICS}
        rs = [r for r in rows if finite(value(r, "mask75", a)) and finite(value(r, "mask75", b))]
        repair = sum(value(r, "mask75", b) == 0 and value(r, "mask75", a) == 1 for r in rs)
        damage = sum(value(r, "mask75", b) == 1 and value(r, "mask75", a) == 0 for r in rs)
        p["mask75_transition"] = {
            "reference_arm": b, "repair": repair, "damage": damage, "net": repair-damage,
            "valid_candidates": len(rs), "missing_candidates": len(rows)-len(rs),
            "net_candidate_fraction": (repair-damage)/len(rs) if rs else None,
            "net_candidate_fraction_ci95": p["mask75"]["candidate"]["ci95"],
            "net_image_macro_fraction_ci95": p["mask75"]["image_macro"]["ci95"],
        }
        p["damage_rate_on_A_success"] = paired_stats(gs, "mask75", a, b, seed, bootstrap, a_success_damage=True)
        p["damage_rate_on_A_success"]["interpretation"] = "Fixed A-success population. Positive difference means more damage in first arm; CI crossing zero is not noninferiority/equivalence."
        table["comparisons"][f"{a}_minus_{b}"] = p
    return table


def audit(rows):
    issues, notes, counts = [], [], {}
    identity_fields = ("split", "image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx")
    keys = [tuple(r.get(k) for k in identity_fields) for r in rows]
    duplicates = sum(n-1 for n in Counter(keys).values())
    if duplicates:
        issues.append(f"Duplicate full identities: {duplicates}")
    for k in identity_fields:
        n = sum(r.get(k) is None for r in rows)
        if n:
            issues.append(f"Missing identity {k}: {n}")
    for split, expected in EXPECTED.items():
        rr = [r for r in rows if r.get("split") == split]
        counts[split] = {"images": len(image_groups(rr)), "candidates": len(rr), "expected_images": expected[0], "expected_candidates": expected[1]}
        if (counts[split]["images"], len(rr)) != expected:
            issues.append(f"Frozen {split} population differs from expected {expected}")
    unexpected = sorted({str(r.get("split")) for r in rows} - set(EXPECTED))
    if unexpected:
        issues.append(f"Unexpected split(s): {unexpected}; no rows deleted")
    for a in (*ARMS, "W"):
        rr = [r for r in rows if a != "W" or r.get("wrong_roi_available") is True]
        bad = sum(not finite(r.get(f"iou_{a}")) or not 0 <= r[f"iou_{a}"] <= 1 for r in rr)
        if bad:
            issues.append(f"Undefined/out-of-range required IoU_{a}: {bad}")
        inconsistent = sum(finite(r.get(f"mask75_{a}")) and finite(r.get(f"iou_{a}"))
                           and int(r[f"mask75_{a}"]) != int(r[f"iou_{a}"] >= .75) for r in rr)
        if inconsistent:
            issues.append(f"Mask75_{a}/IoU inconsistency: {inconsistent}")
    val_rows = [r for r in rows if r.get("split") == "val"]
    if any(not finite(r.get("box_iou")) for r in val_rows):
        issues.append("Missing/nonfinite val box_iou prevents complete primary subgroup definition")
    unknown = sum(not isinstance(r.get("wrong_roi_available"), (bool, np.bool_)) for r in rows)
    if unknown:
        notes.append(f"wrong_roi_available unknown for {unknown} rows; none included in C-W subset")
    pixel_errors = {k: sum(int(r.get(k, 0)) for r in rows)
                    for k in ("zero_bias_padded_pixel_differences", "zero_bias_original_pixel_differences")}
    if any(pixel_errors.values()):
        issues.append(f"Nonzero native/manual baseline replay pixel differences: {pixel_errors}")
    return {"status": "passed" if not issues else "incomplete_or_inconsistent", "issues": issues,
            "notes": notes, "counts": counts, "identity_fields": identity_fields, "duplicate_identities": duplicates,
            "candidate_rows_dropped": 0, "baseline_zero_bias_pixel_errors": pixel_errors,
            "baseline_audited_candidates": sum("zero_bias_original_pixel_differences" in r for r in rows),
            "wrong_roi_flag_unknown": unknown}


def outcome(pair_result, ci_key="ci95"):
    s = pair_result["image_macro"]
    if not pair_result["complete_pairs"] or not finite(s["delta"]) or not all(finite(x) for x in s[ci_key]):
        return "不完整或未定义"
    if s[ci_key][0] > 0:
        return "本次配对区间支持正向差异"
    if s[ci_key][1] < 0:
        return "本次配对区间支持负向差异"
    return "区间跨零，尚未确定差异方向"


def fmt(x, pp=False, signed=False):
    return format(float(x)*(100 if pp else 1), "+.4f" if signed else ".4f") if finite(x) else "未定义"


def effect(p, kind="image_macro", ci_key="ci95"):
    s = p[kind]
    return f"{fmt(s['delta'], True, True)} [{fmt(s[ci_key][0], True, True)}, {fmt(s[ci_key][1], True, True)}]"


def render(result):
    primary = result["tables"]["val:box_good_original_failure"]
    c_b = primary["comparisons"]["C_minus_B"]["iou"]
    c_a = primary["comparisons"]["C_minus_A"]["iou"]
    lines = ["# BGCR：固定官方候选的框引导残差短训练评价", "",
             f"主分层为原Box IoU≥0.75、原图Mask IoU<0.75，共{primary['n_images']}张图片、{primary['n_candidates']}个候选。",
             f"C−B图片macro IoU差为{effect(c_b, ci_key='ci97_5')}个百分点（97.5% CI）：{outcome(c_b, 'ci97_5')}。",
             f"C−A图片macro IoU差为{effect(c_a, ci_key='ci97_5')}个百分点（97.5% CI）：{outcome(c_a, 'ci97_5')}。",
             "预设实用量级：C−B至少0.005（0.5个百分点），C−A为正；两项主要比较用各自97.5%区间作Bonferroni总体95%覆盖控制，另保留95%描述区间。代价通过全体和原成功损伤表报告，不要求所有指标同时提高。", "",
             "本轮协议固定seed0、10轮、最后checkpoint；B/C均使用官方mask BCE，无附加正则或教师项。本脚本只汇总传入的冻结预测，不选择epoch、阈值或训练配置；最终checkpoint身份由运行记录核对。", "",
             "## 组别与范围", "", "|组别|定义|", "|---|---|"]
    lines += [f"|{a}|{LABELS[a]}|" for a in (*ARMS, "W")]
    lines += ["", "所有组保持原型、候选、框、分类分数及正常解码不变。GT仅用于训练监督、预先定义评价分层及评价；推理残差输入不包含GT框/标签。这里评价的是既有官方TAL固定候选，图片此前被研究查看，非新盲测、非完整输出COCO AP。", "",
              "## 主要与全体比较", "", "差值与区间均为百分点。候选平均的区间也按整张图片重采样。", "",
              "|范围|图片/候选|比较|图片macro IoU差 [95% CI]|候选平均IoU差 [95% CI]|Mask75修复/损伤/净增（相对该比较参考组）|", "|---|---:|---|---:|---:|---:|"]
    for name in ("val:box_good_original_failure", "val:all", "fit:all", "dev:all"):
        t = result["tables"][name]
        for comp in ("C_minus_B", "C_minus_A", "B_minus_A"):
            p = t["comparisons"][comp]
            tr = p["mask75_transition"]
            lines.append(f"|{name}|{t['n_images']}/{t['n_candidates']}|{comp}|{effect(p['iou'])}|{effect(p['iou'], 'candidate')}|{tr['repair']}/{tr['damage']}/{tr['net']:+d}|")
    lines += ["", "## 各split与预声明分层", "", "以下各组均相对A；主分层之外不替代主结果。Coverage/AUC/FPR差为图片macro百分点。", "",
              "|范围|图片/候选|组|macro IoU|IoU差 [95% CI]|Mask75达标数|修复/损伤/净增|Coverage差|AUC差|FPR差|", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, t in result["tables"].items():
        if name.endswith(":wrong_roi_available"):
            continue
        for a in ARMS:
            if a == "A":
                lines.append(f"|{name}|{t['n_images']}/{t['n_candidates']}|A|{fmt(t['image_macro']['iou']['A'])}|参照|{t['mask75_counts']['A']['success']}|—|—|—|—|")
                continue
            p = t["comparisons"][f"{a}_minus_A"]
            tr = p["mask75_transition"]
            lines.append(f"|{name}|{t['n_images']}/{t['n_candidates']}|{a}|{fmt(t['image_macro']['iou'][a])}|{effect(p['iou'])}|{t['mask75_counts'][a]['success']}|{tr['repair']}/{tr['damage']}/{tr['net']:+d}|{fmt(p['coverage']['image_macro']['delta'], True, True)}|{fmt(p['auc']['image_macro']['delta'], True, True)}|{fmt(p['fpr']['image_macro']['delta'], True, True)}|")
    lines += ["", "P3/P4/P5是候选层级，不直接等同目标大小。所有空mask保留；AUC使用连续logit和固定预测框支持，NaN AUC单独计数且不删除其IoU。FPR表示预测框内非目标像素，包括邻居，并非纯背景指标。完整逐指标置信区间见JSON/CSV。", "",
              "## 原成功候选损伤", "", "固定人群：A原图IoU≥0.75。正差值表示C造成更多损伤；跨零不表示等价或非劣。", "",
              "|split|原成功图片/候选|B损伤|C损伤|C−B损伤率，候选口径 [95% CI]|C−A损伤率，候选口径 [95% CI]|", "|---|---:|---:|---:|---:|---:|"]
    for split in EXPECTED:
        t = result["tables"][f"{split}:all"]
        d = t["original_success_damage"]
        lines.append(f"|{split}|{d['n_images']}/{d['n_candidates']}|{d['counts']['B']}|{d['counts']['C']}|{effect(t['comparisons']['C_minus_B']['damage_rate_on_A_success'], 'candidate')}|{effect(t['comparisons']['C_minus_A']['damage_rate_on_A_success'], 'candidate')}|")
    lines += ["", "## 同一C参数的错配ROI对照", "", "仅wrong_roi_available=true的固定子集计算C−W。未提供合法错配对象者仍在A/B/C全体表，绝不把回退ROI的W当成有效错配。", "",
              "|范围|可用图片/候选|C−W macro IoU差 [95% CI]|C−W候选IoU差 [95% CI]|Coverage差|AUC差|FPR差|", "|---|---:|---:|---:|---:|---:|---:|"]
    for split in EXPECTED:
        t = result["tables"][f"{split}:wrong_roi_available"]
        p = t["comparisons"]["C_minus_W"]
        lines.append(f"|{split}|{t['n_images']}/{t['n_candidates']}|{effect(p['iou'])}|{effect(p['iou'], 'candidate')}|{fmt(p['coverage']['image_macro']['delta'], True, True)}|{fmt(p['auc']['image_macro']['delta'], True, True)}|{fmt(p['fpr']['image_macro']['delta'], True, True)}|")
    t = result["tables"]["val:box_good_original_failure_wrong_roi_available"]
    p = t["comparisons"]["C_minus_W"]
    lines.append(f"|val主分层的错配可用者|{t['n_images']}/{t['n_candidates']}|{effect(p['iou'])}|{effect(p['iou'], 'candidate')}|{fmt(p['coverage']['image_macro']['delta'], True, True)}|{fmt(p['auc']['image_macro']['delta'], True, True)}|{fmt(p['fpr']['image_macro']['delta'], True, True)}|")
    lines += ["", "真实ROI优于该错配只能支持本实验下实例对应关系有作用，不自动证明某种唯一缺失信息，也不替代C−B/C−A。", "",
              "## 审计与解释边界", "",
              f"审计状态：{result['audit']['status']}。Bootstrap={result['bootstrap']}，seed={result['bootstrap_seed']}。两项主要比较采用97.5% percentile区间进行Bonferroni总体95%覆盖控制；其余95%区间为描述性逐项区间。",
              "训练种子0与bootstrap种子为不同用途。只有一次种子的固定预算短训练，不能据此宣称跨种子稳定性、方法最优或完整部署提升。未声明的实用门槛不在结果后补造；不使用论文达标标准触发下一轮。"]
    lines += [f"- {x}" for x in result["audit"]["issues"]+result["audit"]["notes"]]
    lines += ["", "完成这次训练与评价后停止。本汇总不追加训练、扩数据、换checkpoint、校准、门控或新实验。"]
    return "\n".join(lines)+"\n"


def summarize(rows, out, seed=20261002, bootstrap=5000):
    rows = list(rows)
    if not rows:
        raise ValueError("Empty candidate input")
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables, availability = {}, {}
    for split in EXPECTED:
        rr = [r for r in rows if r.get("split") == split]
        wrong = [r for r in rr if r.get("wrong_roi_available") is True]
        tables[f"{split}:all"] = make_table(rr, seed, bootstrap, "all fixed candidates; secondary to prespecified val failure stratum")
        tables[f"{split}:wrong_roi_available"] = make_table(wrong, seed, bootstrap, "available-only paired mismatch control", True)
        availability[split] = {"total_candidates": len(rr), "available_candidates": len(wrong), "available_images": len(image_groups(wrong)),
                               "unavailable_candidates": sum(r.get("wrong_roi_available") is False for r in rr),
                               "unknown_candidates": sum(not isinstance(r.get("wrong_roi_available"), (bool, np.bool_)) for r in rr)}
    vr = [r for r in rows if r.get("split") == "val"]
    subgroups = {
        "box_good_original_failure": [r for r in vr if finite(r.get("box_iou")) and r["box_iou"] >= .75 and finite(r.get("iou_A")) and r["iou_A"] < .75],
        "original_success": [r for r in vr if finite(r.get("iou_A")) and r["iou_A"] >= .75],
        "original_failure": [r for r in vr if finite(r.get("iou_A")) and r["iou_A"] < .75],
    }
    subgroups.update({f"P{lev+3}": [r for r in vr if r.get("pyramid_level") == lev] for lev in (0, 1, 2)})
    for name, rr in subgroups.items():
        tables[f"val:{name}"] = make_table(rr, seed, bootstrap, "primary prespecified stratum" if name == "box_good_original_failure" else "prespecified descriptive stratum")
    rr = [r for r in subgroups["box_good_original_failure"] if r.get("wrong_roi_available") is True]
    tables["val:box_good_original_failure_wrong_roi_available"] = make_table(rr, seed, bootstrap, "primary-stratum available-only paired mismatch control", True)
    result = {
        "schema": "bgcr_fixed_candidate_summary_v1", "arms": LABELS, "audit": audit(rows),
        "primary_stratum": "val:box_good_original_failure", "primary_comparisons": ["C_minus_B", "C_minus_A"],
        "primary_metric": "image-macro original-image Mask IoU", "secondary_statistic": "candidate mean, whole-image bootstrap",
        "bootstrap": bootstrap, "bootstrap_seed": seed, "bootstrap_unit": "paired complete image clusters",
        "intervals": "95% descriptive pointwise percentile; two primary comparisons use 97.5% intervals for Bonferroni family95%; extra stored97.5% intervals do not imply wider-family control", "units": "fractions; times100 gives percentage points",
        "primary_locked_criteria": {"C_minus_B_minimum_iou": .005, "C_minus_A_positive_iou": True,
                                    "primary_interval_key": "ci97_5", "family_alpha": .05,
                                    "number_of_primary_comparisons": 2, "requires_all_metrics_improve": False},
        "checkpoint_policy": "protocol: seed0, exactly10epochs, last checkpoint; summary does not inspect or select checkpoint",
        "decode": "frozen_evaluation: process_mask(upsample=True) -> binary input mask -> scale_masks(real ratio_pad) -> >0.5",
        "auc_fpr_support": "continuous640-grid logits inside fixed predicted box; raw COCO GT nearest resize/pad",
        "undefined_auc": "retained candidate for other metrics; undefined count per arm and stratum",
        "scope": "previously viewed images, frozen official one-to-one TAL candidates, no full-output COCO AP or new blind test",
        "wrong_roi_availability": availability, "tables": tables,
        "automatic_followup": False, "automatic_training": False,
    }
    result["primary_readout"] = {k: outcome(tables[result["primary_stratum"]]["comparisons"][k]["iou"], "ci97_5") for k in result["primary_comparisons"]}
    write_json(out/"RESULTS.json", result)
    (out/"REPORT.md").write_text(render(result), encoding="utf-8")
    with (out/"PER_IMAGE.jsonl").open("w", encoding="utf-8") as stream:
        for one in image_groups(rows):
            rec = {"split": one[0].get("split"), "image_id": one[0].get("image_id"), "n_candidates": len(one)}
            for a in (*ARMS, "W"):
                ar = one if a != "W" else [r for r in one if r.get("wrong_roi_available") is True]
                for m in METRICS:
                    xs = [value(r, m, a) for r in ar]
                    rec[f"{m}_{a}"] = avg(xs)
                    rec[f"n_defined_{m}_{a}"] = sum(finite(v) for v in xs)
            bg = [r for r in one if finite(r.get("box_iou")) and r["box_iou"] >= .75 and finite(r.get("iou_A")) and r["iou_A"] < .75]
            rec["n_box_good_original_failure"] = len(bg)
            rec["box_good_original_failure"] = {f"{m}_{a}": avg([value(r, m, a) for r in bg]) for a in ARMS for m in METRICS}
            rec["n_wrong_roi_available"] = sum(r.get("wrong_roi_available") is True for r in one)
            original_success = [r for r in one if finite(r.get("iou_A")) and r["iou_A"] >= .75]
            rec["n_original_success"] = len(original_success)
            rec["original_success_damage_counts"] = {a: sum(value(r, "mask75", a) == 0 for r in original_success) for a in ARMS}
            stream.write(json.dumps(clean(rec), ensure_ascii=False, allow_nan=False)+"\n")
    with (out/"ABLATION_TABLE.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("stratum", "role", "images", "candidates", "comparison", "metric", "statistic", "arm_mean", "reference_mean", "delta", "ci95_low", "ci95_high", "ci97_5_low", "ci97_5_high", "undefined_candidates", "repair", "damage", "net_mask75"))
        for name, t in tables.items():
            for comp, p in t["comparisons"].items():
                tr = p["mask75_transition"]
                for m in METRICS:
                    for kind in ("image_macro", "candidate"):
                        s = p[m][kind]
                        writer.writerow((name, t["role"], t["n_images"], t["n_candidates"], comp, m, kind,
                                         s["arm_mean"], s["reference_mean"], s["delta"], *s["ci95"], *s["ci97_5"], p[m]["missing_or_undefined_candidates"],
                                         tr["repair"], tr["damage"], tr["net"]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--bootstrap", type=int, default=5000)
    a = p.parse_args()
    source = a.input or (a.run/"PER_CANDIDATE.jsonl" if a.run else None)
    if source is None:
        p.error("provide --run or --input")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    result = summarize(rows, a.out or a.run or source.parent, a.seed, a.bootstrap)
    print(json.dumps({"audit": result["audit"]["status"], "primary_readout": result["primary_readout"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
