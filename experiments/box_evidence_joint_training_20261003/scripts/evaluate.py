"""Fixed-official-candidate evaluation for box-evidence coefficient projection.

Public APIs:
  evaluate_image(x, coeffs, coco, chunk_size=4) -> list[dict]
  append_rows(path, rows) -> number_written
  summarize(rows, out, expected_counts=None, seed=20261003, bootstrap=5000,
            run_info=None) -> dict

``x`` is the compact prepared-image schema, not raw-indexed output. The runner
replays the evidence readout branches and supplies [N,32] coefficients for
A, N, D, S, optionally SW; historical SF metrics are joined without replay. This module never runs/reselects assignment, chooses
an epoch, tunes a threshold, trains a module, or performs COCO AP evaluation.
GT is used only for metrics and post-selection donor-identity annotation.

``expected_counts`` may contain, per split, planned_images, effective_images
(or images), and candidates. Planned and effective image counts may differ.
Missing expected counts remain unverified; no population size is hard-coded.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ARMS = ("A", "N", "D", "S", "SF")
METRICS = ("iou", "mask75", "coverage", "auc", "fpr", "bce")
PAIRS = (("S", "N"), ("S", "A"), ("S", "D"), ("N", "A"), ("D", "A"), ("S", "SF"))
PRIMARY_GROUP = "val:box_good_mask_bad"
PRIMARY_COMPARISON = "S_minus_N"
LABELS = {
    "A": "原始冻结模型",
    "N": "原生one-to-one系数分支普通微调",
    "D": "原生系数分支与空间证据直接残差联合训练",
    "S": "原生系数分支与空间证据原型投影残差联合训练",
    "SF": "历史冻结原生系数头的投影S；只复用同身份逐候选指标，不重放模型",
    "SW": "同一冻结S参数，仅替换空间证据读取框；自身当前native query/c、A/K/G及最终解码框保持原样",
}


def finite(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and math.isfinite(float(x))


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def write_json(path, data):
    Path(path).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def append_rows(path, rows):
    """Append evaluated rows. Caller uses an independent run and owns resume IDs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(clean(row), ensure_ascii=False, allow_nan=False)+"\n")
            count += 1
    return count


def _torch():
    # Delayed imports allow CPU-only JSON aggregation without importing a model.
    import torch
    import torch.nn.functional as F
    from ultralytics.utils import ops
    return torch, F, ops


def _pixel_auc_fpr(logits, labels, support):
    scores = logits[support].detach().cpu().numpy().astype(np.float64, copy=False)
    truth = labels[support].detach().cpu().numpy().astype(bool, copy=False)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite logits, cannot silently discard pixels")
    n_pos, n_neg = int(truth.sum()), int((~truth).sum())
    fpr = float(((scores > 0) & ~truth).sum()/n_neg) if n_neg else float("nan")
    if not n_pos or not n_neg:
        return float("nan"), fpr
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores))+1]
    ends = np.r_[starts[1:], len(scores)]
    ranks = np.repeat((starts+ends+1)*.5, ends-starts)
    auc = (ranks[truth[order]].sum()-n_pos*(n_pos+1)*.5)/(n_pos*n_neg)
    return float(auc), fpr


def _padded_gt(original, ratio_pad, input_shape):
    _, F, _ = _torch()
    oh, ow = original.shape
    ih, iw = input_shape
    if ratio_pad is None:
        gain = min(ih/oh, iw/ow)
        nh, nw = round(oh*gain), round(ow*gain)
        left, top = round((iw-nw)/2-.1), round((ih-nh)/2-.1)
    else:
        ratios, pads = ratio_pad
        rh, rw = (float(ratios), float(ratios)) if isinstance(ratios, (float, int)) else map(float, ratios)
        nh, nw = round(oh*rh), round(ow*rw)
        left, top = round(float(pads[0])-.1), round(float(pads[1])-.1)
    right, bottom = iw-nw-left, ih-nh-top
    if min(nh, nw) <= 0 or min(left, top, right, bottom) < 0:
        raise ValueError("Invalid frozen letterbox geometry")
    resized = F.interpolate(original.float()[None, None], (nh, nw), mode="nearest")
    return F.pad(resized, (left, right, top, bottom))[0, 0].bool()


def _logits(proto, coeff, input_shape):
    _, F, _ = _torch()
    z = (coeff@proto.float().flatten(1)).reshape(-1, *proto.shape[-2:])
    return F.interpolate(z[None], input_shape, mode="bilinear", align_corners=False)[0]


def _scale_binary(padded, original_shape, ratio_pad):
    _, _, ops = _torch()
    return ops.scale_masks(padded[None], original_shape, ratio_pad=ratio_pad)[0] > .5


def _box_iou(a, b):
    # Both are same-candidate frozen input-coordinate xyxy boxes; no matching.
    intersection = (min(a[2], b[2])-max(a[0], b[0])).clip(min=0)*(min(a[3], b[3])-max(a[1], b[1])).clip(min=0)
    area_a = max(float(a[2]-a[0]), 0)*max(float(a[3]-a[1]), 0)
    area_b = max(float(b[2]-b[0]), 0)*max(float(b[3]-b[1]), 0)
    return float(intersection)/max(area_a+area_b-float(intersection), 1e-12)


def _donor_metadata(source, x_rows):
    donor = source.get("wrong_box_donor", {}) or {}
    available = source.get("wrong_box_available", donor.get("donor_valid"))
    result = {"wrong_box_available": bool(available) if isinstance(available, (bool, np.bool_)) else None}
    for key in ("annotation_id", "image_id", "raw_id"):
        v = source.get("wrong_box_donor_"+key, donor.get("donor_"+key, donor.get(key)))
        result["wrong_box_donor_"+key] = int(v) if v is not None else None
    # Resolve GT identity ONLY AFTER the runner has selected its donor by
    # prediction information. Never use annotation IDs to select a donor here.
    if result["wrong_box_donor_annotation_id"] is None and result["wrong_box_donor_raw_id"] is not None:
        qimage = result["wrong_box_donor_image_id"]
        candidates = [r for r in x_rows if int(r["raw_id"]) == result["wrong_box_donor_raw_id"]
                      and (qimage is None or int(r["image_id"]) == qimage)]
        if len(candidates) == 1:
            result["wrong_box_donor_annotation_id"] = int(candidates[0]["annotation_id"])
            result["wrong_box_donor_image_id"] = int(candidates[0]["image_id"])
    aid = result["wrong_box_donor_annotation_id"]
    if result["wrong_box_available"] is not True:
        relation = "unavailable"
    elif aid is None:
        relation = "unknown_gt"
    elif aid == int(source["annotation_id"]):
        relation = "same_gt"
    else:
        relation = "different_gt"
    result["wrong_box_gt_relation"] = relation
    return result


def evaluate_image(x, coeffs, coco, chunk_size=4):
    """Normal full-prototype decode at frozen compact candidate identities.

    The runner must use the unchanged x['c0'] as A. Input tensors are compact
    rows, so raw_id is metadata and NEVER indexes coefficients or x['boxes'].
    All arms use the same native full-prototype decoder; only their supplied coefficients differ.
    """
    torch, _, _ = _torch()
    with torch.no_grad():
        return _evaluate_image(x, coeffs, coco, chunk_size)


def _evaluate_image(x, coeffs, coco, chunk_size):
    torch, _, ops = _torch()
    if "A" not in coeffs or any(a not in (*ARMS, "SW") for a in coeffs):
        raise ValueError("Supply original A and coefficient arms from N/D/S/SW")
    proto = x["proto"].detach().float()
    if proto.ndim != 3 or proto.shape[0] != 32:
        raise ValueError("Full [32,H,W] prototype required, not cropped solver pixels")
    n, device = len(x["rows"]), proto.device
    boxes = x["boxes"].detach().to(device=device, dtype=torch.float32)
    c0 = x["c0"].detach().to(device=device, dtype=torch.float32)
    if boxes.shape != (n, 4) or c0.shape != (n, 32):
        raise ValueError("compact boxes/c0 must match rows; never raw-index prepared tensors")
    arms = {}
    for name, c in coeffs.items():
        if c.shape != (n, 32):
            raise ValueError(f"{name}: expected compact [{n},32], got {tuple(c.shape)}")
        arms[name] = c.detach().to(device=device, dtype=torch.float32)
        if not torch.isfinite(arms[name]).all():
            raise ValueError(f"Nonfinite coefficients in arm {name}")
    torch.testing.assert_close(arms["A"], c0, rtol=0, atol=0)
    if not torch.isfinite(proto).all() or not torch.isfinite(boxes).all():
        raise ValueError("Nonfinite frozen prototype or prediction box")
    shape = tuple(map(int, x.get("input_shape", (640, 640))))
    original = tuple(map(int, x["original_shape"]))
    ratio_pad = x["ratio_pad"]
    levels = x["levels"].detach().cpu().tolist() if torch.is_tensor(x["levels"]) else list(x["levels"])
    if len(levels) != n or any(int(v) not in (0, 1, 2) for v in levels):
        raise ValueError("levels must be compact 0/1/2")
    out = []
    for start in range(0, n, max(1, int(chunk_size))):
        end = min(n, start+max(1, int(chunk_size)))
        bb = boxes[start:end]
        base_pad = ops.process_mask(proto, arms["A"][start:end], bb, shape, upsample=True)
        base_logits = _logits(proto, arms["A"][start:end], shape)
        manual_pad = ops.crop_mask(base_logits.clone(), bb).gt(0).byte()
        base_masks = _scale_binary(base_pad, original, ratio_pad)
        manual_masks = _scale_binary(manual_pad, original, ratio_pad)
        padded_diff = (base_pad.bool() != manual_pad.bool()).flatten(1).sum(1)
        image_diff = (base_masks != manual_masks).flatten(1).sum(1)
        if (padded_diff != 0).any() or (image_diff != 0).any():
            raise AssertionError("Original native decode/manual zero-bias replay differs")
        supports = ops.crop_mask(torch.ones((end-start, *shape), device=device), bb).bool()
        truths, padded_truths, records = [], [], []
        for offset, idx in enumerate(range(start, end)):
            source = x["rows"][idx]
            annotation = coco.anns[int(source["annotation_id"])]
            iid = int(source.get("image_id", annotation["image_id"]))
            if int(annotation["image_id"]) != iid:
                raise AssertionError("Frozen annotation/image identity mismatch")
            gt = torch.as_tensor(coco.annToMask(annotation).astype(bool), device=device)
            if tuple(gt.shape) != original:
                raise AssertionError("COCO mask and frozen original_shape differ")
            pad_gt = _padded_gt(gt, ratio_pad, shape)
            truths.append(gt); padded_truths.append(pad_gt)
            box = boxes[idx].detach().cpu().tolist()
            area = float(annotation.get("area", int(gt.sum())))
            w, h = (box[2]-box[0])/shape[1], (box[3]-box[1])/shape[0]
            row = {
                "split": str(x["split"]), "image_id": iid, "annotation_id": int(source["annotation_id"]),
                "branch": str(source.get("branch", "one2one")), "raw_id": int(source["raw_id"]),
                "pyramid_level": int(levels[idx]), "target_gt_idx": int(source.get("target_gt_idx", source.get("gt_index", -1))),
                "class_id": int(annotation["category_id"]), "class_id_system": "COCO category_id; evaluation metadata only",
                "box_xyxy": box, "box_coordinate_system": "original frozen letterbox xyxy",
                "box_w_normalized": w, "box_h_normalized": h, "box_area_normalized": w*h,
                "area": area, "size_group": "small" if area < 32**2 else "medium" if area < 96**2 else "large",
                "pixel_support_count": int(supports[offset].sum()), "pixel_support_gt_positive": int((supports[offset] & pad_gt).sum()),
                "zero_bias_padded_pixel_differences": int(padded_diff[offset]),
                "zero_bias_original_pixel_differences": int(image_diff[offset]),
            }
            if finite(source.get("box_iou")):
                row.update(box_iou=float(source["box_iou"]), box_iou_source="frozen candidate metadata")
            elif "target_boxes" in x:
                gt_box = x["target_boxes"][idx].detach().cpu().numpy()
                row.update(box_iou=_box_iou(np.asarray(box), gt_box), box_iou_source="same-candidate frozen input target_box; no rematching")
            else:
                row.update(box_iou=None, box_iou_source="missing")
            if "predicted_class_id" in source:
                row["predicted_class_id"] = int(source["predicted_class_id"])
            row.update(_donor_metadata(source, x["rows"]))
            records.append(row)

        def collect(name, masks, logits):
            for k, row in enumerate(records):
                mask, gt = masks[k], truths[k]
                intersection, union = int((mask & gt).sum()), int((mask | gt).sum())
                row[f"iou_{name}"] = intersection/max(union, 1)
                row[f"coverage_{name}"] = intersection/max(int(gt.sum()), 1)
                row[f"mask75_{name}"] = int(row[f"iou_{name}"] >= .75)
                row[f"auc_{name}"], row[f"fpr_{name}"] = _pixel_auc_fpr(logits[k], padded_truths[k], supports[k])
                row[f"empty_mask_{name}"] = not bool(mask.any())

        collect("A", base_masks, base_logits)
        for name, coeff in arms.items():
            if name == "A":
                continue
            cs = coeff[start:end]
            padded = ops.process_mask(proto, cs, bb, shape, upsample=True)
            masks = _scale_binary(padded, original, ratio_pad)
            collect(name, masks, _logits(proto, cs, shape))
            for k, row in enumerate(records):
                row[f"padded_pixel_changes_{name}_vs_A"] = int((padded[k].bool() != base_pad[k].bool()).sum())
                row[f"original_pixel_changes_{name}_vs_A"] = int((masks[k] != base_masks[k]).sum())
            del padded, masks
        for k, row in enumerate(records):
            source = x["rows"][start+k]
            row["original_status"] = "success" if row["mask75_A"] else "failure"
            row["box_good_mask_bad"] = bool(row["box_iou"] >= .75 and row["iou_A"] < .75) if finite(row["box_iou"]) else None
            if finite(source.get("initial_iou")):
                row["cached_baseline_iou_absolute_error"] = abs(row["iou_A"]-source["initial_iou"])
        out.extend(records)
    return out


def avg(xs):
    a = [float(x) for x in xs if finite(x)]
    return float(np.mean(a)) if a else None


def ci(xs):
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    return np.quantile(a, [.025, .975]).tolist() if len(a) else [None, None]


def image_groups(rows):
    d = defaultdict(list)
    for r in rows:
        d[(str(r.get("split")), str(r.get("image_id")))].append(r)
    return [d[k] for k in sorted(d)]


def metric_value(r, metric, arm):
    if metric == "mask75":
        return int(r[f"iou_{arm}"] >= .75) if finite(r.get(f"iou_{arm}")) else None
    return r.get(f"{metric}_{arm}")


def paired(gs, arm, ref, metric, seed, bootstrap, success_damage=False):
    na, sa, sb, eligible_total = [], [], [], 0
    for one in gs:
        eligible = [r for r in one if not success_damage or (finite(r.get("iou_A")) and r["iou_A"] >= .75)]
        eligible_total += len(eligible)
        pairs = []
        for r in eligible:
            a, b = metric_value(r, metric, arm), metric_value(r, metric, ref)
            if finite(a) and finite(b):
                pairs.append((1-a, 1-b) if success_damage else (a, b))
        na.append(len(pairs)); sa.append(sum(float(a) for a, _ in pairs)); sb.append(sum(float(b) for _, b in pairs))
    na, sa, sb = np.asarray(na), np.asarray(sa), np.asarray(sb)
    ok = na > 0
    ma = np.divide(sa, na, out=np.zeros(len(na)), where=ok)
    mb = np.divide(sb, na, out=np.zeros(len(na)), where=ok)
    macro_draws, candidate_draws = [], []
    if len(gs):
        rng = np.random.default_rng(seed)
        for begin in range(0, bootstrap, 128):
            ix = rng.integers(0, len(gs), (min(128, bootstrap-begin), len(gs)))
            ni, nc = ok[ix].sum(1), na[ix].sum(1)
            macro_draws.extend(np.divide((ma-mb)[ix].sum(1), ni, out=np.full(len(ix), np.nan), where=ni > 0))
            candidate_draws.extend(np.divide((sa-sb)[ix].sum(1), nc, out=np.full(len(ix), np.nan), where=nc > 0))
    return {
        "image_macro": {"arm_mean": avg(ma[ok]), "reference_mean": avg(mb[ok]), "delta": avg((ma-mb)[ok]),
                        "ci95": ci(macro_draws), "valid_images": int(ok.sum())},
        "candidate": {"arm_mean": float(sa.sum()/na.sum()) if na.sum() else None,
                      "reference_mean": float(sb.sum()/na.sum()) if na.sum() else None,
                      "delta": float((sa-sb).sum()/na.sum()) if na.sum() else None,
                      "ci95": ci(candidate_draws), "valid_candidates": int(na.sum())},
        "complete_pairs": bool(eligible_total and na.sum() == eligible_total),
        "population_candidates": eligible_total, "missing_or_undefined_candidates": int(eligible_total-na.sum()),
        "undefined_bootstrap_draws": int(np.sum(~np.isfinite(macro_draws))),
    }


def make_table(rows, seed, bootstrap, role, include_sw=False):
    gs = image_groups(rows)
    arms = (*ARMS, "SW") if include_sw else ARMS
    comparisons = (*PAIRS, ("S", "SW")) if include_sw else PAIRS
    out = {"role": role, "n_images": len(gs), "n_candidates": len(rows), "arms": arms,
           "candidate": {}, "image_macro": {}, "undefined": {}, "mask75_counts": {}, "comparisons": {}}
    for metric in METRICS:
        out["candidate"][metric] = {a: avg([metric_value(r, metric, a) for r in rows]) for a in arms}
        out["image_macro"][metric] = {a: avg([avg([metric_value(r, metric, a) for r in g]) for g in gs]) for a in arms}
        out["undefined"][metric] = {a: sum(not finite(metric_value(r, metric, a)) for r in rows) for a in arms}
    for a in arms:
        vs = [metric_value(r, "mask75", a) for r in rows if finite(metric_value(r, "mask75", a))]
        out["mask75_counts"][a] = {"success": int(sum(vs)), "defined": len(vs), "undefined": len(rows)-len(vs)}
    successes = [r for r in rows if finite(r.get("iou_A")) and r["iou_A"] >= .75]
    out["original_success_damage"] = {
        "n_candidates": len(successes), "n_images": len(image_groups(successes)),
        "counts": {a: sum(metric_value(r, "mask75", a) == 0 for r in successes) for a in arms},
        "undefined": {a: sum(not finite(metric_value(r, "mask75", a)) for r in successes) for a in arms},
    }
    for a, b in comparisons:
        p = {metric: paired(gs, a, b, metric, seed, bootstrap) for metric in METRICS}
        rs = [r for r in rows if finite(metric_value(r, "mask75", a)) and finite(metric_value(r, "mask75", b))]
        repair = sum(metric_value(r, "mask75", a) == 1 and metric_value(r, "mask75", b) == 0 for r in rs)
        damage = sum(metric_value(r, "mask75", a) == 0 and metric_value(r, "mask75", b) == 1 for r in rs)
        p["mask75_transition"] = {"reference": b, "repair": repair, "damage": damage, "net": repair-damage,
                                  "valid_candidates": len(rs), "missing_candidates": len(rows)-len(rs),
                                  "net_candidate_fraction": (repair-damage)/len(rs) if rs else None,
                                  "net_candidate_fraction_ci95": p["mask75"]["candidate"]["ci95"],
                                  "net_image_macro_fraction_ci95": p["mask75"]["image_macro"]["ci95"]}
        p["damage_on_A_success"] = paired(gs, a, b, "mask75", seed, bootstrap, success_damage=True)
        p["damage_on_A_success"]["interpretation"] = "Original A-success population; positive difference means more damage. CI crossing zero does not prove noninferiority."
        out["comparisons"][f"{a}_minus_{b}"] = p
    return out


def audit(rows, expected_counts):
    issues, notes, counts = [], [], {}
    fields = ("split", "image_id", "annotation_id", "branch", "raw_id", "pyramid_level", "target_gt_idx")
    keys = [tuple(r.get(k) for k in fields) for r in rows]
    duplicates = sum(n-1 for n in Counter(keys).values())
    if duplicates:
        issues.append(f"Duplicate permanent identities: {duplicates}")
    for k in fields:
        missing = sum(r.get(k) is None for r in rows)
        if missing:
            issues.append(f"Missing identity {k}: {missing}")
    for split in sorted({str(r.get("split")) for r in rows} | set(expected_counts or {})):
        rs = [r for r in rows if r.get("split") == split]
        observed = {"effective_images": len(image_groups(rs)), "candidates": len(rs)}
        exp = (expected_counts or {}).get(split)
        if exp is not None:
            if not isinstance(exp, dict):
                raise TypeError("expected_counts values must be dicts with effective_images/images, candidates, optional planned_images")
            ni = exp.get("effective_images", exp.get("images"))
            nc = exp.get("candidates")
            if ni is not None and int(ni) != observed["effective_images"]:
                issues.append(f"{split}: effective image count does not match frozen manifest")
            if nc is not None and int(nc) != observed["candidates"]:
                issues.append(f"{split}: candidate count does not match frozen manifest")
            observed.update(expected=exp, planned_images=exp.get("planned_images"), count_verified=ni is not None and nc is not None)
        else:
            observed.update(expected=None, planned_images=None, count_verified=False)
            notes.append(f"{split}: expected population count not supplied; observed counts only")
        counts[split] = observed
    if "val" not in counts or not counts["val"]["candidates"]:
        issues.append("No val candidates for the fixed primary comparison")
    for a in (*ARMS, "SW"):
        rs = rows if a != "SW" else [r for r in rows if r.get("wrong_box_available") is True]
        bad = sum(not finite(r.get(f"iou_{a}")) or not 0 <= r[f"iou_{a}"] <= 1 for r in rs)
        if bad:
            issues.append(f"Required IoU_{a} missing/nonfinite/outside [0,1]: {bad}")
        inconsistent = sum(finite(r.get(f"mask75_{a}")) and finite(r.get(f"iou_{a}")) and
                           int(r[f"mask75_{a}"]) != int(r[f"iou_{a}"] >= .75) for r in rs)
        if inconsistent:
            issues.append(f"Mask75_{a} disagrees with original-image IoU: {inconsistent}")
    if any(not finite(r.get("box_iou")) for r in rows if r.get("split") == "val"):
        issues.append("Undefined val Box IoU; primary population is not fully determined")
    for k in ("zero_bias_padded_pixel_differences", "zero_bias_original_pixel_differences"):
        if any(r.get(k, 0) != 0 for r in rows):
            issues.append(f"Nonzero baseline replay discrepancy: {k}")
    cached = [r["cached_baseline_iou_absolute_error"] for r in rows if finite(r.get("cached_baseline_iou_absolute_error"))]
    if cached and max(cached) > 1e-6:
        issues.append("Cached baseline IoU error exceeds 1e-6")
    same_raw = [r for r in rows if r.get("wrong_box_available") is True and
                r.get("wrong_box_donor_raw_id") == r.get("raw_id") and
                r.get("wrong_box_donor_image_id", r.get("image_id")) == r.get("image_id")]
    if same_raw:
        notes.append(f"SW marked available but points to same raw candidate: {len(same_raw)}; no independent wrong-instance claim")
    image_sets = {s: {r["image_id"] for r in rows if r.get("split") == s} for s in counts}
    for a, b in (("fit", "dev"), ("fit", "val"), ("dev", "val")):
        if image_sets.get(a, set()) & image_sets.get(b, set()):
            issues.append(f"Image identity overlap between {a} and {b}")
    return {"status": "passed" if not issues else "incomplete_or_inconsistent", "issues": issues, "notes": notes,
            "counts": counts, "identity_fields": fields, "duplicate_identities": duplicates, "candidate_rows_dropped": 0,
            "baseline_replay_audited_candidates": sum("zero_bias_original_pixel_differences" in r for r in rows),
            "cached_baseline_iou_checked_candidates": len(cached), "cached_baseline_iou_max_error": max(cached) if cached else None}


def assess(primary, versus_original, guard, integrity, smoke=False):
    """Fixed conjunctive method decision; other metrics remain descriptive."""
    point = primary["image_macro"]["delta"]
    lower = primary["image_macro"]["ci95"][0]
    base_lower = versus_original["image_macro"]["ci95"][0]
    guard_lower = guard["image_macro"]["ci95"][0]
    complete = not integrity["issues"] and all(
        p["complete_pairs"] and finite(p["image_macro"]["delta"])
        and all(finite(v) for v in p["image_macro"]["ci95"])
        for p in (primary, versus_original, guard))
    practical = bool(point >= .005) if finite(point) else False
    primary_positive = bool(lower > 0) if finite(lower) else False
    exceeds_original = bool(base_lower > 0) if finite(base_lower) else False
    guard_passed = bool(guard_lower > -.001) if finite(guard_lower) else False
    passed = bool(complete and primary_positive and exceeds_original and guard_passed and not smoke)
    if smoke:
        statement = "零初始化解码smoke；不作方法成败或性能守护判断。"
    elif not complete:
        statement = "输入核对或正式比较不完整，本轮不通过；保留产物并停止。"
    elif passed:
        statement = "本轮固定对照通过：目标组联合投影可靠超过普通原生微调及原模型，且全体性能守护通过。"
    else:
        statement = "本轮未同时通过固定收益与全体性能守护条件，按有限协议停止。"
    return {
        "statement": statement, "method_passed": passed, "decision": "CODE_SMOKE_ONLY" if smoke else "PASS_FIXED_PILOT" if passed else "STOP_FIXED_PILOT",
        "complete_and_audited": bool(complete), "practical_primary_effect": .005,
        "primary_point_reaches_practical_effect": practical,
        "practical_effect_is_descriptive_not_mandatory": True,
        "primary_paired_ci_lower_above_zero": primary_positive,
        "primary_S_minus_A_ci_lower_above_zero": exceeds_original,
        "all_val_guard_margin": -.001, "all_val_S_minus_A_ci_lower": guard_lower,
        "all_val_guard_passed": bool(complete and guard_passed and not smoke),
        "automatic_next_experiment": False, "stop_after_reporting": True,
        "interpretation": "主对象S−N的95%CI下界>0，同时该组S−A下界>0；+0.5 IoU百分点只报告实用量级是否达到，不作为硬否决；全val S−A的95%CI下界须>−0.1百分点，排除更差于预设margin才称守护通过。其它指标全部报告，不要求全部改善。失败即停止，不自动调参或扩训。",
    }


def fmt(x, pp=False, signed=False):
    return format(float(x)*(100 if pp else 1), "+.4f" if signed else ".4f") if finite(x) else "未定义"


def effect(p, kind="image_macro"):
    s = p[kind]
    return f"{fmt(s['delta'], True, True)} [{fmt(s['ci95'][0], True, True)}, {fmt(s['ci95'][1], True, True)}]"


def report(result):
    primary = result["tables"][PRIMARY_GROUP]
    p = primary["comparisons"][PRIMARY_COMPARISON]
    lines = ["# 原生系数分支与框内空间证据联合训练：固定候选评价", "", result["assessment"]["statement"], "",
             f"主对象为val中原Box IoU≥0.75且原图Mask IoU<0.75，共{primary['n_images']}张有效图片、{primary['n_candidates']}个候选。主比较S−N图片macro IoU差为{effect(p['iou'])}个百分点。",
             result["assessment"]["interpretation"], "", "|组|含义|", "|---|---|"]
    lines += [f"|{a}|{label}|" for a, label in LABELS.items()]
    lines += ["", "冻结官方one-to-one TAL候选、neck输出、原型、预测框、类别、分数及解码。N/S/D均重新运行并训练原生系数分支；S/D再联合训练新增框证据分支。固定最后第12轮checkpoint；smoke使用零初始化而非训练checkpoint。本评价不是完整推理COCO AP，也不声称新的盲测。", "",
              "## 冻结评价范围", "", "|split|计划图片|有效图片|候选|数量核对|", "|---|---:|---:|---:|---|"]
    for split, c in result["audit"]["counts"].items():
        lines.append(f"|{split}|{c['planned_images'] if c['planned_images'] is not None else '未提供'}|{c['effective_images']}|{c['candidates']}|{c['count_verified']}|")
    lines += ["", "正式评价仅dev前1000图、val前2000图及预先固定顺序fit前1000图；smoke仅fit前8/dev前3/val前3。均先按INDEX顺序截取，再保留候选，不按质量筛图。无候选图保留清单但不伪造IoU。旧全量fit本轮未重评；训练记录的全fit trajectory loss由变化中的参数产生，不是最后checkpoint的全fit损失。", "",
              "## 配对原图掩码结果", "", "差值和区间单位为百分点；候选均值CI同样整图重采样。", "",
              "|范围|图片/候选|比较|图片macro IoU差 [95% CI]|候选IoU差 [95% CI]|Mask75修复/损伤/净增|", "|---|---:|---|---:|---:|---:|"]
    names = [PRIMARY_GROUP, "val:all"]+[f"{s}:all" for s in ("fit", "dev") if f"{s}:all" in result["tables"]]
    for name in names:
        t = result["tables"][name]
        for a, b in PAIRS:
            z = t["comparisons"][f"{a}_minus_{b}"]; tr = z["mask75_transition"]
            lines.append(f"|{name}|{t['n_images']}/{t['n_candidates']}|{a}−{b}|{effect(z['iou'])}|{effect(z['iou'], 'candidate')}|{tr['repair']}/{tr['damage']}/{tr['net']:+d}|")
    lines += ["", "S−SF由历史冻结头S的同身份指标直接配对，不重复运行旧模型。两配置训练并非同一批优化器过程，新增原生分支学习率1e-4；此比较说明联合与冻结配置的效用差，不能替代同轮S−N控制或单独证明优化机制。", "", "## 连续监督损失", "", "BCE按缓存的官方实例polygon/owner标签、GT目标框支持、area归一化和segmentation gain计算；与原图COCO GT掩码指标的评价对象相同、标签口径不同。下表是本次固定checkpoint评价子集的候选平均，不是训练轨迹loss。", "",
              "|split|候选|A BCE|N BCE|D BCE|S BCE|历史SF BCE|", "|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("fit", "dev", "val"):
        t = result["tables"].get(f"{split}:all")
        if t:
            values = t["candidate"]["bce"]
            lines.append(f"|{split}|{t['n_candidates']}|{fmt(values['A'])}|{fmt(values['N'])}|{fmt(values['D'])}|{fmt(values['S'])}|{fmt(values['SF'])}|")
    lines += ["", "## 次要质量与原成功损伤", "", "以下S−N图片macro差为百分点；完整各组值、候选均值和CI见JSON/CSV。", "",
              "|范围|Coverage差 [95% CI]|AUC差 [95% CI]|FPR差 [95% CI]|S/N在A原成功上的损伤数|S−N损伤率 [候选95% CI]|", "|---|---:|---:|---:|---:|---:|"]
    for name in names:
        t = result["tables"][name]; z = t["comparisons"]["S_minus_N"]; damage = t["original_success_damage"]
        lines.append(f"|{name}|{effect(z['coverage'])}|{effect(z['auc'])}|{effect(z['fpr'])}|{damage['counts']['S']}/{damage['counts']['N']}，母数{damage['n_candidates']}|{effect(z['damage_on_A_success'], 'candidate')}|")
    lines += ["", "AUC使用同一固定预测框支持内的连续logit，COCO GT按nearest缩放填充。未定义AUC单列计数；FPR包括邻居及背景，不是纯背景泄漏。损伤率正差表示S更差，CI跨零不证明非劣。", "",
              "## 预声明val描述分层", "", "|分层|图片/候选|S−N macro IoU差 [95% CI]|S相对A修复/损伤/净增|", "|---|---:|---:|---:|"]
    for name, t in result["tables"].items():
        if not name.startswith("val:") or name in names or ":SW_" in name:
            continue
        z = t["comparisons"]["S_minus_N"]; tr = t["comparisons"]["S_minus_A"]["mask75_transition"]
        lines.append(f"|{name}|{t['n_images']}/{t['n_candidates']}|{effect(z['iou'])}|{tr['repair']}/{tr['damage']}/{tr['net']:+d}|")
    lines += ["", "大小用原COCO area阈值；P3/P4/P5是候选层级。原成功/失败一律由A固定。", "",
              "## 冻结S的错框诊断", "", "供体仅按预测类别、框面积/长宽比与raw ID确定，事后才标注GT关系。SW只替换sampling_boxes；自身当前native h/query/c、A/K/G及最终解码框保持原样。可用供体子集与同GT/不同GT/未知GT分别报告。", "",
              "|范围|图片/候选|S−SW macro IoU差 [95% CI]|S−SW候选IoU差 [95% CI]|", "|---|---:|---:|---:|"]
    for name, t in result["tables"].items():
        if ":SW_" in name:
            z = t["comparisons"]["S_minus_SW"]
            lines.append(f"|{name}|{t['n_images']}/{t['n_candidates']}|{effect(z['iou'])}|{effect(z['iou'], 'candidate')}|")
    lines += ["", "true优于wrong只说明本头依赖区域对应，错框可能分布外，不能替代S−N、S−A和全体守护。", "",
              "## 完整性与结束", "", f"输入审计：{result['audit']['status']}。整图配对bootstrap {result['bootstrap']}次，seed={result['seed']}。主条件按固定交集判定，其它CI是未作多重比较校正的描述性结果。"]
    lines += [f"- {v}" for v in result["audit"]["issues"]+result["audit"]["notes"]]
    lines += ["", "完成本轮后停止，不根据val效果追加结构、门控、正则扫描或训练轮数。"]
    return "\n".join(lines)+"\n"


def summarize(rows, out, expected_counts=None, seed=20261003, bootstrap=5000, run_info=None):
    rows = list(rows)
    if not rows or bootstrap <= 0:
        raise ValueError("Nonempty rows and positive bootstrap required")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    tables, wrong_summary = {}, {}
    splits = [s for s in ("fit", "dev", "val") if any(r.get("split") == s for r in rows)]
    for split in splits:
        rs = [r for r in rows if r.get("split") == split]
        tables[f"{split}:all"] = make_table(rs, seed, bootstrap, "all candidates; val S-A has fixed noninferiority margin -.001")
        available = [r for r in rs if r.get("wrong_box_available") is True]
        rels = {rel: [r for r in available if r.get("wrong_box_gt_relation") == rel]
                for rel in ("different_gt", "same_gt", "unknown_gt")}
        classified = sum(map(len, rels.values()))
        unclassified = [r for r in available if r.get("wrong_box_gt_relation") not in rels]
        rels["unknown_gt"] += unclassified
        wrong_summary[split] = {"available": len(available), "unavailable_or_unknown": len(rs)-len(available),
                                "donor_GT_relation_counts": {k: len(v) for k,v in rels.items()},
                                "relation_missing_or_unrecognized": len(unclassified)}
        for name, subset in {"available": available, **rels}.items():
            tables[f"{split}:SW_{name}"] = make_table(subset, seed, bootstrap, "available-only frozen-S mismatch control; descriptive", True)
    vr = [r for r in rows if r.get("split") == "val"]
    primary = [r for r in vr if finite(r.get("box_iou")) and r["box_iou"] >= .75 and finite(r.get("iou_A")) and r["iou_A"] < .75]
    subsets = {"box_good_mask_bad": primary,
               "original_success": [r for r in vr if finite(r.get("iou_A")) and r["iou_A"] >= .75],
               "original_failure": [r for r in vr if finite(r.get("iou_A")) and r["iou_A"] < .75]}
    subsets.update({f"P{l+3}": [r for r in vr if r.get("pyramid_level") == l] for l in (0,1,2)})
    subsets.update({f"size_{s}": [r for r in vr if r.get("size_group") == s] for s in ("small", "medium", "large")})
    for name, rs in subsets.items():
        tables[f"val:{name}"] = make_table(rs, seed, bootstrap, "primary prespecified failure stratum" if name == "box_good_mask_bad" else "prespecified descriptive stratum")
    for name in ("available", "different_gt", "same_gt", "unknown_gt"):
        subset = [r for r in primary if r.get("wrong_box_available") is True and
                  (name == "available" or r.get("wrong_box_gt_relation", "unknown_gt") == name)]
        tables[f"val:SW_primary_{name}"] = make_table(subset, seed, bootstrap, "primary stratum mismatch control; descriptive", True)
    integrity = audit(rows, expected_counts)
    result = {
        "schema": "box_evidence_joint_eval_v1", "arms": LABELS, "audit": integrity,
        "seed": seed, "bootstrap": bootstrap, "units": "quality metrics are fractions; times100 is percentage points; BCE is native loss units",
        "primary_group": PRIMARY_GROUP, "primary_comparison": PRIMARY_COMPARISON,
        "primary_metric": "original-image Mask IoU, image macro", "secondary_comparison": "S_minus_A",
        "mechanism_comparison": "S_minus_D", "historical_comparison": "S_minus_SF",
        "guardrail": "val:all S_minus_A image-macro IoU CI lower > -.001; other metrics descriptive",
        "intervals": "paired whole-image percentile95%; other comparisons descriptive, not multiplicity-adjusted",
        "checkpoint_policy": "fixed final checkpoint selected by training protocol; evaluator does not choose checkpoint",
        "run_info": run_info, "expected_counts": expected_counts, "wrong_box_control": wrong_summary,
        "decode": "full proto -> process_mask(upsample=True) -> input binary mask -> scale_masks(real ratio_pad) -> >.5",
        "metric_label": "IoU/coverage: raw original COCO annToMask; AUC/FPR: same GT nearest letterbox, continuous640 logits on original prediction-box support",
        "scope": "fixed official one-to-one TAL candidates; not full-output COCO AP; not asserted as a new blind test",
        "tables": tables, "automatic_followup": False,
    }
    result["assessment"] = assess(tables[PRIMARY_GROUP]["comparisons"][PRIMARY_COMPARISON]["iou"],
        tables[PRIMARY_GROUP]["comparisons"]["S_minus_A"]["iou"],
        tables["val:all"]["comparisons"]["S_minus_A"]["iou"], integrity,
        smoke=bool((run_info or {}).get("smoke")))
    write_json(out/"RESULTS.json", result)
    (out/"REPORT.md").write_text(report(result), encoding="utf-8")
    with (out/"PER_IMAGE.jsonl").open("w", encoding="utf-8") as stream:
        for one in image_groups(rows):
            rec = {"split": one[0].get("split"), "image_id": one[0].get("image_id"), "n_candidates": len(one)}
            for arm in (*ARMS, "SW"):
                ar = one if arm != "SW" else [r for r in one if r.get("wrong_box_available") is True]
                for metric in METRICS:
                    values = [metric_value(r, metric, arm) for r in ar]
                    rec[f"{metric}_{arm}"] = avg(values)
                    rec[f"n_defined_{metric}_{arm}"] = sum(finite(v) for v in values)
            bg = [r for r in one if finite(r.get("box_iou")) and r["box_iou"] >= .75 and finite(r.get("iou_A")) and r["iou_A"] < .75]
            rec["n_box_good_mask_bad"] = len(bg)
            rec["box_good_mask_bad"] = {f"{metric}_{arm}": avg([metric_value(r, metric, arm) for r in bg]) for arm in ARMS for metric in METRICS}
            rec["n_wrong_box_available"] = sum(r.get("wrong_box_available") is True for r in one)
            stream.write(json.dumps(clean(rec), ensure_ascii=False, allow_nan=False)+"\n")
    with (out/"ABLATION_TABLE.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("stratum","role","images","candidates","comparison","metric","statistic","arm_mean","reference_mean","delta","ci95_low","ci95_high","undefined","repair","damage","net_mask75"))
        for name, t in tables.items():
            for comp, p in t["comparisons"].items():
                tr = p["mask75_transition"]
                for metric in METRICS:
                    for kind in ("image_macro", "candidate"):
                        s = p[metric][kind]
                        writer.writerow((name,t["role"],t["n_images"],t["n_candidates"],comp,metric,kind,s["arm_mean"],s["reference_mean"],s["delta"],*s["ci95"],p[metric]["missing_or_undefined_candidates"],tr["repair"],tr["damage"],tr["net"]))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expected-counts", type=Path)
    ap.add_argument("--run-info", type=Path)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--bootstrap", type=int, default=5000)
    a = ap.parse_args()
    rows = [json.loads(line) for line in a.input.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    expected = json.loads(a.expected_counts.read_text(encoding="utf-8-sig")) if a.expected_counts else None
    info = json.loads(a.run_info.read_text(encoding="utf-8-sig")) if a.run_info else None
    result = summarize(rows, a.out, expected, a.seed, a.bootstrap, info)
    print(json.dumps(clean(result["assessment"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
