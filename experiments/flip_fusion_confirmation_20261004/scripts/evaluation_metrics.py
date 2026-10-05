"""Fixed-official-candidate metrics for same-budget prototype-guided selection.

Public APIs:
  evaluate_image(x, coeffs, coco, chunk_size=4) -> list[dict]
  append_rows(path, rows) -> number_written
  summarize(rows, out, expected_counts=None, seed=20261003, bootstrap=5000,
            run_info=None) -> dict

``x`` is the compact prepared-image schema, not raw-indexed output. The runner
replays the evidence readout branches and supplies [N,32] coefficients for
A, U, Q, P; historical N metrics may be joined without model replay. This module never runs/reselects assignment, chooses
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


ARMS = ("A", "U", "Q", "P")
METRICS = ("iou", "mask75", "coverage", "auc", "fpr", "bce")
PAIRS = (("P", "Q"), ("P", "U"), ("P", "A"), ("Q", "A"), ("U", "A"))
PRIMARY_GROUP = "val:box_good_mask_bad"
PRIMARY_COMPARISON = "P_minus_Q"
LABELS = {
    "A": "原始冻结YOLO系数",
    "U": "同预算均匀64格＋联合系数分支",
    "Q": "原始logit不确定性最高64格＋联合系数分支",
    "P": "原型响应互补性与不确定性选64格＋联合系数分支",
}


def enable_historical_native():
    global ARMS, PAIRS
    if "N" not in ARMS:
        ARMS = (*ARMS, "N")
        PAIRS = (*PAIRS, ("P", "N"))
        LABELS["N"] = "历史联合训练Study中的原生系数分支普通微调；同身份指标复用"



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
        raise ValueError("Supply original A and coefficient arms from U/Q/P")
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


def assess(primary, original, guard, integrity, smoke=False):
    point = primary["image_macro"]["delta"]
    lower = primary["image_macro"]["ci95"][0]
    original_lower = original["image_macro"]["ci95"][0]
    guard_lower = guard["image_macro"]["ci95"][0]
    complete = not integrity["issues"] and all(
        item["complete_pairs"] and finite(item["image_macro"]["delta"])
        and all(finite(v) for v in item["image_macro"]["ci95"])
        for item in (primary, original, guard))
    better_q = bool(finite(lower) and lower > 0)
    better_a = bool(finite(original_lower) and original_lower > 0)
    guard_passed = bool(finite(guard_lower) and guard_lower > -.001)
    passed = bool(complete and better_q and better_a and guard_passed and not smoke)
    if smoke:
        statement = "初始化解码核对，不作正式方法判断。"
    elif not complete:
        statement = "评价身份、覆盖或统计输入未完整核对，本轮尚未完成；保留记录并停止。"
    elif passed:
        statement = "在本轮固定候选上，原型选点相对不确定性选点及原模型显示目标组掩码收益，全体性能守护通过。"
    else:
        statement = "本轮未建立预设的目标组增益与全体性能守护组合，按有限协议结束，不自动追加结构或训练。"
    return dict(statement=statement, method_passed=passed,
        decision="CODE_SMOKE_ONLY" if smoke else "PASS_FIXED_PILOT" if passed else "STOP_FIXED_PILOT",
        complete_and_audited=bool(complete), primary_P_minus_Q_ci_positive=better_q,
        target_P_minus_A_ci_positive=better_a, all_val_P_minus_A_ci_lower=guard_lower,
        all_val_guard_margin=-.001, all_val_guard_passed=guard_passed,
        practical_primary_effect=.005,
        primary_point_reaches_practical_effect=bool(finite(point) and point >= .005),
        practical_effect_is_descriptive_not_mandatory=True,
        interpretation="主目标组P−Q图片macro IoU区间下界>0，目标组P−A下界>0，全val P−A下界>−0.1百分点。0.5百分点只衡量效应量级，不作硬否决。P−U及其它指标如实报告，不要求所有指标同时改善。",
        automatic_next_experiment=False, stop_after_reporting=True)


def fmt(value, pp=False, signed=False):
    return format(float(value) * (100 if pp else 1), "+.4f" if signed else ".4f") if finite(value) else "未定义"


def effect(value, statistic="image_macro"):
    item = value[statistic]
    return f"{fmt(item['delta'], True, True)} [{fmt(item['ci95'][0], True, True)}, {fmt(item['ci95'][1], True, True)}]"


def report(result):
    main = result["tables"][PRIMARY_GROUP]
    primary = main["comparisons"][PRIMARY_COMPARISON]
    lines = ["# 原型引导的同预算证据选点：正常原图掩码评价", "",
        result["assessment"]["statement"], "",
        f"主目标组为原Box IoU≥0.75且原图Mask IoU<0.75：{main['n_images']}张图片、{main['n_candidates']}个官方one-to-one候选。P−Q图片macro IoU差为{effect(primary['iou'])}个百分点。",
        result["assessment"]["interpretation"], "", "|组|定义|", "|---|---|"]
    lines += [f"|{arm}|{meaning}|" for arm, meaning in LABELS.items()]
    lines += ["", "三臂从同一原权重开始，训练原生one-to-one系数分支与相同点证据网络；只改变冻结的64格选点和相应原型投影矩阵。原型、预测框、分类、原官方TAL候选与解码固定。固定第12轮，不以dev/val选择checkpoint。", "",
        "## 评价范围", "", "|分片|计划图|有效图|候选|数量核对|", "|---|---:|---:|---:|---|"]
    for split, row in result["audit"]["counts"].items():
        lines.append(f"|{split}|{row['planned_images']}|{row['effective_images']}|{row['candidates']}|{row['count_verified']}|")
    lines += ["", "fit仅评价预先固定INDEX前1000张计划图片，dev前1000、val前2000；无正样本图保留清单，不据结果删图。本val此前已被研究查看，不是新盲测；本报告为固定GT条件官方正样本评价，不报告完整推理COCO AP。", "",
        "## 配对原图掩码结果", "", "所有差值和置信区间为百分点，候选均值区间也整张图片重采样。", "",
        "|范围|图/候选|比较|图片macro IoU差 [95% CI]|候选IoU差 [95% CI]|Mask75修复/损伤/净增|", "|---|---:|---|---:|---:|---:|"]
    names = [PRIMARY_GROUP, "val:all", "dev:all", "fit:all"]
    for name in names:
        table = result["tables"].get(name)
        if table is None:
            continue
        for arm, reference in PAIRS:
            pair = table["comparisons"][f"{arm}_minus_{reference}"]
            tr = pair["mask75_transition"]
            lines.append(f"|{name}|{table['n_images']}/{table['n_candidates']}|{arm}−{reference}|{effect(pair['iou'])}|{effect(pair['iou'], 'candidate')}|{tr['repair']}/{tr['damage']}/{tr['net']:+d}|")
    lines += ["", "历史N只通过同身份逐候选记录配对，不重新执行模型。若有该组，P−N是跨配置历史参考，不能替代本轮同参数量、同预算P−Q/P−U。", "",
        "## 官方监督损失", "", "使用原polygon overlap标签、完整GT框支持、面积归一化和原segmentation gain。损失与原图COCO GT评价口径分开。以下是固定末轮评价子集均值，不是训练中的变化参数轨迹loss。", "",
        "|分片|候选|" + "|".join(f"{arm} BCE" for arm in ARMS) + "|",
        "|---|---:|" + "|".join("---:" for _ in ARMS) + "|"]
    for split in ("fit", "dev", "val"):
        table = result["tables"].get(f"{split}:all")
        if table:
            lines.append(f"|{split}|{table['n_candidates']}|" + "|".join(fmt(table['candidate']['bce'][arm]) for arm in ARMS) + "|")
    lines += ["", "## Coverage、排序与误报", "", "图片macro差值为百分点。AUC取固定预测框支持内连续logit，GT由原COCO掩码nearest letterbox获得；FPR包含所有GT外像素，不等同纯背景。", "",
        "|范围|比较|Coverage差 [95% CI]|AUC差 [95% CI]|FPR差 [95% CI]|", "|---|---|---:|---:|---:|"]
    for name in names:
        if name not in result["tables"]:
            continue
        for arm, reference in PAIRS:
            pair = result["tables"][name]["comparisons"][f"{arm}_minus_{reference}"]
            lines.append(f"|{name}|{arm}−{reference}|{effect(pair['coverage'])}|{effect(pair['auc'])}|{effect(pair['fpr'])}|")
    lines += ["", "## 预声明val描述分层", "", "|分层|图/候选|P−Q macro IoU差 [95% CI]|P−A修复/损伤/净增|", "|---|---:|---:|---:|"]
    for name, table in result["tables"].items():
        if not name.startswith("val:") or name in names:
            continue
        tr = table["comparisons"]["P_minus_A"]["mask75_transition"]
        lines.append(f"|{name}|{table['n_images']}/{table['n_candidates']}|{effect(table['comparisons']['P_minus_Q']['iou'])}|{tr['repair']}/{tr['damage']}/{tr['net']:+d}|")
    lines += ["", "成功/失败由A固定，P3/P4/P5为候选层级而不是目标尺寸；small/medium/large来自原COCO area。分层结果不能替代全体或主要比较。", "",
        "## 审计与结束", "", f"审计状态：{result['audit']['status']}。图片配对bootstrap={result['bootstrap']}，seed={result['seed']}。其它区间为描述性区间，未进行多重比较校正。"]
    lines += [f"- {note}" for note in result["audit"]["issues"] + result["audit"]["notes"]]
    lines += ["", "本轮结束后停止。矩阵条件、GT辅助上界或训练loss均不能代替正常掩码效用，不自动追加点数、网络、门控或训练预算。"]
    return "\n".join(lines) + "\n"


def summarize(rows, out, expected_counts=None, seed=20261003, bootstrap=5000, run_info=None):
    rows = list(rows)
    if not rows or bootstrap <= 0:
        raise ValueError("Nonempty evaluated rows and positive bootstrap required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {}
    for split in ("fit", "dev", "val"):
        subset = [row for row in rows if row["split"] == split]
        if subset:
            tables[f"{split}:all"] = make_table(subset, seed, bootstrap, "full prespecified evaluation subset")
    val = [row for row in rows if row["split"] == "val"]
    subsets = {
        "box_good_mask_bad": [r for r in val if finite(r.get("box_iou")) and r["box_iou"] >= .75 and r["iou_A"] < .75],
        "original_success": [r for r in val if r["iou_A"] >= .75],
        "original_failure": [r for r in val if r["iou_A"] < .75],
    }
    subsets.update({f"P{level+3}": [r for r in val if r["pyramid_level"] == level] for level in (0, 1, 2)})
    subsets.update({f"size_{size}": [r for r in val if r["size_group"] == size] for size in ("small", "medium", "large")})
    for name, subset in subsets.items():
        tables[f"val:{name}"] = make_table(subset, seed, bootstrap,
            "primary prespecified stratum" if name == "box_good_mask_bad" else "prespecified descriptive stratum")
    integrity = audit(rows, expected_counts)
    info = run_info or {}
    if not info.get("historical_native_joined", False):
        integrity["notes"].append("Historical native N rows unavailable; no N comparison is fabricated or rerun.")
    result = dict(schema="prototype-guided-evidence-selection-eval-v1", arms=LABELS,
        audit=integrity, seed=seed, bootstrap=bootstrap, tables=tables, run_info=info,
        primary_group=PRIMARY_GROUP, primary_comparison=PRIMARY_COMPARISON,
        primary_metric="original-image Mask IoU, image macro",
        support_comparisons=["P_minus_U", "P_minus_A"],
        units="quality values are fractions; times100 is percentage points, not COCO AP; BCE is native loss units",
        intervals="paired whole-image percentile95%; candidate means also resample whole images",
        expected_counts=expected_counts,
        decode="original full proto -> process_mask(upsample=True) -> input binary mask -> scale_masks(real ratio_pad) -> >.5",
        metric_label="IoU/coverage: original COCO annToMask; AUC/FPR: same GT nearest letterbox, continuous640 logits on fixed prediction-box support",
        scope="official fixed one-to-one TAL candidates; historically viewed evaluation images; no complete COCO AP",
        automatic_followup=False)
    result["assessment"] = assess(tables[PRIMARY_GROUP]["comparisons"][PRIMARY_COMPARISON]["iou"],
        tables[PRIMARY_GROUP]["comparisons"]["P_minus_A"]["iou"],
        tables["val:all"]["comparisons"]["P_minus_A"]["iou"], integrity, bool(info.get("smoke")))
    write_json(out / "RESULTS.json", result)
    (out / "REPORT.md").write_text(report(result), encoding="utf-8")
    with (out / "PER_IMAGE.jsonl").open("w", encoding="utf-8") as stream:
        for group in image_groups(rows):
            record = dict(split=group[0]["split"], image_id=group[0]["image_id"], n_candidates=len(group))
            for arm in ARMS:
                for metric in METRICS:
                    values = [metric_value(r, metric, arm) for r in group]
                    record[f"{metric}_{arm}"] = avg(values)
                    record[f"n_defined_{metric}_{arm}"] = sum(finite(v) for v in values)
            target = [r for r in group if finite(r.get("box_iou")) and r["box_iou"] >= .75 and r["iou_A"] < .75]
            record["n_box_good_mask_bad"] = len(target)
            record["box_good_mask_bad"] = {f"{metric}_{arm}": avg([metric_value(r, metric, arm) for r in target])
                                          for arm in ARMS for metric in METRICS}
            stream.write(json.dumps(clean(record), ensure_ascii=False, allow_nan=False) + "\n")
    with (out / "ABLATION_TABLE.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("stratum", "images", "candidates", "comparison", "metric", "statistic", "arm_mean", "reference_mean", "delta", "ci95_low", "ci95_high", "undefined", "repair", "damage", "net_mask75"))
        for name, table in tables.items():
            for comparison, pair in table["comparisons"].items():
                tr = pair["mask75_transition"]
                for metric in METRICS:
                    for statistic in ("image_macro", "candidate"):
                        item = pair[metric][statistic]
                        writer.writerow((name, table["n_images"], table["n_candidates"], comparison, metric, statistic,
                            item["arm_mean"], item["reference_mean"], item["delta"], *item["ci95"],
                            pair[metric]["missing_or_undefined_candidates"], tr["repair"], tr["damage"], tr["net"]))
    return result

