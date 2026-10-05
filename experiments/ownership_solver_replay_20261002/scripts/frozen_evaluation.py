"""Frozen-candidate original-image evaluation for the PCDCR experiment.

Public API
----------
evaluate_image(x, coeffs, coco, scalar_bias=None, chunk_size=4) -> list[dict]
    ``x`` contains compact candidate tensors, NOT full raw-indexed outputs.
    ``coeffs`` maps arm names to [N, 32] tensors and must contain baseline A.
    A scalar bias adds arm S with z_S = z_A + scalar_bias on fixed box support.
    The function checks scalar_bias=0 replay against native process_mask.
summarize(rows, out, seed=0, bootstrap=5000) -> dict
    Writes RESULTS.json, PER_IMAGE.json, PER_CANDIDATE.jsonl and
    ABLATION_TABLE.csv. No fitting, checkpoint selection or gating is performed.

All IoU and coverage labels are original COCO instance masks. AUC/FPR use
continuous input-resolution logits and nearest-resized original COCO labels
inside the same frozen predicted-box support in every arm. Undefined AUC is
retained as NaN; its candidate remains in all other evaluations.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils import ops


ARMS = ("A", "B", "C", "D", "M", "S")
METRICS = ("iou", "mask75", "coverage", "auc", "fpr")
PRIMARY_COMPARISONS = (("D", "B"), ("D", "C"), ("D", "M"), ("D", "A"), ("D", "S"))


def pixel_auc_fpr(logits, labels, support):
    """Tie-correct AUROC and zero-logit FPR on a fixed pixel support."""
    scores = logits[support].detach().cpu().numpy().astype(np.float64, copy=False)
    truth = labels[support].detach().cpu().numpy().astype(bool, copy=False)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite continuous logits; do not silently drop candidate.")
    positives, negatives = int(truth.sum()), int((~truth).sum())
    fpr = float(((scores > 0) & ~truth).sum() / negatives) if negatives else float("nan")
    if not positives or not negatives:
        return float("nan"), fpr
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    ranks = np.repeat((starts + ends + 1) * 0.5, ends - starts)
    auc = (ranks[truth[order]].sum() - positives * (positives + 1) * 0.5) / (positives * negatives)
    return float(auc), fpr


def _padded_gt(original, ratio_pad, input_shape):
    """Original binary GT -> the frozen input letterbox, nearest interpolation."""
    oh, ow = original.shape
    ih, iw = input_shape
    if ratio_pad is None:
        gain = min(ih / oh, iw / ow)
        nh, nw = round(oh * gain), round(ow * gain)
        left, top = round((iw - nw) / 2 - 0.1), round((ih - nh) / 2 - 0.1)
    else:
        ratios, pads = ratio_pad
        if isinstance(ratios, (float, int)):
            rh = rw = float(ratios)
        else:
            rh, rw = map(float, ratios)
        nh, nw = round(oh * rh), round(ow * rw)
        left, top = round(float(pads[0]) - 0.1), round(float(pads[1]) - 0.1)
    right, bottom = iw - nw - left, ih - nh - top
    if min(nh, nw) <= 0 or min(left, top, right, bottom) < 0:
        raise ValueError(f"Invalid frozen letterbox geometry: {original.shape}, {ratio_pad}, {input_shape}")
    resized = F.interpolate(original.float()[None, None], (nh, nw), mode="nearest")
    return F.pad(resized, (left, right, top, bottom))[0, 0].bool()


def _logits(proto, coefficients, input_shape):
    spatial = (coefficients @ proto.float().flatten(1)).reshape(-1, *proto.shape[-2:])
    return F.interpolate(spatial[None], input_shape, mode="bilinear", align_corners=False)[0]


def _scale_binary(padded, original_shape, ratio_pad):
    return ops.scale_masks(padded[None], original_shape, ratio_pad=ratio_pad)[0] > 0.5


def zero_bias_replay(proto, coefficients, boxes, input_shape, original_shape, ratio_pad):
    """Return native/manual zero-bias masks and a pixel-exact replay audit."""
    native_pad = ops.process_mask(proto, coefficients, boxes, input_shape, upsample=True)
    logits = _logits(proto, coefficients, input_shape)
    manual_pad = ops.crop_mask(logits.clone(), boxes).gt(0).byte()
    native = _scale_binary(native_pad, original_shape, ratio_pad)
    manual = _scale_binary(manual_pad, original_shape, ratio_pad)
    pad_diff = (native_pad.bool() != manual_pad.bool()).flatten(1).sum(1)
    original_diff = (native != manual).flatten(1).sum(1)
    audit = {
        "padded_pixel_differences": pad_diff.detach().cpu().tolist(),
        "original_pixel_differences": original_diff.detach().cpu().tolist(),
        "passed": bool((pad_diff == 0).all() and (original_diff == 0).all()),
    }
    if not audit["passed"]:
        raise AssertionError(f"Zero-bias replay is not native-equivalent: {audit}")
    return native, logits, audit


@torch.no_grad()
def evaluate_image(x, coeffs, coco, scalar_bias=None, chunk_size=4):
    """Evaluate all supplied arms on one image without modifying candidate IDs.

    ``raw_id`` is metadata only; all tensors here already use compact order.
    ``input_shape`` may be supplied in x; it defaults to the frozen 640x640.
    scalar_bias is one globally frozen scalar, never an instance-specific value.
    """
    if "A" not in coeffs:
        raise ValueError("coeffs must contain the frozen original A arm")
    if "S" in coeffs:
        raise ValueError("Pass the S calibration with scalar_bias, not coefficient tensors")
    proto = x["proto"].detach().float()
    if proto.ndim != 3 or proto.shape[0] != 32:
        raise ValueError(f"Expected complete [32,H,W] prototype, got {tuple(proto.shape)}")
    device = proto.device
    count = len(x["rows"])
    boxes = x["boxes"].detach().to(device=device, dtype=torch.float32)
    if tuple(boxes.shape) != (count, 4):
        raise ValueError("x boxes must be compact [N,4], aligned with rows")
    arms = {}
    for arm, value in coeffs.items():
        if tuple(value.shape) != (count, 32):
            raise ValueError(f"Arm {arm}: expected compact [{count},32], got {tuple(value.shape)}")
        arms[arm] = value.detach().to(device=device, dtype=torch.float32)
        if not torch.isfinite(arms[arm]).all():
            raise ValueError(f"Nonfinite coefficients in arm {arm}")
    c0 = x["c0"].detach().to(device=device, dtype=torch.float32)
    torch.testing.assert_close(arms["A"], c0, rtol=0, atol=0)
    if scalar_bias is not None:
        scalar_bias = float(scalar_bias)
        if not np.isfinite(scalar_bias):
            raise ValueError("scalar_bias must be a finite globally frozen scalar")
    input_shape = tuple(map(int, x.get("input_shape", (640, 640))))
    original_shape = tuple(map(int, x["original_shape"]))
    ratio_pad = x["ratio_pad"]
    results = []
    levels = x["levels"].detach().cpu().tolist() if torch.is_tensor(x["levels"]) else x["levels"]
    for start in range(0, count, max(1, int(chunk_size))):
        end = min(count, start + max(1, int(chunk_size)))
        bb = boxes[start:end]
        base_masks, base_logits, replay = zero_bias_replay(
            proto, arms["A"][start:end], bb, input_shape, original_shape, ratio_pad
        )
        supports = ops.crop_mask(torch.ones((end-start, *input_shape), device=device), bb).bool()
        originals, padded_truth, chunk_rows = [], [], []
        for offset, idx in enumerate(range(start, end)):
            source = x["rows"][idx]
            aid = int(source["annotation_id"])
            annotation = coco.anns[aid]
            iid = int(source.get("image_id", x.get("image_id", annotation["image_id"])))
            if int(annotation["image_id"]) != iid:
                raise ValueError("COCO annotation/image identity mismatch")
            gt = torch.as_tensor(coco.annToMask(annotation).astype(bool), device=device)
            if tuple(gt.shape) != original_shape:
                raise ValueError(f"GT original shape {gt.shape} != frozen shape {original_shape}")
            pad_gt = _padded_gt(gt, ratio_pad, input_shape)
            originals.append(gt)
            padded_truth.append(pad_gt)
            area = float(annotation.get("area", source.get("area", int(gt.sum()))))
            lev = int(levels[idx])
            if lev not in (0, 1, 2):
                raise ValueError("pyramid level must be compact 0/1/2 for P3/P4/P5")
            box_xyxy = boxes[idx].detach().cpu().tolist()
            box_w = (box_xyxy[2] - box_xyxy[0]) / input_shape[1]
            box_h = (box_xyxy[3] - box_xyxy[1]) / input_shape[0]
            rr = {
                "split": str(x["split"]), "image_id": iid, "annotation_id": aid,
                "branch": str(source.get("branch", "one2one")),
                "raw_id": int(source["raw_id"]), "pyramid_level": lev,
                "target_gt_idx": int(source.get("target_gt_idx", source.get("gt_index", -1))),
                "class_id": int(annotation["category_id"]),
                "class_id_system": "COCO_category_id; evaluation metadata only",
                "box_xyxy": box_xyxy,
                "box_coordinate_system": "frozen letterbox input xyxy; no additional clipping",
                "box_w_normalized": box_w,
                "box_h_normalized": box_h,
                "box_area_normalized": box_w * box_h,
                "area": area,
                "size_group": "small" if area < 32**2 else "medium" if area < 96**2 else "large",
                "pixel_support_count": int(supports[offset].sum()),
                "pixel_support_gt_positive": int((supports[offset] & pad_gt).sum()),
                "zero_bias_padded_pixel_differences": int(replay["padded_pixel_differences"][offset]),
                "zero_bias_original_pixel_differences": int(replay["original_pixel_differences"][offset]),
            }
            if "box_iou" in source:
                rr["box_iou"] = float(source["box_iou"])
            if "predicted_class_id" in source:
                rr["predicted_class_id"] = int(source["predicted_class_id"])
                rr["predicted_class_id_system"] = "model_zero_based_class_index"
            chunk_rows.append(rr)

        def collect(arm, masks, logits):
            for k, rr in enumerate(chunk_rows):
                gt, mask = originals[k], masks[k]
                intersection = int((mask & gt).sum())
                union = int((mask | gt).sum())
                rr[f"iou_{arm}"] = intersection / max(union, 1)
                rr[f"coverage_{arm}"] = intersection / max(int(gt.sum()), 1)
                rr[f"mask75_{arm}"] = int(rr[f"iou_{arm}"] >= 0.75)
                rr[f"auc_{arm}"], rr[f"fpr_{arm}"] = pixel_auc_fpr(logits[k], padded_truth[k], supports[k])

        collect("A", base_masks, base_logits)
        for arm, coefficients in arms.items():
            if arm == "A":
                continue
            cs = coefficients[start:end]
            padded = ops.process_mask(proto, cs, bb, input_shape, upsample=True)
            masks = _scale_binary(padded, original_shape, ratio_pad)
            collect(arm, masks, _logits(proto, cs, input_shape))
            del masks, padded
        if scalar_bias is not None:
            biased = base_logits + scalar_bias
            padded = ((biased > 0) & supports).byte()
            masks = _scale_binary(padded, original_shape, ratio_pad)
            collect("S", masks, biased)
            for k, rr in enumerate(chunk_rows):
                rr["scalar_bias_S"] = scalar_bias
                rr["scalar_bias_sign"] = "z_S=z_A+b"
                rr["auc_S_minus_A_numeric"] = rr["auc_S"] - rr["auc_A"]
                rr["auc_S_minus_A_abs_numeric"] = abs(rr["auc_S_minus_A_numeric"])
                rr["scalar_bias_auc_note"] = "AUC is invariant in exact arithmetic; FP32 addition can merge nearby ranks into ties; observed difference retained"
            del masks, padded, biased
        for k, rr in enumerate(chunk_rows):
            source = x["rows"][start+k]
            if "initial_iou" in source:
                rr["cached_baseline_iou_absolute_error"] = abs(rr["iou_A"] - float(source["initial_iou"]))
            rr["original_status"] = "success" if rr["mask75_A"] else "failure"
        results.extend(chunk_rows)
    return results


def _finite_mean(values):
    a = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(a)
    return float(a[valid].mean()) if valid.any() else float("nan")


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def _paired_stats(by_image, metric, arm, reference, bootstrap, rng):
    """Pair candidate measurements, then resample entire images for both estimands."""
    sums, counts = [], []
    for image_rows in by_image:
        left = np.array([r[f"{metric}_{arm}"] for r in image_rows], dtype=np.float64)
        right = np.array([r[f"{metric}_{reference}"] for r in image_rows], dtype=np.float64)
        valid = np.isfinite(left) & np.isfinite(right)
        sums.append(float((left[valid] - right[valid]).sum()))
        counts.append(int(valid.sum()))
    sums, counts = np.array(sums), np.array(counts)
    good = counts > 0
    per = np.divide(sums, counts, out=np.zeros_like(sums), where=good)
    macro = float(per[good].mean()) if good.any() else float("nan")
    candidate = float(sums.sum()/counts.sum()) if counts.sum() else float("nan")
    macro_boot, candidate_boot = [], []
    for begin in range(0, bootstrap, 256):
        draw = rng.integers(0, len(by_image), size=(min(256, bootstrap-begin), len(by_image)))
        macro_counts = good[draw].sum(1)
        candidate_counts = counts[draw].sum(1)
        macro_boot.extend(np.divide(per[draw].sum(1), macro_counts, out=np.full(len(draw), np.nan), where=macro_counts>0))
        candidate_boot.extend(np.divide(sums[draw].sum(1), candidate_counts, out=np.full(len(draw), np.nan), where=candidate_counts>0))
    def interval(samples):
        values = np.asarray(samples)
        finite = values[np.isfinite(values)]
        return np.quantile(finite, [0.025, 0.975]).tolist() if finite.size else [float("nan")]*2
    return {
        "image_macro": {"delta": macro, "ci95": interval(macro_boot), "valid_images": int(good.sum())},
        "candidate": {"delta": candidate, "ci95": interval(candidate_boot), "valid_candidates": int(counts.sum())},
        "bootstrap_empty_draws": int(np.isnan(macro_boot).sum()),
    }


def _table(rr, arms, seed, bootstrap, group_name):
    by_id = defaultdict(list)
    for row in rr:
        by_id[int(row["image_id"])].append(row)
    ids = sorted(by_id)
    by_image = [by_id[i] for i in ids]
    table = {"n_images":len(ids), "n_candidates":len(rr), "candidate":{}, "image_macro":{}, "comparisons":{}, "mask75_counts":{}, "undefined":{}}
    if not rr:
        return table
    for metric in METRICS:
        table["candidate"][metric] = {arm:_finite_mean([r[f"{metric}_{arm}"] for r in rr]) for arm in arms}
        table["image_macro"][metric] = {arm:_finite_mean([_finite_mean([r[f"{metric}_{arm}"] for r in one]) for one in by_image]) for arm in arms}
        table["undefined"][metric] = {arm:sum(not np.isfinite(r[f"{metric}_{arm}"]) for r in rr) for arm in arms}
    # Damage is always defined relative to ORIGINAL A success, including when
    # comparing D to C/M/etc. Never redefine the at-risk population per arm.
    success_by_image = [
        [{f"damage_{arm}":int(not r[f"mask75_{arm}"]) for arm in arms}
         for r in one if bool(r["mask75_A"])]
        for one in by_image
    ]
    n_original_success = sum(len(one) for one in success_by_image)
    table["original_success_damage"] = {
        "population": "A original-image MaskIoU >= 0.75; damage_X=1[MaskIoU_X<0.75]",
        "n_candidates": n_original_success,
        "n_images_with_original_success": sum(bool(one) for one in success_by_image),
        "damage_counts": {arm:sum(r[f"damage_{arm}"] for one in success_by_image for r in one) for arm in arms},
        "candidate_rates": {arm:_finite_mean([r[f"damage_{arm}"] for one in success_by_image for r in one]) for arm in arms},
        "image_macro_rates": {arm:_finite_mean([_finite_mean([r[f"damage_{arm}"] for r in one]) for one in success_by_image]) for arm in arms},
    }
    pairs = list(PRIMARY_COMPARISONS)
    pairs += [(arm,"A") for arm in arms if arm != "A" and (arm,"A") not in pairs]
    for arm, reference in pairs:
        if arm not in arms or reference not in arms:
            continue
        name = f"{arm}_minus_{reference}"
        # Every metric uses the same reproducible image draw sequence.
        result = {metric:_paired_stats(by_image,metric,arm,reference,bootstrap,np.random.default_rng(seed)) for metric in METRICS}
        repair = sum(not r[f"mask75_{reference}"] and r[f"mask75_{arm}"] for r in rr)
        damage = sum(r[f"mask75_{reference}"] and not r[f"mask75_{arm}"] for r in rr)
        result["mask75_transition"] = {"repair":repair,"damage":damage,"net":repair-damage,"net_candidate_fraction":(repair-damage)/len(rr)}
        damage_stats = _paired_stats(success_by_image,"damage",arm,reference,bootstrap,np.random.default_rng(seed))
        damage_stats["population"] = "Original A success; identical eligible candidates for both arms"
        damage_stats["direction"] = "Positive delta means more original-success damage in first arm"
        damage_stats["interpretation"] = "A CI spanning zero does not establish equivalence or noninferiority"
        for estimand in ("candidate","image_macro"):
            low, high = damage_stats[estimand]["ci95"]
            damage_stats[estimand]["increase_detected_by_95ci"] = bool(np.isfinite(low) and low > 0)
            damage_stats[estimand]["upper_bound_at_or_below_zero"] = bool(np.isfinite(high) and high <= 0)
        result["damage_rate_on_A_success"] = damage_stats
        table["comparisons"][name] = result
    table["mask75_counts"] = {arm:sum(r[f"mask75_{arm}"] for r in rr) for arm in arms}
    table["role"] = "primary_all_candidates" if group_name == "all" else "prespecified_descriptive_stratum"
    return table


def summarize(rows, out, seed=0, bootstrap=5000):
    """Save fixed-arm reports; CIs are paired image-cluster bootstrap intervals."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    arms = [arm for arm in ARMS if all(f"iou_{arm}" in r for r in rows)]
    present = {key[4:] for row in rows for key in row if key.startswith("iou_")}
    if present != set(arms) or "A" not in arms:
        raise ValueError("Every candidate must have the same supported arms; never silently drop arm measurements")
    keys = [(r["split"],r["image_id"],r["annotation_id"],r["branch"],r["raw_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate permanent candidate identities in evaluation rows")
    tables, image_rows = {}, []
    splits = [s for s in ("fit","dev","val") if any(r["split"] == s for r in rows)]
    splits += sorted({r["split"] for r in rows} - set(splits))
    for split in splits:
        selected = [r for r in rows if r["split"] == split]
        groups = {"all":selected}
        if split == "val":
            groups.update({f"P{level+3}":[r for r in selected if r["pyramid_level"] == level] for level in range(3)})
            groups.update({f"original_{name}":[r for r in selected if bool(r["mask75_A"]) == flag] for name,flag in (("success",True),("failure",False))})
            groups.update({f"size_{size}":[r for r in selected if r["size_group"] == size] for size in ("small","medium","large")})
        for name, rr in groups.items():
            tables[f"{split}:{name}"] = _table(rr,arms,seed,bootstrap,name)
        by_image = defaultdict(list)
        for r in selected:
            by_image[int(r["image_id"])].append(r)
        for iid, one in sorted(by_image.items()):
            rec = {"split":split,"image_id":iid,"n_candidates":len(one)}
            for metric in METRICS:
                for arm in arms:
                    rec[f"{metric}_{arm}"] = _finite_mean([r[f"{metric}_{arm}"] for r in one])
                    rec[f"{metric}_{arm}_valid_candidates"] = int(sum(np.isfinite(r[f"{metric}_{arm}"]) for r in one))
            for arm,reference in PRIMARY_COMPARISONS:
                if arm in arms and reference in arms:
                    for metric in METRICS:
                        rec[f"{metric}_{arm}_minus_{reference}"] = _finite_mean([r[f"{metric}_{arm}"]-r[f"{metric}_{reference}"] for r in one])
            original_success = [r for r in one if bool(r["mask75_A"])]
            rec["n_original_success"] = len(original_success)
            for arm in arms:
                rec[f"original_success_damage_count_{arm}"] = sum(not r[f"mask75_{arm}"] for r in original_success)
                rec[f"original_success_damage_rate_{arm}"] = _finite_mean([int(not r[f"mask75_{arm}"]) for r in original_success])
            image_rows.append(rec)
    scalar_auc_differences = np.array([r.get("auc_S_minus_A_numeric",float("nan")) for r in rows],dtype=np.float64)
    valid_scalar_auc = scalar_auc_differences[np.isfinite(scalar_auc_differences)]
    result = {
        "arms":arms,"seed":seed,"bootstrap":bootstrap,"bootstrap_unit":"image; paired arms, entire candidate clusters",
        "main_metric":"val:all image_macro original-image Mask IoU",
        "primary_comparisons":[f"{a}_minus_{b}" for a,b in PRIMARY_COMPARISONS if a in arms and b in arms],
        "intervals":"95% pointwise percentile image-cluster bootstrap; not multiplicity-adjusted",
        "units":"All metric values and deltas are fractions; multiply by 100 for percentage points",
        "statistical_scope":"Frozen official TAL candidates on reused research images; not blind testing and not COCO AP",
        "auc_support":"Continuous 640-grid logits within fixed predicted box; raw COCO GT nearest resize/pad; identical support per arm",
        "undefined_auc_policy":"Retain candidate in IoU, coverage and Mask75; NaN AUC is explicitly counted",
        "decode":"process_mask(upsample=True) -> binary input mask -> scale_masks(real ratio_pad) -> >0.5",
        "zero_bias_replay_audit":{
            "padded_pixel_differences":sum(r.get("zero_bias_padded_pixel_differences",0) for r in rows),
            "original_pixel_differences":sum(r.get("zero_bias_original_pixel_differences",0) for r in rows),
            "audited_candidates":sum("zero_bias_original_pixel_differences" in r for r in rows),
        },
        "scalar_bias_auc_audit": {
            "theory": "A shared additive scalar leaves pixel ordering and ROC AUC unchanged in exact arithmetic",
            "floating_point_note": "Actual FP32 logit addition may create ties. Measured S-A AUC differences are retained, never overwritten with zero",
            "defined_candidates": int(valid_scalar_auc.size),
            "undefined_or_not_evaluated_candidates": int(len(rows)-valid_scalar_auc.size),
            "nonzero_difference_candidates": int(np.count_nonzero(valid_scalar_auc)),
            "max_absolute_difference": float(np.abs(valid_scalar_auc).max()) if valid_scalar_auc.size else float("nan"),
            "mean_absolute_difference": float(np.abs(valid_scalar_auc).mean()) if valid_scalar_auc.size else float("nan"),
            "mean_signed_difference": float(valid_scalar_auc.mean()) if valid_scalar_auc.size else float("nan"),
        },
        "tables":tables,
    }
    _json(out/"RESULTS.json",result)
    _json(out/"PER_IMAGE.json",image_rows)
    with (out/"PER_CANDIDATE.jsonl").open("w",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=True)+"\n")
    fields = ["split_group","arm","n_images","n_candidates","image_macro_iou","candidate_iou","image_macro_iou_delta_vs_A","image_macro_iou_delta_ci95_low","image_macro_iou_delta_ci95_high","candidate_iou_delta_vs_A","candidate_iou_delta_ci95_low","candidate_iou_delta_ci95_high","mask75_count","repair_vs_A","damage_vs_A","net_mask75_count_vs_A","coverage_image_macro","auc_image_macro","fpr_image_macro"]
    with (out/"ABLATION_TABLE.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for group,table in tables.items():
            if not table["n_candidates"]:
                continue
            for arm in arms:
                row={"split_group":group,"arm":arm,"n_images":table["n_images"],"n_candidates":table["n_candidates"],"image_macro_iou":table["image_macro"]["iou"][arm],"candidate_iou":table["candidate"]["iou"][arm],"mask75_count":table["mask75_counts"][arm]}
                for metric in ("coverage","auc","fpr"):
                    row[f"{metric}_image_macro"]=table["image_macro"][metric][arm]
                if arm!="A":
                    comparison=table["comparisons"][f"{arm}_minus_A"]
                    for estimand in ("image_macro","candidate"):
                        stat=comparison["iou"][estimand]
                        row[f"{estimand}_iou_delta_vs_A"]=stat["delta"]
                        row[f"{estimand}_iou_delta_ci95_low"],row[f"{estimand}_iou_delta_ci95_high"]=stat["ci95"]
                    trans=comparison["mask75_transition"]
                    row.update(repair_vs_A=trans["repair"],damage_vs_A=trans["damage"],net_mask75_count_vs_A=trans["net"])
                writer.writerow(row)
    return result
