"""Independently verify transferred ACD prototype-tail readouts without inference.

Usage: python verify_readout.py --run <RUN_ACD_PROTO_PAIRED_EVAL_S0>
       python verify_readout.py --run <run> --out <new-audit.json>

The only non-stdlib dependency is NumPy. Inputs are opened read-only; no SSH,
network, model inference, package installation, or evaluator import is used.
Default output is stdout. --out creates a NEW audit file exclusively and refuses
to overwrite any existing file. Missing/incomplete/invalid inputs exit nonzero.

This recomputes AP/AP75/APsmall from standard COCO accumulated precision and
checks paired point estimates against image/instance records. It does not rerun
COCO matching, reconstruct masks from RLE, revalidate GT annotations, or recompute
bootstrap confidence intervals; those remain explicit limits of this audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

VERSION = "acd_proto_tail_readout_verify_v1"
EVALUATOR_VERSION = "acd_proto_tail_eval_v1"
OVERLAY_PREFIXES = ("model.23.cv4.", "model.23.one2one_cv4.",
                    "model.23.proto.cv2.", "model.23.proto.cv3.")
OVERLAY_KIND = "mask_coefficient_proto_tail_ema_final"
EXPECTED_IMAGES = 5000
EXPECTED_ARMS = 3
ABS_TOLERANCE = 1e-12


class Audit:
    def __init__(self, run):
        self.run = run
        self.checks = 0
        self.errors = []
        self.missing = []
        self.evidence = {}
        self.results = {}

    def check(self, condition, description, detail=None):
        self.checks += 1
        if not condition:
            error = {"check": description}
            if detail is not None:
                error["detail"] = detail
            self.errors.append(error)
        return bool(condition)

    def require(self, path):
        if not path.is_file():
            self.missing.append(str(path))
            return False
        return True

    def fingerprint(self, path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        self.evidence[str(path.relative_to(self.run))] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}

    def read_json(self, path):
        if not self.require(path):
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            self.fingerprint(path)
            return value
        except (OSError, ValueError) as error:
            self.check(False, f"read JSON {path.relative_to(self.run)}", str(error))
            return None

    def value_equal(self, actual, recorded, description):
        if actual is None or recorded is None:
            return self.check(actual is None and recorded is None, description,
                              {"recomputed": actual, "recorded": recorded})
        finite_number = isinstance(recorded, (int, float)) and not isinstance(recorded, bool) and math.isfinite(recorded)
        return self.check(finite_number and math.isclose(float(actual), float(recorded), rel_tol=0., abs_tol=ABS_TOLERANCE),
                          description, {"recomputed": actual, "recorded": recorded})

    def report(self):
        return {"audit_version": VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "run": str(self.run), "status": "missing" if self.missing else "failed" if self.errors else "passed",
                "passed": not self.missing and not self.errors, "checks": self.checks,
                "missing": self.missing, "errors": self.errors, "evidence": self.evidence,
                "results": self.results, "absolute_tolerance": ABS_TOLERANCE,
                "limitations": ["Aggregate readout verification only; COCO matches, mask decoding and GT are not recomputed",
                                "Image-cluster bootstrap point estimates and stored sample/seed metadata are checked; confidence intervals are not recomputed",
                                "Stored frozen-state digests and parity records are cross-checked; model tensors are not loaded"]}


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def ap_from_accumulated(arrays, np):
    """Standard COCO precision has axes [T,R,K,A,M], excluding values -1."""
    required = {"precision", "recall", "scores", "iou_thresholds", "recall_thresholds", "category_ids", "max_detections", "area_ranges"}
    if not required.issubset(arrays):
        raise ValueError(f"Missing accumulated fields: {sorted(required-set(arrays))}")
    precision = arrays["precision"]
    thresholds = arrays["iou_thresholds"]
    recalls = arrays["recall_thresholds"]
    categories = arrays["category_ids"]
    areas = arrays["area_ranges"]
    maxima = arrays["max_detections"]
    expected_shape = (len(thresholds), len(recalls), len(categories), len(areas), len(maxima))
    if precision.ndim != 5 or precision.shape != expected_shape:
        raise ValueError(f"Invalid precision axes/shape: {precision.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(precision)) or not np.all((precision == -1) | ((precision >= 0) & (precision <= 1))):
        raise ValueError("Precision contains nonfinite or out-of-range values")
    if len(thresholds) != 10 or not np.allclose(thresholds, np.linspace(.5, .95, 10), rtol=0, atol=1e-14):
        raise ValueError("Not standard COCO IoU thresholds .50:.05:.95")
    if len(recalls) != 101 or not np.allclose(recalls, np.linspace(0, 1, 101), rtol=0, atol=1e-14):
        raise ValueError("Not standard 101-point COCO recall grid")
    if not np.array_equal(maxima, [1, 10, 100]):
        raise ValueError("Not standard COCO maxDets [1,10,100]")
    expected_areas = np.array([[0, 1e10], [0, 32**2], [32**2, 96**2], [96**2, 1e10]], dtype=float)
    if areas.shape != expected_areas.shape or not np.array_equal(areas, expected_areas):
        raise ValueError("Not standard COCO all/small/medium/large area ranges")
    if arrays["scores"].shape != precision.shape or arrays["recall"].shape != (len(thresholds), len(categories), len(areas), len(maxima)):
        raise ValueError("Inconsistent accumulated scores/recall axes")
    m = int(np.flatnonzero(maxima == 100)[0])
    t75 = int(np.flatnonzero(np.isclose(thresholds, .75, rtol=0, atol=1e-14))[0])

    def mean_valid(values):
        valid = values[values > -1]
        return float(np.mean(valid)) if valid.size else None

    return {"AP": mean_valid(precision[:, :, :, 0, m]),
            "AP75": mean_valid(precision[t75:t75+1, :, :, 0, m]),
            "APsmall": mean_valid(precision[:, :, :, 1, m])}


def verify_coco(audit, summary, names, ids, np):
    all_arrays = {}
    recomputed = {}
    for name in names:
        metrics = audit.read_json(audit.run / name / "COCO_METRICS.json")
        if metrics is None:
            continue
        audit.check(metrics.get("image_ids") == sorted(ids), f"{name}: COCO metric image identities")
        summary_metrics = summary["metrics"].get(name, {})
        recomputed[name] = {}
        for iou_type in ("segm", "bbox"):
            path = audit.run / name / f"COCO_{iou_type}_ACCUMULATED.npz"
            if not audit.require(path):
                continue
            try:
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                audit.fingerprint(path)
                value = ap_from_accumulated(arrays, np)
                recomputed[name][iou_type] = value
                audit.check(len(arrays["category_ids"]) == 80, f"{name}/{iou_type}: official 80 category identities")
                for key, result in value.items():
                    audit.check(result is not None, f"{name}/{iou_type}/{key}: defined on full COCO val")
                    audit.value_equal(result, metrics.get(iou_type, {}).get(key), f"{name}/{iou_type}/{key}: NPZ vs COCO_METRICS")
                    audit.value_equal(result, summary_metrics.get(iou_type, {}).get(key), f"{name}/{iou_type}/{key}: NPZ vs SUMMARY")
                if iou_type == "bbox":
                    all_arrays[name] = arrays
            except (OSError, ValueError, IndexError, KeyError) as error:
                audit.check(False, f"{name}/{iou_type}: accumulated COCO tensors", str(error))
    if len(all_arrays) == len(names):
        reference = names[0]
        for name in names[1:]:
            audit.check(set(all_arrays[name]) == set(all_arrays[reference]), f"bbox {name} vs {reference}: same accumulated fields")
            for key in sorted(set(all_arrays[name]) & set(all_arrays[reference])):
                audit.check(np.array_equal(all_arrays[name][key], all_arrays[reference][key]),
                            f"bbox {name} vs {reference}: {key} exactly equal")
    baseline = summary.get("baseline")
    if baseline in recomputed:
        for name in names:
            if name == baseline or name not in recomputed:
                continue
            for kind in ("segm", "bbox"):
                for key in ("AP", "AP75", "APsmall"):
                    b = recomputed[baseline].get(kind, {}).get(key)
                    a = recomputed[name].get(kind, {}).get(key)
                    delta = 100*(a-b) if a is not None and b is not None else None
                    stored = summary.get("delta_vs_baseline", {}).get(name, {}).get(kind, {}).get(key+"_points")
                    audit.value_equal(delta, stored, f"{name}/{kind}/{key}: AP-point delta vs baseline")
    audit.results["coco_recomputed"] = recomputed


def read_jsonl(audit, path):
    if not audit.require(path):
        return
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    audit.check(False, f"{path.name}:{line_number}: blank JSONL record")
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("Expected JSON object")
                    yield line_number, value
                except ValueError as error:
                    audit.check(False, f"{path.name}:{line_number}: JSONL record", str(error))
        audit.fingerprint(path)
    except OSError as error:
        audit.check(False, f"read {path.relative_to(audit.run)}", str(error))


def verify_pair(audit, name, baseline, summary, ids, np):
    folder = audit.run / f"paired_{name}_vs_{baseline}"
    pair = audit.read_json(folder / "SUMMARY.json")
    if pair is None:
        return
    audit.check(summary.get("paired", {}).get(name) == pair, f"{name}: pair SUMMARY matches top-level SUMMARY")
    audit.check(pair.get("paired_readout_valid") is True, f"{name}: paired readout valid")
    parity = pair.get("parity", {})
    audit.check(parity.get("readout_view") == "output detections; raw candidate geometry/five-state diagnosis not computed here",
                f"{name}: output-level diagnostic scope is explicit")
    for key in ("frozen_state_exact_equal", "detection_index_equal", "all_images_box_class_exact_equal"):
        audit.check(parity.get(key) is True, f"{name}: parity {key}")
    for key in ("detection_count_mismatch_images", "class_mismatch_detections", "box_max_abs_error", "confidence_max_abs_error"):
        audit.check(parity.get(key) == 0, f"{name}: parity {key} zero")
    audit.check(parity.get("image_count") == EXPECTED_IMAGES, f"{name}: parity covers 5000 images")
    rows = {}
    for line, image in read_jsonl(audit, folder / "IMAGES.jsonl"):
        iid = image.get("image_id")
        label = f"{name}/image line {line}"
        valid_id = integer(iid) and iid in ids and iid not in rows
        audit.check(valid_id, label+": unique expected image identity")
        audit.check(image.get("paired_valid") is True, label+": paired_valid")
        keys = ("matched_detections", "success", "damage", "failure", "repair")
        valid_counts = all(integer(image.get(key)) and image[key] >= 0 for key in keys)
        audit.check(valid_counts, label+": integer nonnegative counts")
        valid_delta = finite_number(image.get("mask_iou_delta_sum"))
        audit.check(valid_delta, label+": finite mask IoU delta sum")
        if valid_id and valid_counts and valid_delta:
            audit.check(image["matched_detections"] == image["success"] + image["failure"], label+": matches=success+failure")
            audit.check(image["damage"] <= image["success"] and image["repair"] <= image["failure"], label+": damage/repair denominators")
            rows[iid] = image
    audit.check(len(rows) == EXPECTED_IMAGES and set(rows) == set(ids), f"{name}: exactly 5000 complete paired image records")
    id_position = {iid: index for index, iid in enumerate(ids)}
    # Same nine sufficient statistics as the written evaluator schema, but
    # independently reconstructed from the explicit instance rows.
    values = np.zeros((len(ids), 9), dtype=np.float64)
    seen = set()
    unique_gt = set()
    instance_lines = 0
    for line, instance in read_jsonl(audit, folder / "INSTANCES.jsonl"):
        instance_lines += 1
        label = f"{name}/instance line {line}"
        iid, index, gid = instance.get("image_id"), instance.get("detection_index"), instance.get("annotation_id")
        identity = (iid, index)
        valid_id = integer(iid) and iid in id_position and integer(index) and index >= 0 and integer(gid) and gid > 0
        audit.check(valid_id, label+": valid image/detection/GT identities")
        audit.check(identity not in seen, label+": detection identity is unique")
        seen.add(identity)
        numeric = ("baseline_mask_iou", "method_mask_iou", "delta_mask_iou", "baseline_box_iou", "baseline_score", "gt_area")
        valid_numbers = all(finite_number(instance.get(key)) for key in numeric)
        audit.check(valid_numbers, label+": finite IoU/score/area values")
        valid_flags = all(isinstance(instance.get(key), bool) for key in ("baseline_success", "damage", "repair"))
        audit.check(valid_flags, label+": boolean success/damage/repair flags")
        if not (valid_id and valid_numbers and valid_flags):
            continue
        a, b, delta = (instance[key] for key in ("baseline_mask_iou", "method_mask_iou", "delta_mask_iou"))
        audit.check(0 <= a <= 1 and 0 <= b <= 1, label+": mask IoU range")
        audit.check(.5 <= instance["baseline_box_iou"] <= 1, label+": box match IoU>=.5")
        audit.check(.001 < instance["baseline_score"] <= 1 and instance["gt_area"] >= 0, label+": score/area range")
        audit.check(integer(instance.get("category_id")) and instance["category_id"] > 0, label+": category identity")
        audit.value_equal(b-a, delta, label+": delta=method-baseline")
        success = a >= .75
        damage = success and b < .75
        repair = not success and b >= .75
        audit.check(instance["baseline_success"] == success and instance["damage"] == damage and instance["repair"] == repair,
                    label+": exact .75 threshold classification")
        values[id_position[iid]] += [1, delta, int(success), int(damage), int(not success), int(repair), a, b, delta if success else 0]
        unique_gt.add((iid, gid))
    for iid, image in rows.items():
        reconstructed = values[id_position[iid]]
        for key, col in (("matched_detections", 0), ("success", 2), ("damage", 3), ("failure", 4), ("repair", 5)):
            audit.check(image[key] == int(reconstructed[col]), f"{name}/image {iid}: {key} vs instance rows")
        audit.value_equal(float(reconstructed[1]), image["mask_iou_delta_sum"], f"{name}/image {iid}: delta sum vs instance rows")
    audit.check(instance_lines == int(values[:, 0].sum()), f"{name}: all instance lines are accounted for")
    audit.check(pair.get("unique_matched_gt_instances") == len(unique_gt), f"{name}: unique matched GT count")
    statistics = pair.get("statistics", {})
    sums = values.sum(0)
    counts = {"matched_detection_count": int(sums[0]), "baseline_success_count": int(sums[2]),
              "damage_count": int(sums[3]), "baseline_failure_count": int(sums[4]), "repair_count": int(sums[5]),
              "images": len(rows), "images_with_matched_detections": int((values[:, 0] > 0).sum())}
    for key, value in counts.items():
        audit.check(statistics.get(key) == value, f"{name}: SUMMARY statistics {key}", {"recomputed": value, "recorded": statistics.get(key)})
    definitions = {"mean_mask_iou_delta": (1, 0), "damage_rate_of_baseline_success": (3, 2),
                   "repair_rate_of_baseline_failure": (5, 4), "baseline_mean_mask_iou": (6, 0),
                   "method_mean_mask_iou": (7, 0), "baseline_success_mean_iou_delta": (8, 2)}
    estimates = {}
    for key, (numerator, denominator) in definitions.items():
        point = float(sums[numerator]/sums[denominator]) if sums[denominator] else None
        estimates[key] = point
        audit.value_equal(point, statistics.get(key, {}).get("value"), f"{name}: point estimate {key}")
    populated = values[:, 0] > 0
    macro = float(np.mean(values[populated, 1]/values[populated, 0])) if populated.any() else None
    audit.value_equal(macro, statistics.get("image_macro_mask_iou_delta"), f"{name}: image macro delta")
    # These four point estimates are also derivable from IMAGES alone. This
    # independent total verifies that the image aggregation matches SUMMARY.
    image_sums = {key: sum(image[key] for image in rows.values()) for key in ("matched_detections", "success", "damage", "failure", "repair")}
    image_delta = math.fsum(image["mask_iou_delta_sum"] for image in rows.values())
    for key, numerator, denominator in (("mean_mask_iou_delta", image_delta, image_sums["matched_detections"]),
                                        ("damage_rate_of_baseline_success", image_sums["damage"], image_sums["success"]),
                                        ("repair_rate_of_baseline_failure", image_sums["repair"], image_sums["failure"])):
        point = numerator/denominator if denominator else None
        audit.value_equal(point, statistics.get(key, {}).get("value"), f"{name}: IMAGES-only point {key}")
    audit.check(statistics.get("bootstrap_samples") == 5000 and statistics.get("bootstrap_seed") == 0,
                f"{name}: predeclared 5000 bootstrap samples, seed 0")
    audit.results.setdefault("paired_recomputed", {})[name] = counts | {
        "instance_lines": instance_lines, "unique_matched_gt_instances": len(unique_gt),
        "point_estimates": estimates, "image_macro_mask_iou_delta": macro}


def verify_overlay_metadata(audit, name, receipt, configuration):
    """Cross-check declared reconstruction/freeze scope; never load model tensors."""
    model = receipt.get("model", {})
    if not audit.check(isinstance(model, dict), f"{name}: model metadata object"):
        return None
    audit.check(model.get("overlay_scope_prefixes") == list(OVERLAY_PREFIXES),
                f"{name}: exact coefficient/prototype-tail scope")
    audit.check(model.get("reconstructed_coefficient_only") is False,
                f"{name}: does not claim coefficient-only reconstruction")
    mutable = model.get("overlay_non_bn_parameter_keys")
    valid_mutable = (isinstance(mutable, list) and bool(mutable)
                     and all(isinstance(key, str) and key.startswith(OVERLAY_PREFIXES) for key in mutable)
                     and len(set(mutable)) == len(mutable) and mutable == sorted(mutable))
    audit.check(valid_mutable, f"{name}: sorted unique non-BN overlay parameter identities")
    if valid_mutable:
        audit.check(all(any(key.startswith(prefix) for key in mutable) for prefix in OVERLAY_PREFIXES),
                    f"{name}: all four scope components have mutable parameters")
        audit.check(receipt.get("overlay_non_bn_parameter_keys") == mutable,
                    f"{name}: receipt and model agree on non-BN mutable scope")
        coefficients = [key for key in mutable if key.startswith(OVERLAY_PREFIXES[:2])]
        audit.check(receipt.get("coefficient_state_keys") == coefficients,
                    f"{name}: legacy coefficient field covers only actual coefficient parameters")
    bn = model.get("overlay_bn_state_keys")
    valid_bn = (isinstance(bn, list) and bool(bn)
                and all(isinstance(key, str) and key.startswith(OVERLAY_PREFIXES) for key in bn)
                and bn == sorted(set(bn)))
    audit.check(valid_bn, f"{name}: frozen overlay BN identities present")
    if valid_bn and valid_mutable:
        audit.check(set(bn).isdisjoint(mutable), f"{name}: BN is excluded from mutable parameters")
    audit.check(model.get("frozen_digest_excludes") ==
                "only non-BN parameters in the declared overlay scope; all BN and all other state retained",
                f"{name}: all BN and outside-scope state remain in frozen digest")
    trained = name in ("baseline", "acd")
    audit.check(model.get("reconstructed_mask_tail_overlay") is trained,
                f"{name}: expected official vs trained reconstruction")
    keys = model.get("overlay_state_keys")
    if trained:
        valid_keys = (isinstance(keys, list) and bool(keys)
                      and all(isinstance(key, str) and key.startswith(OVERLAY_PREFIXES) for key in keys)
                      and keys == sorted(set(keys)))
        audit.check(valid_keys, f"{name}: actual loaded overlay keys recorded")
        if valid_keys and valid_mutable and valid_bn:
            audit.check(set(mutable).issubset(keys) and set(bn).issubset(keys),
                        f"{name}: load scope includes all recorded trainable and BN state")
        audit.check(model.get("overlay_kind") == OVERLAY_KIND and model.get("coefficient_kind") == OVERLAY_KIND,
                    f"{name}: fixed final EMA overlay kind")
        audit.check(type(model.get("overlay_epoch")) is int and model["overlay_epoch"] == 3
                    and model.get("coefficient_epoch") == 3, f"{name}: fixed epoch 3 readout")
        audit.check(model.get("overlay_audit_passed") is True and model.get("all_overlay_bn_exact_official") is True,
                    f"{name}: loader recorded audit and exact official BN checks")
    else:
        audit.check(keys == [], f"{name}: official reference loads no overlay")
    fingerprint = receipt.get("fingerprint", {})
    audit.check(fingerprint.get("evaluator_version") == EVALUATOR_VERSION,
                f"{name}: new evaluator cache identity")
    for key, value in configuration.items():
        audit.check(fingerprint.get(key) == value, f"{name}: fingerprint matches common configuration {key}")
    audit.check(receipt.get("readout_view") ==
                "postprocessed native one2one detections after top-k/conf; not full raw candidates",
                f"{name}: prediction receipt declares output-level scope")
    weights = fingerprint.get("weights", {})
    expected_weight_fields = {"base_weights", "coefficients"} if trained else {"checkpoint"}
    valid_weights = isinstance(weights, dict) and set(weights) == expected_weight_fields
    audit.check(valid_weights, f"{name}: official vs overlay weight fingerprint fields")
    if valid_weights:
        for key, value in weights.items():
            valid_value = (isinstance(value, dict) and isinstance(value.get("sha256"), str)
                           and len(value["sha256"]) == 64 and isinstance(value.get("path"), str)
                           and integer(value.get("bytes")) and value["bytes"] > 0)
            audit.check(valid_value, f"{name}/{key}: weight content fingerprint present")
        base = weights.get("base_weights" if trained else "checkpoint", {})
        audit.check(base.get("path") == model.get("source_checkpoint"),
                    f"{name}: source checkpoint path matches content fingerprint")
        if trained:
            audit.check(weights["coefficients"].get("path") == model.get("coefficient_file"),
                        f"{name}: overlay file path matches content fingerprint")
            audit.check(model.get("overlay_base_weights_sha256") == base.get("sha256"),
                        f"{name}: loader checked overlay initialization against original checkpoint hash")
    else:
        base = {}
    return {"mutable": mutable, "bn": bn, "loaded": keys,
            "source_checkpoint": model.get("source_checkpoint"),
            "base_sha256": base.get("sha256")}


def verify(run, np):
    audit = Audit(run)
    summary = audit.read_json(run / "SUMMARY.json")
    if summary is None:
        return audit.report()
    audit.check(summary.get("status") == "complete", "top-level evaluation complete")
    audit.check(summary.get("image_count") == EXPECTED_IMAGES, "top-level image count is 5000")
    audit.check(summary.get("covers_all_annotation_images") is True, "covers all annotation images")
    configuration = summary.get("configuration", {})
    audit.check(configuration.get("evaluator_version") == EVALUATOR_VERSION, "prototype-tail evaluator version")
    audit.check(configuration.get("overlay_scope_prefixes") == list(OVERLAY_PREFIXES), "configured exact overlay scope")
    ids = summary.get("configuration", {}).get("image_ids", [])
    if not audit.check(isinstance(ids, list) and len(ids) == EXPECTED_IMAGES and all(integer(i) for i in ids) and len(set(ids)) == EXPECTED_IMAGES,
                       "configuration has 5000 unique integer image identities"):
        return audit.report()
    metrics = summary.get("metrics")
    if not audit.check(isinstance(metrics, dict) and len(metrics) == EXPECTED_ARMS,
                       "exactly three model arms have COCO metrics"):
        return audit.report()
    names = list(metrics)
    audit.check(set(names) == {"baseline", "official", "acd"}, "baseline, official and ACD arm identities")
    baseline = summary.get("baseline")
    if not audit.check(baseline in names, "baseline is a recorded model arm"):
        return audit.report()
    if not audit.check(all(isinstance(name, str) and name and Path(name).name == name and name not in (".", "..") for name in names), "safe model directory names"):
        return audit.report()
    receipts = summary.get("prediction_receipts", {})
    frozen_hashes = []
    overlay_metadata = {}
    for name in names:
        receipt = audit.read_json(run / name / "COMPLETE.json")
        if receipt is None:
            continue
        audit.check(receipts.get(name) == receipt, f"{name}: prediction receipt matches SUMMARY")
        audit.check(receipt.get("status") == "prediction_complete" and receipt.get("image_count") == EXPECTED_IMAGES,
                    f"{name}: 5000-image prediction completion receipt")
        digest = receipt.get("frozen_state_sha256")
        audit.check(isinstance(digest, str) and len(digest) == 64, f"{name}: frozen-state digest present")
        frozen_hashes.append(digest)
        overlay_metadata[name] = verify_overlay_metadata(audit, name, receipt, configuration)
    audit.check(len(frozen_hashes) == EXPECTED_ARMS and len(set(frozen_hashes)) == 1, "all three stored frozen-state digests exactly equal")
    if len(overlay_metadata) == EXPECTED_ARMS and all(overlay_metadata.values()):
        reference = overlay_metadata[baseline]
        for name, metadata in overlay_metadata.items():
            audit.check(metadata["mutable"] == reference["mutable"] and metadata["bn"] == reference["bn"],
                        f"{name}: all arms declare identical mutable and frozen BN scope")
            audit.check(metadata["source_checkpoint"] == reference["source_checkpoint"],
                        f"{name}: all arms use the same original checkpoint path")
            audit.check(metadata["base_sha256"] == reference["base_sha256"],
                        f"{name}: all arms use the same original checkpoint content hash")
        audit.check(overlay_metadata["baseline"]["loaded"] == overlay_metadata["acd"]["loaded"],
                    "baseline and ACD load exactly the same state-key scope")
    verify_coco(audit, summary, names, ids, np)
    for name in names:
        if name != baseline:
            verify_pair(audit, name, baseline, summary, ids, np)
    return audit.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Transferred final evaluation Run directory containing SUMMARY.json")
    parser.add_argument("--out", type=Path, help="Optional NEW audit JSON file; existing paths are refused")
    args = parser.parse_args()
    run = args.run.resolve()
    try:
        import numpy as np
    except ImportError as error:
        report = {"audit_version": VERSION, "run": str(run), "status": "missing_dependency", "passed": False,
                  "errors": [str(error)], "missing_dependency": "numpy"}
    else:
        try:
            report = verify(run, np)
        except Exception as error:
            report = {"audit_version": VERSION, "run": str(run), "status": "failed", "passed": False,
                      "errors": [{"check": "unhandled invalid input", "detail": f"{type(error).__name__}: {error}"}]}
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.out is not None:
        try:
            with args.out.resolve().open("x", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
        except OSError as error:
            print(json.dumps({"status": "audit_output_refused", "passed": False, "out": str(args.out.resolve()), "error": str(error)}))
            return 2
    print(rendered)
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    sys.exit(main())
