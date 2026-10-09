"""Independent TriFlow aggregate readout verification, NumPy plus stdlib only.

AP accumulation and paired point-estimate checks reused from the completed ACD
verifier, source SHA256 8fa15e5788a53ffc501cb96a82506ea419fbf609247db381d1df8ddb34598ef9. No evaluator import, inference, GT mask re-matching,
or bootstrap CI recomputation. --out exclusively creates a new audit file.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
VERSION="triflow_readout_verify_v1"
EVALUATOR_VERSION="triflow_eval_v1"
EXPECTED_IMAGES=5000
EXPECTED_ARMS=2
ABS_TOLERANCE=1e-12
OFFICIAL_SHA256="16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"

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


def content_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_method(audit, summary):
    configuration = summary.get("configuration", {})
    method = configuration.get("method", {})
    audit.check(method.get("gt_used_in_forward") is False, "method deployment excludes GT")
    audit.check(method.get("learned_coefficient_residual") is False, "method does not directly learn a coefficient residual")
    audit.check(method.get("first_post_conf_native_rows") == 64 and method.get("instance_chunk") == 4,
                "fixed first64 native rows, inference chunks4")
    audit.check(method.get("eligible_confidence") == .001, "fixed deployment confidence eligibility")
    expected_sources = {"evaluate_triflow.py", "readout_support.py", "frozen_io.py", "triflow_model.py"}
    source_hashes = configuration.get("method_source_sha256", {})
    audit.check(set(source_hashes) == expected_sources, "all actual method/extraction/readout sources recorded")
    for name, digest in source_hashes.items():
        path = audit.run/"source"/name
        if audit.require(path):
            audit.check(content_sha256(path) == digest, "source snapshot SHA: "+name)
            audit.fingerprint(path)
    head = audit.read_json(audit.run/"HEAD_PROVENANCE.json")
    if head is None:
        return
    audit.check(head == summary.get("head_provenance"), "head provenance equals SUMMARY")
    audit.check(head.get("kind") == "triflow_module_final" and head.get("epoch") == 8
                and head.get("smoke_only") is False, "formal module fixed epoch8, smoke excluded")
    audit.check(head.get("audit_passed") is True and head.get("all_state_finite_fp32") is True,
                "module loader actually checked audited finite FP32 state")
    audit.check(head.get("no_gt_forward") is True, "module loader records GT-free forward")
    audit.check(head.get("base_weights_sha256") == OFFICIAL_SHA256, "head initialized against locked official checkpoint")
    encoded = json.dumps(head.get("configuration", {}), sort_keys=True, separators=(",", ":")).encode()
    audit.check(hashlib.sha256(encoded).hexdigest() == head.get("configuration_sha256"), "head configuration content SHA")
    audit.check(method.get("head_configuration_sha256") == head.get("configuration_sha256"), "runtime and trained configuration identities")
    archived = audit.run/"head_provenance"
    training = audit.read_json(archived/"TRAINING_AUDIT.json")
    inputs = audit.read_json(archived/"TRAINING_INPUTS.json")
    task = audit.read_json(archived/"TASK_GRADIENT_EVIDENCE.json")
    completion = audit.read_json(archived/"TRAINING_COMPLETE.json")
    cache = audit.read_json(archived/"CACHE_RECEIPT.json")
    if cache is not None:
        audit.check(content_sha256(archived/"CACHE_RECEIPT.json") == head.get("cache_receipt_sha256"), "head binds actual frozen cache receipt bytes")
        audit.check(head.get("data_hashes") == {"annotation_sha256": cache.get("annotation_sha256"),
                                               "images_list_sha256": cache.get("images_list_sha256")}, "head/cache training data identities")
        audit.check(cache.get("frozen_integrity", {}).get("passed") is True, "actual extraction freeze audit present")
    if training is not None:
        audit.check(content_sha256(archived/"TRAINING_AUDIT.json") == head.get("training_audit_sha256"), "head binds actual training audit bytes")
        audit.check(training.get("passed") is True and training.get("epochs") == 8, "actual module training completed audited epoch8")
        for key in ("all_applied_gradients_finite", "all_parameters_finite", "all_groups_really_changed",
                    "compiler_task_gradient_observed_nonzero", "frozen_cache_receipt_integrity_passed",
                    "compiler_task_phi_and_attention_same_probe_observed_nonzero"):
            audit.check(training.get(key) is True, "training proof "+key)
        audit.check(training.get("final_head_state_sha256") == head.get("loaded_state_sha256"), "loader state equals trained final state digest")
        audit.check(training.get("sources") == head.get("training_sources"), "training/head source identities")
    if inputs is not None:
        audit.check(inputs.get("epochs") == 8 and inputs.get("seed") == 0, "fixed8 epochs seed0 input record")
        audit.check(inputs.get("sources") == head.get("training_sources"), "training inputs/head source identities")
        audit.check(inputs.get("configuration_sha256") == head.get("configuration_sha256"), "training inputs/head configuration identity")
    if task is not None:
        audit.check(task.get("any_observed_nonzero") is True and task.get("no_hidden_warmup") is True,
                    "actual task-only compiler backward evidence and no hidden warmup")
        audit.check(content_sha256(archived/"TASK_GRADIENT_EVIDENCE.json") == head.get("task_gradient_evidence_sha256"),
                    "loaded head provenance binds actual task gradient evidence bytes")
        joint_count = 0
        for index, row in enumerate(task.get("evidence", [])):
            phi, attention = row.get("phi_task_gradient_norm"), row.get("cross_attention_task_gradient_norm")
            finite = finite_number(phi) and finite_number(attention)
            audit.check(finite and row.get("all_task_gradients_finite") is True, f"actual task probe {index}: finite phi and attention norms")
            if finite:
                actual_joint = phi > 0 and attention > 0
                audit.check(row.get("task_gradient_phi_attention_same_probe_nonzero") is actual_joint,
                            f"actual task probe {index}: joint nonzero flag matches both norms")
                joint_count += int(actual_joint)
        audit.check(joint_count > 0 and task.get("any_observed_phi_attention_same_probe_nonzero") is True,
                    "actual same probe nonzero phi and cross-attention task gradients observed")
        audit.check(joint_count == head.get("phi_attention_same_probe_count"), "loader jointly observed task probe count")
        if training is not None:
            audit.check(joint_count == training.get("compiler_task_phi_attention_same_probe_count"),
                        "training audit jointly observed task probe count")
    if completion is not None:
        audit.check(completion.get("head_sha256") == head.get("head_sha256") and completion.get("audit_passed") is True,
                    "formal completion binds loaded head bytes")
    for name, digest in head.get("training_sources", {}).items():
        path = archived/name
        if audit.require(path):
            audit.check(content_sha256(path) == digest, "trained source archive SHA: "+name)
            if name in ("frozen_io.py", "triflow_model.py"):
                audit.check(source_hashes.get(name) == digest, "runtime source equals actual training source: "+name)
    audit.results["method_source_contract"] = {"formal_epoch": head.get("epoch"),
                                              "configuration_sha256": head.get("configuration_sha256"),
                                              "head_sha256": head.get("head_sha256"), "no_gt_forward": head.get("no_gt_forward")}


def verify_replay(audit, summary, ids):
    replay = audit.read_json(audit.run/"BASELINE_REPLAY_VERIFICATION.json")
    original = audit.read_json(audit.run/"ORIGINAL_BASELINE_RECEIPT.json")
    if replay is None or original is None:
        return
    audit.check(replay == summary.get("baseline_replay"), "baseline replay receipt equals SUMMARY")
    audit.check(replay.get("status") == "passed" and replay.get("images") == 5000,
                "actual full5000 baseline replay passed")
    for key in ("all_5000_images_replayed", "all_native_forward_exact", "all_baseline_identity_and_rle_exact"):
        audit.check(replay.get(key) is True, "baseline replay "+key)
    audit.check(content_sha256(audit.run/"ORIGINAL_BASELINE_RECEIPT.json") == replay.get("original_baseline_receipt_sha256"),
                "original baseline receipt preserved verbatim")
    audit.check(replay.get("original_frozen_state_sha256") == original.get("frozen_state_sha256"),
                "old digest retained with explicit old scope")
    audit.check(isinstance(replay.get("original_digest_scope"), str) and "not equal" in replay["original_digest_scope"],
                "legacy partial digest distinguished from current full model digest")
    integrity = replay.get("current_frozen_model_integrity", {})
    audit.check(integrity.get("passed") is True and integrity.get("all_state_exact") is True,
                "actual full-model before/after freeze audit passed")
    audit.check(integrity.get("initial_state_sha256") == integrity.get("final_state_sha256"),
                "actual full frozen state unchanged")
    rows, detections, chain = {}, 0, []
    for line, row in read_jsonl(audit, audit.run/"BASELINE_REPLAY_IMAGES.jsonl"):
        iid = row.get("image_id")
        valid = integer(iid) and iid in ids and iid not in rows
        audit.check(valid, f"replay line {line}: unique supplied image identity")
        audit.check(row.get("native_replay_exact") is True and row.get("baseline_rows_rle_exact") is True,
                    f"replay line {line}: actual native and identity/RLE equality")
        audit.check(integer(row.get("detection_count")) and row["detection_count"] >= 0,
                    f"replay line {line}: nonnegative detection count")
        if valid:
            rows[iid] = row
            path = audit.run/"baseline"/"images"/f"{iid:012d}.json"
            if audit.require(path):
                digest = content_sha256(path)
                audit.check(digest == row.get("source_image_cache_sha256"), f"baseline {iid}: original image cache bytes copied exactly")
                chain.append(digest)
            detections += row.get("detection_count", 0)
    audit.check(len(rows) == 5000 and set(rows) == set(ids), "5000 unique complete baseline replay records")
    audit.check(detections == replay.get("detections"), "baseline replay detection total")
    audit.check(hashlib.sha256("".join(chain).encode()).hexdigest() == replay.get("baseline_image_cache_sha256_chain"),
                "baseline image-cache hash chain")
    selected, refined, unsupported, images, count_total = 0, 0, 0, set(), {"boundary_valid_count": 0, "root_count": 0,
                                            "neighbor_transition_count": 0, "active_anchor_count": 0,
                                            "trust_saturated_instances": 0}
    for line, record in read_jsonl(audit, audit.run/"COMPILER_IMAGES.jsonl"):
        iid = record.get("image_id")
        audit.check(integer(iid) and iid in ids and iid not in images, f"compiler line {line}: unique image identity")
        images.add(iid)
        audit.check(record.get("gt_used_in_forward") is False, f"compiler line {line}: GT-free deployment")
        audit.check(record.get("unselected_coefficients_exact") is True and record.get("box_class_confidence_order_exact") is True,
                    f"compiler line {line}: unselected coefficient and native identity parity")
        chosen = record.get("selected_rows", [])
        eligible = record.get("eligible_output_rows", [])
        audit.check(chosen == eligible[:64] and record.get("selected_count") == len(chosen), f"compiler line {line}: exact first64 selection")
        selected += len(chosen)
        mappings = record.get("mappings", [])
        unsupported_records = record.get("unsupported_roi", [])
        executable = record.get("executable_rows", [])
        unsupported_rows = [item.get("native_output_row") for item in unsupported_records]
        audit.check(len(mappings) == record.get("refined_count") == len(executable), f"compiler line {line}: executable source identity mapping count")
        audit.check(record.get("unsupported_roi_count") == len(unsupported_records) and
                    set(executable).isdisjoint(unsupported_rows) and set(executable+unsupported_rows) == set(chosen),
                    f"compiler line {line}: first64 domain split without replacement")
        audit.check(all(item.get("kept_c0_exact") is True and item.get("reason") == "empty_native_prototype_roi" for item in unsupported_records),
                    f"compiler line {line}: explicit unsupported ROI identity readout")
        refined += len(executable)
        unsupported += len(unsupported_records)
        for mapping in mappings:
            own = mapping.get("raw_index")
            neighbors = mapping.get("predicted_neighbor_raw_indices", [])
            valid = mapping.get("neighbor_valid", [])
            audit.check(all(not v or neighbors[j] != own for j, v in enumerate(valid)),
                        f"compiler line {line}: predicted neighbor excludes same raw index")
        for chunk in record.get("chunks", []):
            diag = chunk.get("diagnostics", {})
            for key in ("boundary_valid_count", "root_count", "neighbor_transition_count", "active_anchor_count"):
                count_total[key] += sum(diag.get(key, []))
            count_total["trust_saturated_instances"] += sum(bool(v) for v in diag.get("trust_saturated", []))
    audit.check(len(images) == 5000 and images == set(ids), "5000 complete compiler inference records")
    audit.check(selected == summary.get("selected_instances"), "selected-instance total vs SUMMARY")
    audit.check(refined == summary.get("refined_instances") and unsupported == summary.get("unsupported_roi_instances")
                and selected == refined+unsupported, "actual refinement/unsupported first64 counts vs SUMMARY")
    count_total["invalid_boundary_root_count"] = count_total["boundary_valid_count"]-count_total["root_count"]
    audit.check(count_total == summary.get("compiler_counts"), "compiler root/anchor/trust totals reconstructed")
    audit.results["baseline_replay"] = {"images": len(rows), "detections": detections,
                                        "compiler_images": len(images), "selected_instances": selected}


def verify(run, np):
    audit = Audit(run)
    summary = audit.read_json(run/"SUMMARY.json")
    if summary is None:
        return audit.report()
    audit.check(summary.get("status") == "complete", "top-level evaluation complete")
    audit.check(summary.get("image_count") == 5000 and summary.get("covers_all_annotation_images") is True,
                "all original 5000 annotation images evaluated")
    configuration = summary.get("configuration", {})
    audit.check(configuration.get("evaluator_version") == EVALUATOR_VERSION, "TriFlow evaluator version")
    fixed = {"imgsz": 640, "shape": [640, 640], "batch": 1, "half": False, "conf": .001, "max_det": 300,
             "coco_maxDets": [1, 10, 100], "rect": False, "scaleup": False, "augment": False,
             "native_one2one": True, "tf32": False}
    for key, value in fixed.items():
        audit.check(configuration.get("config", {}).get(key) == value, "fixed official readout config "+key)
    ids = configuration.get("image_ids", [])
    if not audit.check(isinstance(ids, list) and len(ids) == 5000 and len(set(ids)) == 5000 and all(integer(i) for i in ids),
                       "5000 unique integer image identities"):
        return audit.report()
    metrics = summary.get("metrics", {})
    if not audit.check(set(metrics) == {"baseline", "triflow"} and summary.get("baseline") == "baseline", "exact baseline/TriFlow arms"):
        return audit.report()
    verify_method(audit, summary)
    verify_replay(audit, summary, ids)
    frozen = []
    for name in ("baseline", "triflow"):
        receipt = audit.read_json(run/name/"COMPLETE.json")
        if receipt is None:
            continue
        audit.check(receipt == summary.get("prediction_receipts", {}).get(name), name+": receipt equals SUMMARY")
        audit.check(receipt.get("status") == "prediction_complete" and receipt.get("image_count") == 5000, name+": prediction complete5000")
        audit.check(receipt.get("frozen_digest_scope") == "all actual frozen official model state_dict tensors", name+": full actual frozen digest scope")
        digest = receipt.get("frozen_state_sha256")
        audit.check(isinstance(digest, str) and len(digest) == 64, name+": frozen digest valid")
        frozen.append(digest)
        model = receipt.get("model", {})
        audit.check(model.get("official_base_sha256") == OFFICIAL_SHA256 and model.get("official_weights_unchanged") is True,
                    name+": official weights unchanged")
        audit.check(model.get("learned_module_loaded") is (name == "triflow") and model.get("no_gt_forward") is True,
                    name+": expected module deployment and no GT")
        if name == "triflow":
            audit.check(model.get("head") == summary.get("head_provenance"), "TriFlow actual head provenance identity")
        for key, value in configuration.items():
            audit.check(receipt.get("fingerprint", {}).get(key) == value, name+": fingerprint common configuration "+key)
        predictions = run/name/"predictions.json"
        if audit.require(predictions):
            audit.check(content_sha256(predictions) == receipt.get("predictions_sha256"), name+": actual prediction bytes SHA")
            metric = audit.read_json(run/name/"COCO_METRICS.json")
            if metric is not None:
                audit.check(metric.get("predictions_sha256") == receipt.get("predictions_sha256"), name+": COCO metrics bind prediction bytes")
        if name == "baseline":
            audit.check(receipt.get("source_original_predictions_sha256") == receipt.get("predictions_sha256"), "baseline original prediction bytes preserved")
    audit.check(len(frozen) == 2 and len(set(frozen)) == 1, "both arms share the actual same full frozen model digest")
    verify_coco(audit, summary, ["baseline", "triflow"], ids, np)
    verify_pair(audit, "triflow", "baseline", summary, ids, np)
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

