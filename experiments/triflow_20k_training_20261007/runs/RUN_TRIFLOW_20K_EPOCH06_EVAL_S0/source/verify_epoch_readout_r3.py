"""Independent TriFlow epoch-progress aggregate readout verification, NumPy plus stdlib only.

AP accumulation and paired point-estimate checks reused from the completed ACD
verifier, source SHA256 8fa15e5788a53ffc501cb96a82506ea419fbf609247db381d1df8ddb34598ef9. No evaluator import, inference, GT mask re-matching,
or bootstrap CI recomputation. --out exclusively creates a new audit file.
The pilot aggregate verifier was authenticated as SHA256
0cb12c2b70963f95b2876e9962df1dc85f717f72cbcd06cf1b276e13452bce03;
its final-head checks are replaced by actual epoch-boundary snapshot evidence.
This verifier does not declare the scheduled training complete, even at epoch8.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from diagnostics_contract_r3 import verify_contract, REQUIRED_DIAGNOSTICS
VERSION="triflow_20k_epoch_readout_verify_r3_v1"
EVALUATOR_VERSION="triflow_20k_epoch_eval_r3_v1"
EXPECTED_IMAGES=5000
EXPECTED_ARMS=2
ABS_TOLERANCE=1e-12
OFFICIAL_SHA256="16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"

class Audit:
    def __init__(self, run):
        self.run = run
        self.checks = 0
        self.errors = []
        self.error_count = 0
        self.missing = []
        self.missing_count = 0
        self.evidence = {}
        self.results = {}

    def check(self, condition, description, detail=None):
        self.checks += 1
        if not condition:
            self.error_count += 1
            error = {"check": description}
            if detail is not None:
                error["detail"] = detail
            if len(self.errors) < 50:
                self.errors.append(error)
        return bool(condition)

    def require(self, path):
        if not path.is_file():
            self.missing_count += 1
            if len(self.missing) < 50:
                self.missing.append(str(path))
            return False
        return True

    def fingerprint(self, path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        self.evidence[path.relative_to(self.run).as_posix()] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}

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
                "verifier_source": str(Path(__file__).resolve()), "verifier_source_sha256": content_sha256(Path(__file__)),
                "interpreter": sys.executable, "python_version": sys.version,
                "run": str(self.run), "status": "missing" if self.missing else "failed" if self.errors else "passed",
                "passed": not self.missing and not self.errors, "checks": self.checks,
                "missing": self.missing, "missing_count": self.missing_count,
                "errors": self.errors, "error_count": self.error_count,
                "failure_examples_limited_to": 50, "evidence": self.evidence,
                "results": self.results, "absolute_tolerance": ABS_TOLERANCE,
                "limitations": ["Aggregate readout verification only; COCO matches, mask decoding and GT are not recomputed",
                                "Paired point estimates and sample/seed metadata are checked only when measured in this Run; matching and bootstrap confidence intervals are not recomputed",
                                "Stored frozen-state digests and parity records are cross-checked; model tensors are not loaded",
                                "Stored original training identity/actual coverage/source evidence is cross-checked; original training images and JSON are not reloaded",
                                "One training seed does not quantify seed uncertainty; paired IoU intervals are not AP intervals"]}


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
    if not np.all(np.isfinite(arrays["scores"])) or not np.all(np.isfinite(arrays["recall"])):
        raise ValueError("Accumulated scores or recall contain nonfinite values")
    if not np.all((arrays["recall"] == -1) | ((arrays["recall"] >= 0) & (arrays["recall"] <= 1))):
        raise ValueError("Accumulated recall contains invalid values")
    expected_categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                           21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                           41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                           59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
                           80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    if not np.array_equal(categories, expected_categories):
        raise ValueError("Accumulated categories are not the canonical 80 original COCO category IDs")
    m = int(np.flatnonzero(maxima == 100)[0])
    t75 = int(np.flatnonzero(np.isclose(thresholds, .75, rtol=0, atol=1e-14))[0])

    def mean_valid(values):
        valid = values[values > -1]
        return float(np.mean(valid)) if valid.size else None

    t50 = int(np.flatnonzero(np.isclose(thresholds, .5, rtol=0, atol=1e-14))[0])
    return {"AP": mean_valid(precision[:, :, :, 0, m]),
            "AP50": mean_valid(precision[t50:t50+1, :, :, 0, m]),
            "AP75": mean_valid(precision[t75:t75+1, :, :, 0, m]),
            "APsmall": mean_valid(precision[:, :, :, 1, m]),
            "APmedium": mean_valid(precision[:, :, :, 2, m]),
            "APlarge": mean_valid(precision[:, :, :, 3, m]),
            "AR1": mean_valid(arrays["recall"][:, :, 0, 0]),
            "AR10": mean_valid(arrays["recall"][:, :, 0, 1]),
            "AR100": mean_valid(arrays["recall"][:, :, 0, m]),
            "ARsmall": mean_valid(arrays["recall"][:, :, 1, m]),
            "ARmedium": mean_valid(arrays["recall"][:, :, 2, m]),
            "ARlarge": mean_valid(arrays["recall"][:, :, 3, m])}


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


def canonical_sha(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_subset_selector(audit, evidence, head):
    """Independent ID-rank reconstruction; no GT mask/category is read."""
    universe, selected = evidence.get("original_image_ids"), evidence.get("selected_image_ids")
    valid = isinstance(universe, list) and len(universe) == 118287 and all(type(iid) is int for iid in universe) and universe == sorted(set(universe))
    if not audit.check(valid, "selector archived original118287 unique sorted image-ID universe"):
        return
    expected = sorted(sorted(universe, key=lambda iid: (hashlib.sha256(("triflow_train20k_seed0:"+str(iid)).encode()).hexdigest(), iid))[:20000])
    identity = canonical_sha(expected)
    audit.check(selected == expected, "locked SHA-ranked20000 subset reconstructed independently from archived original IDs")
    audit.check(evidence.get("image_identity_sha256") == evidence.get("declared_subset_identity_sha256") == identity,
                "actual20k declared/visited identity SHA matches reconstructed selection")
    selector = evidence.get("subset_selector", {})
    audit.check(selector.get("selector_name") == "sha256_triflow_train20k_seed0_v1" and selector.get("selection_seed") == 0 and selector.get("rank_prefix") == "triflow_train20k_seed0:" and selector.get("selection_size") == 20000,
                "20k selector algorithm/seed/rank prefix fixed")
    audit.check(selector.get("uses_gt_categories_masks_or_image_contents") is False and selector.get("universe_images") == 118287 and selector.get("selected_images") == 20000,
                "selector reads ID rank only; original universe and declared subset counts explicit")
    audit.check(selector.get("universe_image_identity_sha256") == canonical_sha(universe) and selector.get("selected_image_identity_sha256") == identity,
                "selector original/selected identity hashes")
    audit.check(canonical_sha(selector) == evidence.get("subset_selector_sha256") == head.get("subset_selector_sha256") and selector == head.get("subset_selector"),
                "loader and provider bind same exact20k selector metadata")


def original_diagnostic_keys(run):
    parsed = ast.parse((run/"source"/"triflow_model.py").read_text(encoding="utf-8-sig"))
    model = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "TriFlowModel")
    compiler = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "_compile")
    for node in ast.walk(compiler):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and key.value == "diagnostics" and isinstance(value, ast.Dict):
                    return {ast.literal_eval(key) for key in value.keys}
    raise ValueError("Locked original compiler diagnostic field declaration not found")


def verify_diagnostics_execution(audit, summary, head, capture):
    execution = head.get("diagnostics_execution", {})
    configuration = summary.get("configuration", {})
    audit.check(execution == configuration.get("diagnostics_execution") == capture.get("diagnostics_execution"),
                "captured training runtime, evaluator runtime and HEAD provenance bind identical actual execution metadata")
    try:
        receipt = verify_contract(audit.run, execution)
    except (ValueError, OSError, KeyError, TypeError) as error:
        audit.check(False, "actual archived diagnostics runtime/parity/addendum/source contract", repr(error))
        return
    audit.check(execution.get("actual_runtime_changed") is True and execution.get("original_scientific_contract_preserved") is True,
                "original four sources identify scientific base; actual R3 runtime change is explicitly declared")
    runtime_sources = configuration.get("method_source_sha256", {})
    audit.check(runtime_sources.get("diagnostics_runtime.py") == execution.get("runtime_sha256")
                and runtime_sources.get("diagnostics_contract_r3.py") == execution.get("adapter_sha256"),
                "runtime and adapter actual evaluation source archives match execution contract")
    audit.check(content_sha256(Path(__file__).with_name("diagnostics_contract_r3.py")) == execution.get("adapter_sha256"),
                "independent verifier uses the same declared tensor-free diagnostics provenance adapter")
    for key in ("runtime", "parity", "addendum", "adapter", "verifier"):
        audit.fingerprint(audit.run/execution[key+"_source_file"])
    scope = "minimal" if summary.get("evaluation_metrics_scope") == "coco_ap_only" else "full"
    audit.check(head.get("evaluation_diagnostic_scope") == configuration.get("diagnostic_scope") == scope,
                "actual AP-only minimal or paired full diagnostic scope is explicit")
    audit.check(head.get("runtime_loaded_source_file") == execution.get("runtime_source_file"),
                "actual module runtime loaded from immutable evaluation archive")
    audit.check(capture.get("diagnostics_mode_at_boundary") in ("minimal", "full"),
                "actual paused parent diagnostic mode recorded")
    mode_counts = capture.get("diagnostics_mode_counts_this_execution", {})
    audit.check(set(mode_counts) == {"minimal", "full"} and all(integer(value) and value >= 0 for value in mode_counts.values())
                and sum(mode_counts.values()) > 0, "actual R3 training diagnostic mode calls recorded without inflating resumed-prefix counts")
    audit.results["diagnostics_execution"] = {"runtime_sha256": execution["runtime_sha256"],
        "parity_receipt_sha256": execution["parity_sha256"], "addendum_sha256": execution["addendum_sha256"],
        "parity_engineering_only": receipt["smoke_only"], "scientific_scale_claimed": receipt["scientific_scale_claimed"],
        "evaluation_scope": scope, "unmeasured_fields_are_not_zero_filled": True}


def verify_method(audit, summary, expected_epoch=None):
    """Authenticate a completed-epoch snapshot, without final-training claims."""
    configuration = summary.get("configuration", {})
    method = configuration.get("method", {})
    audit.check(method.get("gt_used_in_forward") is False, "epoch deployment excludes GT")
    audit.check(method.get("learned_coefficient_residual") is False, "unchanged ownership compiler")
    audit.check(method.get("first_post_conf_native_rows") == 64 and method.get("instance_chunk") == 4
                and method.get("eligible_confidence") == .001, "fixed first64 output domain and inference chunks4")
    runtime_names = {"evaluate_epoch_r3.py", "readout_support.py", "frozen_io.py", "triflow_model.py", "diagnostics_runtime.py", "diagnostics_contract_r3.py"}
    runtime_sources = configuration.get("method_source_sha256", {})
    audit.check(set(runtime_sources) == runtime_names, "all actual epoch runtime source identities declared")
    for name in runtime_names:
        path = audit.run / "source" / name
        if audit.require(path):
            audit.check(content_sha256(path) == runtime_sources.get(name), "epoch runtime source archive SHA: " + name)
            audit.fingerprint(path)
    audit.check(configuration.get("evaluator_sha256") == runtime_sources.get("evaluate_epoch_r3.py"),
                "actual evaluator source SHA equals archived epoch source")
    for name, expected in (("triflow_model.py", "1b96e0767ab1985875cdfec33cdd6e98a666ecd3ee5caefc7dbca9e80c35691e"),
                           ("frozen_io.py", "cff75eec547d9ec12eda5409bd273231ab99c16cdc6119d75f6d413098363293"),
                           ("readout_support.py", "475b3c1e9ad538579aed1b6fc536dab2ce053a3a526d80ba352e82e297243488")):
        audit.check(runtime_sources.get(name) == expected, "locked unchanged pilot source: " + name)
    head = audit.read_json(audit.run / "HEAD_PROVENANCE.json")
    epoch_provenance = audit.read_json(audit.run / "EPOCH_PROVENANCE.json")
    if head is None or epoch_provenance is None:
        return
    audit.check(head == epoch_provenance == summary.get("head_provenance"), "actual epoch and head provenance equal SUMMARY")
    epoch = head.get("evaluation_epoch")
    if not audit.check(integer(epoch) and 1 <= epoch <= 8, "actual completed evaluation epoch is one through eight"):
        return
    if expected_epoch is not None:
        audit.check(epoch == expected_epoch, "actual snapshot epoch equals observer-declared expected epoch")
    audit.check(head.get("kind") == "triflow_20k_training_snapshot" and head.get("intermediate_snapshot") is True,
                "actual epoch-boundary training snapshot, not a fabricated final head")
    audit.check(head.get("final_training_audit_claimed") is False, "epoch readout explicitly makes no final-training audit claim")
    for key in ("all_state_finite_fp32", "optimizer_all_finite", "all_completed_epochs_declared_subset_coverage_verified"):
        audit.check(head.get(key) is True, "actual epoch-loader evidence " + key)
    audit.check(head.get("smoke_only") is False and head.get("engineering_subset") is False, "epoch snapshot excludes engineering truncation")
    next_cursor = {"epoch": epoch + 1, "image_position": 0, "chunk_start": 0}
    audit.check(head.get("reason") == "epoch_boundary" and head.get("cursor") == next_cursor
                and head.get("completed_epochs") == epoch and head.get("training_budget_epochs") == 8,
                "actual snapshot is the completed epoch prefix and next-epoch boundary")
    audit.check(summary.get("evaluation_epoch") == epoch and summary.get("intermediate_snapshot") is True
                and summary.get("training_budget_epochs") == 8, "SUMMARY records actual progress epoch and scheduled budget")
    audit.check(head.get("declared_train_images") == 20000 and head.get("complete_declared_train_subset") is True
                and head.get("complete_original_train_split") is False, "declared20k subset coverage, not original full118287 coverage")
    audit.check(head.get("base_weights_sha256") == OFFICIAL_SHA256 and head.get("no_gt_forward") is True,
                "actual loaded snapshot uses locked frozen base and GT-free forward")
    audit.check(head.get("loaded_state_sha256") == head.get("head_state_sha256"), "actual loaded state equals captured snapshot state digest")
    audit.check(canonical_sha(head.get("configuration", {})) == head.get("configuration_sha256"), "snapshot configuration content SHA")
    audit.check(method.get("head_configuration_sha256") == head.get("configuration_sha256"), "runtime/snapshot configuration identities")
    core_source = audit.run / "source" / "triflow_model.py"
    if audit.require(core_source):
        parsed = ast.parse(core_source.read_text(encoding="utf-8-sig"))
        config_class = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "TriFlowConfig")
        defaults = {node.target.id: ast.literal_eval(node.value) for node in config_class.body
                    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
        defaults.update(feature_channels=256, instance_hidden_channels=64)
        audit.check(head.get("configuration") == defaults, "actual epoch configuration equals all locked core defaults/native channels")
    snapshot = audit.run / f"head_epoch_{epoch:02d}.pt"
    if audit.require(snapshot):
        audit.check(content_sha256(snapshot) == head.get("snapshot_sha256") and snapshot.stat().st_size == head.get("snapshot_bytes"),
                    "actual archived epoch snapshot bytes bind provenance")
        audit.fingerprint(snapshot)
    archive = audit.run / "epoch_provenance"
    train_sources = head.get("training_sources", {})
    train_names = {"train_20k.py", "stream_data.py", "frozen_io.py", "triflow_model.py"}
    audit.check(set(train_sources) == train_names, "actual captured training source identities declared")
    for name in train_names:
        path = archive / name
        if audit.require(path):
            audit.check(content_sha256(path) == train_sources.get(name), "actual captured training source archive SHA: " + name)
            audit.fingerprint(path)
            if name in ("frozen_io.py", "triflow_model.py"):
                audit.check(train_sources.get(name) == runtime_sources.get(name), "snapshot/runtime source identity: " + name)
    stream_inputs = audit.read_json(archive / "STREAM_INPUTS.json")
    inputs = audit.read_json(archive / "TRAINING_INPUTS.json")
    epochs_file = audit.read_json(archive / "EPOCHS_AT_BOUNDARY.json")
    task = audit.read_json(archive / "TASK_GRADIENT_EVIDENCE_AT_BOUNDARY.json")
    state = audit.read_json(archive / "SNAPSHOT_STATE_EVIDENCE.json")
    for filename, key in (("STREAM_INPUTS.json", "stream_inputs_sha256"), ("TRAINING_INPUTS.json", "training_inputs_sha256")):
        path = archive / filename
        if audit.require(path):
            audit.check(content_sha256(path) == head.get(key), "actual copied input evidence bytes: " + filename)
    derived_files = {"EPOCHS_AT_BOUNDARY.json", "TASK_GRADIENT_EVIDENCE_AT_BOUNDARY.json", "SNAPSHOT_STATE_EVIDENCE.json"}
    derived_hashes = head.get("derived_evidence_sha256", {})
    audit.check(set(derived_hashes) == derived_files, "exact actual tensor-free boundary evidence files declared")
    for filename in derived_files:
        path = archive / filename
        if audit.require(path):
            audit.check(content_sha256(path) == derived_hashes.get(filename), "actual boundary-derived evidence bytes: " + filename)
    for name, value in (("epoch history", epochs_file), ("task probes", task), ("snapshot state", state)):
        if value is not None:
            audit.check(value.get("derived_from_snapshot_sha256") == head.get("snapshot_sha256"),
                        name + ": derived from this exact archived actual snapshot")
    if stream_inputs is not None:
        audit.check(stream_inputs.get("complete_declared_train_subset") is True and stream_inputs.get("listed_images") == 20000
                    and stream_inputs.get("complete_original_train_split") is False and stream_inputs.get("original_split_images") == 118287
                    and stream_inputs.get("engineering_subset") is False, "actual original input authenticates the declared20k subset")
        audit.check(stream_inputs.get("annotation_sha256") == "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"
                    and stream_inputs.get("base_weights_sha256") == OFFICIAL_SHA256, "actual original JSON and frozen base identities")
        data_names = ("annotation_sha256", "images_list_sha256", "image_identity_sha256", "declared_subset_identity_sha256", "subset_selector_sha256")
        audit.check({key: stream_inputs.get(key) for key in data_names} == head.get("data_hashes"), "actual provider/head data identities")
        audit.check(stream_inputs.get("sources") == {name: train_sources.get(name) for name in ("stream_data.py", "frozen_io.py", "triflow_model.py")},
                    "actual provider/train source identities")
        verify_subset_selector(audit, stream_inputs, head)
    if inputs is not None:
        contract = inputs.get("contract", {})
        audit.check(contract == head.get("contract") and canonical_sha(contract) == inputs.get("contract_sha256") == head.get("contract_sha256"),
                    "actual training input/captured snapshot contract identity")
        audit.check(contract.get("sources") == inputs.get("sources") == train_sources
                    and contract.get("data_hashes") == inputs.get("data_hashes") == head.get("data_hashes"), "actual contract/input/source/data agreement")
        audit.check(contract.get("configuration_sha256") == inputs.get("configuration_sha256") == head.get("configuration_sha256")
                    and contract.get("configuration") == head.get("configuration"), "actual contract/input/module configuration agreement")
        audit.check(contract.get("epochs") == 8 and contract.get("effective_images") == 20000
                    and contract.get("max_images") is None and contract.get("smoke_only") is False
                    and contract.get("engineering_subset") is False and contract.get("complete_declared_train_subset") is True
                    and contract.get("complete_original_train_split") is False, "actual declared20k eight-epoch contract; completed prefix kept separate")
        audit.check(contract.get("effective_image_identity_sha256") == head.get("data_hashes", {}).get("image_identity_sha256"),
                    "declared contract image identity equals actual selected20k universe")
        audit.check(contract.get("seed") == 0 and contract.get("instance_chunk") == 4 and contract.get("augmentation") is False
                    and contract.get("precision") == "FP32, TF32 disabled, no AMP" and contract.get("optimizer") == {
                        "class": "AdamW", "lr": 3e-4, "weight_decay": 1e-4, "clip_grad_norm": 10}, "unchanged actual optimizer/precision/seed recipe")
        execution_name = Path(str(inputs.get("execution_directory", "")).replace("\\", "/")).name
        audit.check(execution_name == head.get("training_run_id"), "actual epoch snapshot source training Run identity")
    epochs = epochs_file.get("epochs", []) if epochs_file is not None else []
    valid_epochs = len(epochs) == epoch and [row.get("epoch") for row in epochs] == list(range(1, epoch + 1))
    audit.check(valid_epochs, "actual completed consecutive epoch prefix, without future epochs")
    if valid_epochs:
        identity = head.get("data_hashes", {}).get("image_identity_sha256")
        audit.check(all(row.get("images") == 20000 and row.get("visited_unique_images") == 20000
                        and row.get("visited_image_identity_sha256") == identity
                        and row.get("actual_completed_image_ids_match_declared") is True for row in epochs),
                    "actual completed epochs each visited the exact declared20k identities")
        audit.check(all(integer(row.get("instances")) and row["instances"] > 0
                        and integer(row.get("applied_steps")) and row["applied_steps"] > 0 for row in epochs),
                    "every completed epoch has actual usable instances and applied updates")
    if state is not None:
        audit.check(state.get("execution_diagnostics_metadata") == head.get("diagnostics_execution"),
                    "actual derived snapshot explicitly binds its R3 execution metadata in addition to original scientific contract")
        audit.check(state.get("kind") == "triflow_20k_training_snapshot" and state.get("run_id") == head.get("training_run_id")
                    and state.get("reason") == "epoch_boundary" and state.get("cursor") == next_cursor
                    and state.get("completed_epochs") == epoch and state.get("is_final_training_audit") is False,
                    "actual derived state remains a completed-epoch snapshot, not final audit")
        applied, attempts = state.get("applied"), state.get("attempts")
        audit.check(integer(applied) and applied > 0 and attempts == applied, "all actual attempted prefix updates applied")
        if valid_epochs:
            audit.check(sum(row.get("applied_steps", 0) for row in epochs) == applied, "epoch applied-step totals equal actual snapshot state")
            audit.check(sum(row.get("instances", 0) for row in epochs) == state.get("instances"), "epoch instance totals equal actual snapshot state")
        finite_groups = state.get("finite_groups", {})
        groups = {"token_projection", "ownership_embeddings", "ownership_projection", "cross_attention", "interaction_norm", "field_head"}
        audit.check(set(finite_groups) == groups and all(value is True for value in finite_groups.values()), "actual prefix gradients remained finite in every parameter group")
        audit.check(state.get("optimizer_all_finite") is True and state.get("head_state_all_finite_fp32") is True
                    and state.get("head_state_sha256") == head.get("loaded_state_sha256"), "actual finite optimizer and FP32 loaded snapshot state")
        audit.check(state.get("contract_sha256") == head.get("contract_sha256"), "actual derived state/snapshot contract identity")
        provider = state.get("provider_state_evidence", {})
        audit.check(provider.get("data_hashes") == head.get("data_hashes")
                    and provider.get("sources") == {name: train_sources.get(name) for name in ("stream_data.py", "frozen_io.py", "triflow_model.py")},
                    "actual snapshot provider source/data evidence")
        audit.check(provider.get("observed_unique_images") == 20000 and provider.get("unique_images", {}).get("images") == 20000
                    and provider.get("observed_image_identity_sha256") == head.get("data_hashes", {}).get("image_identity_sha256"),
                    "actual snapshot provider observed every declared20k identity")
        audit.check(provider.get("dimensions", {}).get("feature_channels") == 256
                    and provider.get("dimensions", {}).get("instance_hidden_channels") == 64, "actual provider native feature/hidden dimensions")
        frozen = state.get("frozen_integrity", {})
        for key in ("passed", "all_state_exact", "all_params_requires_grad_false", "all_bn_eval", "all_gradients_none"):
            audit.check(frozen.get(key) is True, "actual boundary frozen integrity " + key)
        audit.check(frozen == head.get("frozen_integrity"), "actual boundary and loader frozen evidence identical")
        evaluation_digest = summary.get("prediction_receipts", {}).get("baseline", {}).get("frozen_state_sha256")
        audit.check(frozen.get("initial_state_sha256") == frozen.get("final_state_sha256")
                    == provider.get("frozen_initial_state_sha256") == evaluation_digest,
                    "actual training/evaluation frozen model states equal before and after")
    if task is not None:
        probes = task.get("evidence", [])
        audit.check(len(probes) == epoch and [row.get("epoch") for row in probes] == list(range(1, epoch + 1)),
                    "actual task-only probe prefix matches completed epochs")
        joint_count = 0
        for index, row in enumerate(probes):
            phi, attention = row.get("phi_task_gradient_norm"), row.get("cross_attention_task_gradient_norm")
            finite = finite_number(phi) and finite_number(attention) and phi >= 0 and attention >= 0
            audit.check(finite and row.get("all_task_gradients_finite") is True, f"actual prefix task probe {index}: finite phi and attention")
            if finite:
                joint = phi > 0 and attention > 0
                audit.check(row.get("task_gradient_phi_attention_same_probe_nonzero") is joint, f"actual prefix task probe {index}: same-probe nonzero flag")
                joint_count += int(joint)
        audit.check(joint_count > 0 and joint_count == task.get("same_probe_joint_count") == head.get("phi_attention_same_probe_count"),
                    "actual same-probe nonzero phi/attention evidence reaches learned compiler")
    capture = audit.read_json(audit.run / "BOUNDARY_CAPTURE.json")
    if capture is not None:
        verify_diagnostics_execution(audit, summary, head, capture)
        expected_cursor = {"epoch": epoch + 1, "image_position": 0, "chunk_start": 0}
        audit.check(capture.get("capture_passed") is True and capture.get("reason") == "epoch_boundary",
                    "observer captured an actual completed-epoch boundary")
        audit.check(capture.get("engineering_only") is False and capture.get("final_head_kind_claimed") is False,
                    "observer capture is formal progress and does not pretend a final head")
        audit.check(capture.get("epoch") == capture.get("completed_epoch_count") == epoch
                    and capture.get("cursor") == expected_cursor, "observer completed-epoch count and next-unapplied cursor")
        audit.check(capture.get("training_run_id") == head.get("training_run_id")
                    and capture.get("snapshot_sha256") == head.get("snapshot_sha256")
                    and capture.get("head_state_sha256") == head.get("head_state_sha256"), "observer/actual loader snapshot, state and Run identity")
        audit.check(capture.get("training_source_sha256") == train_sources and capture.get("data_hashes") == head.get("data_hashes")
                    and capture.get("contract_sha256") == head.get("contract_sha256"), "observer/actual loader source, data and contract identity")
        callbacks = capture.get("callback_script_sha256", {})
        audit.check(set(callbacks) == {"epoch_evaluator", "epoch_verifier"}
                    and callbacks.get("epoch_evaluator") == runtime_sources.get("evaluate_epoch_r3.py")
                    and callbacks.get("epoch_verifier") == content_sha256(Path(__file__)),
                    "observer bound the actual epoch evaluator and this independent verifier source")
        audit.check(capture.get("observer_source_sha256") == head.get("observer_source_sha256"),
                    "capture and loader bind the same actual observer source")
        observer_source = audit.run / "epoch_provenance" / "train_epoch_observer_r3.py"
        audit.check(capture.get("observer_source_file") == "epoch_provenance/train_epoch_observer_r3.py", "actual observer source file recorded")
        if audit.require(observer_source):
            audit.check(content_sha256(observer_source) == capture.get("observer_source_sha256"), "actual observer source archive SHA")
            audit.fingerprint(observer_source)
    parity = audit.read_json(audit.run / "PARENT_STATE_PARITY.json")
    if parity is not None:
        audit.check(parity.get("epoch") == epoch and parity.get("training_run_id") == head.get("training_run_id"),
                    "actual post-evaluator parent parity epoch/Run identity")
        audit.check(parity.get("scope") == "after evaluator, before independent verifier", "parent parity timing has no verification-output dependency")
        digest_names = {"head_state_sha256", "optimizer_state_sha256", "rng_state_sha256", "training_state_sha256",
                        "provider_state_sha256", "frozen_model_state_sha256", "execution_context_sha256"}
        before, after, equal = parity.get("before", {}), parity.get("after", {}), parity.get("equal_fields", {})
        valid_shape = isinstance(before, dict) and isinstance(after, dict) and isinstance(equal, dict)
        valid_shape = valid_shape and set(before) == set(after) == set(equal) == digest_names
        audit.check(valid_shape, "actual parent parity supplies every scientific state component")
        if valid_shape:
            valid_digests = all(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
                                for value in list(before.values()) + list(after.values()))
            audit.check(valid_digests, "actual parent component digests are SHA256 hex values")
            audit.check(before == after and all(equal[name] is True for name in digest_names)
                        and parity.get("passed") is True, "actual head/optimizer/RNG/training/provider/frozen/context state unchanged after evaluator")
            audit.check(before.get("head_state_sha256") == head.get("head_state_sha256"), "captured module state equals actual paused parent module state")
            if capture is not None:
                audit.check(capture.get("parent_scientific_digests_before") == before, "immutable capture and actual post-evaluator parity share original parent digests")
        audit.check(parity.get("engineering_only") is False, "parent parity is formal progress evaluation evidence")
        capture_path = audit.run / "BOUNDARY_CAPTURE.json"
        if audit.require(capture_path):
            audit.check(parity.get("capture_receipt_sha256") == content_sha256(capture_path), "parent parity binds actual immutable boundary capture bytes")
            audit.check(head.get("capture_receipt_sha256") == content_sha256(capture_path), "loader binds actual immutable boundary capture bytes")
        if capture is not None:
            audit.check(parity.get("observer_source_sha256") == capture.get("observer_source_sha256"), "capture and parity use the same actual observer source")
    audit.results["epoch_snapshot_contract"] = {
        "evaluation_epoch": epoch, "completed_training_epochs": head.get("completed_epochs"),
        "scheduled_training_epochs": 8, "declared_training_images": 20000,
        "snapshot_sha256": head.get("snapshot_sha256"), "loaded_state_sha256": head.get("loaded_state_sha256"),
        "intermediate_snapshot": head.get("intermediate_snapshot"), "final_training_audit_claimed": False,
        "scientific_scope": "completed-epoch progress readout; scheduled final training audit is not supplied here"}

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
    fingerprint = original.get("fingerprint", {})
    configuration = summary.get("configuration", {})
    audit.check(fingerprint.get("weights", {}).get("checkpoint", {}).get("sha256") == OFFICIAL_SHA256,
                "original reused baseline locked official checkpoint bytes")
    audit.check(fingerprint.get("image_ids") == ids and fingerprint.get("annotations", {}).get("sha256") == configuration.get("annotations", {}).get("sha256"),
                "original reused baseline supplied val image/annotation identities")
    audit.check(fingerprint.get("images_list", {}).get("sha256") == configuration.get("images_list", {}).get("sha256"),
                "original reused baseline exact val list bytes")
    audit.check(fingerprint.get("vendor_source_sha256") == configuration.get("vendor_source_sha256"),
                "original reused baseline vendor decoding source bytes")
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
    full_diagnostics = original_diagnostic_keys(audit.run)
    required = set(REQUIRED_DIAGNOSTICS)
    expected_scope = summary.get("configuration", {}).get("diagnostic_scope")
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
            scope, unmeasured = chunk.get("diagnostic_scope"), chunk.get("diagnostics_unmeasured")
            actual_rows = chunk.get("native_output_rows", [])
            audit.check(scope == expected_scope and scope in ("minimal", "full"), f"compiler line {line}: actual chunk diagnostic scope")
            audit.check(isinstance(unmeasured, list) and len(unmeasured) == len(set(unmeasured)),
                        f"compiler line {line}: actual unmeasured diagnostic keys declared once")
            expected_keys = required if scope == "minimal" else full_diagnostics
            audit.check(set(diag) == expected_keys, f"compiler line {line}: actual diagnostic fields agree with scope; omitted fields not fabricated")
            audit.check(isinstance(unmeasured, list) and set(unmeasured) == (full_diagnostics-required if scope == "minimal" else set()),
                        f"compiler line {line}: unmeasured keys match locked original compiler AST")
            for key in required:
                values = diag.get(key)
                audit.check(isinstance(values, list) and len(values) == len(actual_rows), f"compiler line {line}: {key} actual per-instance shape")
                if isinstance(values, list):
                    if key == "trust_saturated":
                        audit.check(all(type(value) is bool for value in values), f"compiler line {line}: actual trust saturation booleans")
                    else:
                        audit.check(all(integer(value) and value >= 0 for value in values), f"compiler line {line}: actual nonnegative compiler counts {key}")
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


def verify(run, np, expected_epoch=None):
    audit = Audit(run)
    audit.results["numpy_version"] = np.__version__
    summary = audit.read_json(run/"SUMMARY.json")
    if summary is None:
        return audit.report()
    audit.check(summary.get("status") == "complete", "top-level evaluation complete")
    completion = audit.read_json(run/"EVALUATION_COMPLETE.json")
    if completion is not None:
        audit.check(completion.get("status") == "completed" and completion.get("images") == 5000,
                    "actual evaluation completion marker")
        audit.check(completion.get("summary_sha256") == content_sha256(run/"SUMMARY.json"), "evaluation completion binds actual SUMMARY bytes")
    inputs = audit.read_json(run/"EVALUATION_INPUTS.json")
    if inputs is not None:
        audit.check(inputs.get("configuration") == summary.get("configuration"), "actual evaluation inputs and summary configurations agree")
        audit.check(inputs.get("head") == summary.get("head_provenance"), "actual evaluation inputs and summary head provenance agree")
    audit.check(summary.get("image_count") == 5000 and summary.get("covers_all_annotation_images") is True,
                "all original 5000 annotation images evaluated")
    configuration = summary.get("configuration", {})
    audit.check(configuration.get("evaluator_version") == EVALUATOR_VERSION, "TriFlow evaluator version")
    audit.check(configuration.get("ultralytics") == "8.4.100", "locked Ultralytics evaluation version")
    audit.check(configuration.get("annotations", {}).get("sha256") == "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f",
                "locked original COCO val2017 annotation bytes")
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
    verify_method(audit, summary, expected_epoch)
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
    scope = summary.get("evaluation_metrics_scope")
    paired = summary.get("paired_evaluated")
    if scope == "coco_ap_only":
        audit.check(paired is False and summary.get("paired") == {}, "AP-only epoch declares paired diagnostics unmeasured")
        audit.results["paired_recomputed"] = None
        audit.results["paired_scope"] = "not measured in this AP-only epoch evaluation"
    elif paired is True:
        audit.check(scope == "coco_ap_and_paired_diagnostic", "paired epoch explicitly declares its metric scope")
        audit.check(set(summary.get("paired", {})) == {"triflow"}, "actual requested paired diagnostic exists")
        verify_pair(audit, "triflow", "baseline", summary, ids, np)
    else:
        audit.check(False, "epoch evaluation explicitly declares AP-only or actual AP-plus-paired scope")
    return audit.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Transferred epoch-progress evaluation Run directory containing SUMMARY.json")
    parser.add_argument("--out", type=Path, help="Optional NEW audit JSON file; existing paths are refused")
    parser.add_argument("--expected-epoch", type=int, choices=range(1, 9),
                        help="Optional observer-declared completed epoch; must match the actual snapshot")
    args = parser.parse_args()
    run = args.run.resolve()
    try:
        import numpy as np
    except ImportError as error:
        report = {"audit_version": VERSION, "run": str(run), "status": "missing_dependency", "passed": False,
                  "errors": [str(error)], "missing_dependency": "numpy"}
    else:
        try:
            report = verify(run, np, args.expected_epoch)
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
    raise SystemExit(main())

