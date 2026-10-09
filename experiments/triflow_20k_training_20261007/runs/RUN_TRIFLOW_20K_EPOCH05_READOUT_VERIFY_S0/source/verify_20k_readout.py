"""Independent 20k TriFlow aggregate readout verification, NumPy plus stdlib only.

AP accumulation and paired point-estimate checks reused from the completed ACD
verifier, source SHA256 8fa15e5788a53ffc501cb96a82506ea419fbf609247db381d1df8ddb34598ef9. No evaluator import, inference, GT mask re-matching,
or bootstrap CI recomputation. --out exclusively creates a new audit file.
The pilot aggregate verifier was authenticated as SHA256
0cb12c2b70963f95b2876e9962df1dc85f717f72cbcd06cf1b276e13452bce03;
its cache-specific head checks are replaced by actual declared20k online evidence.
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
VERSION="triflow_20k_readout_verify_v1"
EVALUATOR_VERSION="triflow_20k_eval_v1"
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
                "verifier_source": str(Path(__file__).resolve()), "verifier_source_sha256": content_sha256(Path(__file__)),
                "interpreter": sys.executable, "python_version": sys.version,
                "run": str(self.run), "status": "missing" if self.missing else "failed" if self.errors else "passed",
                "passed": not self.missing and not self.errors, "checks": self.checks,
                "missing": self.missing, "missing_count": self.missing_count,
                "errors": self.errors, "error_count": self.error_count,
                "failure_examples_limited_to": 50, "evidence": self.evidence,
                "results": self.results, "absolute_tolerance": ABS_TOLERANCE,
                "limitations": ["Aggregate readout verification only; COCO matches, mask decoding and GT are not recomputed",
                                "Image-cluster bootstrap point estimates and stored sample/seed metadata are checked; confidence intervals are not recomputed",
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


def verify_method(audit, summary):
    """Cross-check actual declared20k online evidence, never a pilot-cache receipt."""
    configuration = summary.get("configuration", {})
    method = configuration.get("method", {})
    audit.check(method.get("gt_used_in_forward") is False, "method deployment excludes GT")
    audit.check(method.get("learned_coefficient_residual") is False, "ownership compiler, no learned coefficient residual")
    audit.check(method.get("first_post_conf_native_rows") == 64 and method.get("instance_chunk") == 4,
                "fixed first64 output rows, inference chunks4")
    audit.check(method.get("eligible_confidence") == .001, "fixed output confidence eligibility")
    runtime_names = {"evaluate_20k.py", "readout_support.py", "frozen_io.py", "triflow_model.py"}
    source_hashes = configuration.get("method_source_sha256", {})
    audit.check(set(source_hashes) == runtime_names, "all actual runtime/extraction/readout source hashes declared")
    for name in runtime_names:
        path = audit.run / "source" / name
        if audit.require(path):
            audit.check(content_sha256(path) == source_hashes.get(name), "actual runtime source archive SHA: " + name)
            audit.fingerprint(path)
    audit.check(configuration.get("evaluator_sha256") == source_hashes.get("evaluate_20k.py"),
                "runtime evaluator SHA matches archived actual source")
    for name, expected in (("triflow_model.py", "1b96e0767ab1985875cdfec33cdd6e98a666ecd3ee5caefc7dbca9e80c35691e"),
                           ("frozen_io.py", "cff75eec547d9ec12eda5409bd273231ab99c16cdc6119d75f6d413098363293"),
                           ("readout_support.py", "475b3c1e9ad538579aed1b6fc536dab2ce053a3a526d80ba352e82e297243488")):
        audit.check(source_hashes.get(name) == expected, "protocol unchanged pilot source: " + name)
    head = audit.read_json(audit.run / "HEAD_PROVENANCE.json")
    if head is None:
        return
    audit.check(head == summary.get("head_provenance"), "actual head provenance equals SUMMARY")
    audit.check(head.get("kind") == "triflow_20k_module_final" and head.get("epoch") == 8
                and head.get("smoke_only") is False and head.get("complete_declared_train_subset") is True and head.get("complete_original_train_split") is False,
                "formal declared20k final8 checkpoint; engineering truncation excluded; original full coverage not claimed")
    audit.check(head.get("declared_train_images") == 20000, "actual loader records declared20k training image count")
    for key in ("audit_passed", "all_state_finite_fp32", "no_gt_forward", "all_epochs_declared_subset_coverage_verified"):
        audit.check(head.get(key) is True, "actual loader evidence " + key)
    audit.check(head.get("base_weights_sha256") == OFFICIAL_SHA256, "loaded head uses locked official frozen base")
    audit.check(canonical_sha(head.get("configuration", {})) == head.get("configuration_sha256"),
                "head configuration content SHA")
    core_source = audit.run / "source" / "triflow_model.py"
    if audit.require(core_source):
        parsed = ast.parse(core_source.read_text(encoding="utf-8-sig"))
        config_class = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "TriFlowConfig")
        defaults = {node.target.id: ast.literal_eval(node.value) for node in config_class.body
                    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
        defaults.update(feature_channels=256, instance_hidden_channels=64)
        audit.check(head.get("configuration") == defaults, "actual head configuration equals all locked core defaults and native channels")
    audit.check(method.get("head_configuration_sha256") == head.get("configuration_sha256"),
                "runtime and training configuration identities")
    archived = audit.run / "head_provenance"
    training = audit.read_json(archived / "TRAINING_AUDIT.json")
    inputs = audit.read_json(archived / "TRAINING_INPUTS.json")
    task = audit.read_json(archived / "TASK_GRADIENT_EVIDENCE.json")
    completion = audit.read_json(archived / "TRAINING_COMPLETE.json")
    stream = audit.read_json(archived / "STREAM_RECEIPT.json")
    stream_inputs = audit.read_json(archived / "STREAM_INPUTS.json")
    epochs_file = audit.read_json(archived / "EPOCHS.json")
    for filename, key in (("TRAINING_AUDIT.json", "training_audit_sha256"),
                          ("TASK_GRADIENT_EVIDENCE.json", "task_gradient_evidence_sha256"),
                          ("STREAM_RECEIPT.json", "stream_receipt_sha256"),
                          ("STREAM_INPUTS.json", "stream_inputs_sha256"),
                          ("EPOCHS.json", "epochs_sha256")):
        path = archived / filename
        if audit.require(path):
            audit.check(content_sha256(path) == head.get(key), "loaded head binds actual evidence bytes: " + filename)
    train_names = {"train_20k.py", "stream_data.py", "frozen_io.py", "triflow_model.py"}
    train_sources = head.get("training_sources", {})
    audit.check(set(train_sources) == train_names, "all actual online training sources declared")
    for name in train_names:
        path = archived / name
        if audit.require(path):
            audit.check(content_sha256(path) == train_sources.get(name), "actual training source archive SHA: " + name)
            audit.fingerprint(path)
            if name in ("frozen_io.py", "triflow_model.py"):
                audit.check(train_sources.get(name) == source_hashes.get(name), "training/runtime source identity: " + name)
    if stream is not None and stream_inputs is not None:
        for evidence_name, evidence in (("completed stream", stream), ("stream inputs", stream_inputs)):
            audit.check(evidence.get("complete_declared_train_subset") is True
                        and evidence.get("listed_images") == 20000 and evidence.get("original_split_images") == 118287 and evidence.get("complete_original_train_split") is False,
                        evidence_name + ": locked declared20k subset of original118287 image identities")
            audit.check(evidence.get("engineering_subset") is False, evidence_name + ": formal input excludes engineering subset")
            audit.check(evidence.get("annotation_sha256") == "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d",
                        evidence_name + ": locked complete original COCO training annotations")
            audit.check(evidence.get("base_weights_sha256") == OFFICIAL_SHA256, evidence_name + ": locked official base bytes")
            expected_data = {key: evidence.get(key) for key in ("annotation_sha256", "images_list_sha256", "image_identity_sha256", "declared_subset_identity_sha256", "subset_selector_sha256")}
            audit.check(expected_data == head.get("data_hashes"), evidence_name + ": exact loaded-head training data hashes")
            verify_subset_selector(audit, evidence, head)
            audit.check(evidence.get("sources") == {name: train_sources.get(name) for name in ("stream_data.py", "frozen_io.py", "triflow_model.py")},
                        evidence_name + ": provider source identities")
        audit.check(stream.get("all_listed_images_observed") is True and stream.get("observed_unique_images") == 20000,
                    "actual streaming observed every declared training image")
        audit.check(stream.get("unique_images", {}).get("images") == 20000, "actual unique training image count")
        frozen = stream.get("frozen_integrity", {})
        for key in ("passed", "all_state_exact", "all_params_requires_grad_false", "all_bn_eval", "all_gradients_none"):
            audit.check(frozen.get(key) is True, "online frozen actual integrity " + key)
        audit.check(frozen.get("initial_state_sha256") == frozen.get("final_state_sha256") == stream_inputs.get("frozen_initial_state_sha256"),
                    "online frozen model actual initial/final state identity")
        evaluation_digest = summary.get("prediction_receipts", {}).get("baseline", {}).get("frozen_state_sha256")
        audit.check(frozen.get("final_state_sha256") == evaluation_digest,
                    "training and final evaluation use exactly the same actual full frozen model state")
    if training is not None:
        audit.check(training.get("complete_original_train_split") is False, "20k training audit does not claim original118287 full coverage")
        audit.check(training.get("passed") is True and training.get("epochs") == 8 and training.get("completed_epochs") == 8,
                    "actual online module training passed all eight epochs")
        for key in ("all_applied_gradients_finite", "all_parameters_finite", "all_optimizer_states_finite", "all_groups_really_changed",
                    "compiler_task_gradient_observed_nonzero", "compiler_task_phi_and_attention_same_probe_observed_nonzero",
                    "frozen_input_mutation_checks_passed", "complete_declared_train_subset", "every_epoch_declared_images_complete",
                    "every_epoch_actual_unique_images_complete"):
            audit.check(training.get(key) is True, "actual 20k training proof " + key)
        audit.check(training.get("frozen_yolo_online") is True and training.get("smoke_only") is False,
                    "actual training used online frozen YOLO and declared20k subset")
        attempts, applied = training.get("optimizer_attempts"), training.get("optimizer_applied")
        audit.check(integer(applied) and applied > 0 and attempts == applied, "every recorded attempted optimizer update actually applied")
        audit.check(training.get("images_per_epoch") == 20000, "training audit original full images per epoch")
        if stream is not None:
            audit.check(training.get("frozen_integrity", {}).get("initial_state_sha256") == stream.get("frozen_integrity", {}).get("initial_state_sha256")
                        and training.get("frozen_integrity", {}).get("final_state_sha256") == stream.get("frozen_integrity", {}).get("final_state_sha256"),
                        "training audit/completed stream actual frozen state identities")
        audit.check(training.get("final_head_state_sha256") == head.get("loaded_state_sha256"), "actual loaded final state equals training digest")
        audit.check(training.get("sources") == train_sources and training.get("data_hashes") == head.get("data_hashes"),
                    "training audit/head source and data identities")
        audit.check(training.get("configuration_sha256") == head.get("configuration_sha256"), "training audit/head configuration identities")
        for key in ("stream_receipt_sha256", "stream_inputs_sha256", "epochs_sha256"):
            audit.check(training.get(key) == head.get(key), "training audit/head actual evidence SHA: " + key)
    if inputs is not None:
        contract = inputs.get("contract", {})
        audit.check(canonical_sha(contract) == inputs.get("contract_sha256") == head.get("contract_sha256"),
                    "actual online budget/source/data contract content SHA")
        audit.check(head.get("contract") == contract, "loader and actual training input contracts agree")
        execution_name = Path(str(inputs.get("execution_directory", "")).replace("\\", "/")).name
        audit.check(execution_name == head.get("training_run_id"), "actual loaded training Run identity")
        audit.check(inputs.get("parent") == head.get("parent"), "actual training input/loader resume parent identity")
        audit.check(contract.get("epochs") == inputs.get("epochs") == 8 and inputs.get("seed") == contract.get("seed") == 0,
                    "fixed eight formal epochs seed0")
        audit.check(inputs.get("images") == contract.get("effective_images") == 20000
                    and contract.get("max_images") is None and contract.get("smoke_only") is False
                    and contract.get("complete_declared_train_subset") is True and contract.get("complete_original_train_split") is False,
                    "actual 20k training contract forbids truncated list")
        audit.check(inputs.get("sources") == contract.get("sources") == train_sources, "training input/contract/head source identities")
        audit.check(inputs.get("data_hashes") == contract.get("data_hashes") == head.get("data_hashes"),
                    "training input/contract/head data identities")
        audit.check(contract.get("effective_image_identity_sha256") == head.get("data_hashes", {}).get("image_identity_sha256"),
                    "formal contract declared image identity set equals original full split")
        audit.check(inputs.get("configuration_sha256") == contract.get("configuration_sha256") == head.get("configuration_sha256"),
                    "training input/contract/head configuration identity")
        audit.check(contract.get("instance_chunk") == 4 and contract.get("optimizer") == {
                    "class": "AdamW", "lr": 3e-4, "weight_decay": 1e-4, "clip_grad_norm": 10},
                    "formal unchanged optimization recipe")
    epochs = epochs_file.get("epochs", []) if epochs_file is not None else []
    audit.check(len(epochs) == 8 and [row.get("epoch") for row in epochs] == list(range(1, 9)), "actual eight ordered epoch records")
    if len(epochs) == 8:
        audit.check(all(row.get("images") == 20000 for row in epochs), "actual declared image count covered in every epoch")
        expected_identity = head.get("data_hashes", {}).get("image_identity_sha256")
        audit.check(all(row.get("visited_unique_images") == 20000
                        and row.get("actual_completed_image_ids_match_declared") is True
                        and row.get("visited_image_identity_sha256") == expected_identity for row in epochs),
                    "actual eight unique completed image sets equal full original training split")
        audit.check(all(integer(row.get("instances")) and row["instances"] > 0 and integer(row.get("applied_steps"))
                        and row["applied_steps"] > 0 for row in epochs), "every full epoch has actual usable instances and updates")
        if training is not None:
            audit.check(sum(row.get("applied_steps", 0) for row in epochs) == training.get("optimizer_applied"), "epoch applied steps equal final training audit")
            audit.check(sum(row.get("instances", 0) for row in epochs) == training.get("instances"), "epoch trained instance counts equal final audit")
    if task is not None:
        audit.check(task.get("any_observed_nonzero") is True and task.get("no_hidden_warmup") is True,
                    "actual task-only compiler gradient probes with no hidden warmup")
        evidence = task.get("evidence", [])
        joint_count = 0
        for index, row in enumerate(evidence):
            phi, attention = row.get("phi_task_gradient_norm"), row.get("cross_attention_task_gradient_norm")
            finite = finite_number(phi) and finite_number(attention)
            audit.check(finite and row.get("all_task_gradients_finite") is True, f"task probe {index}: actual finite phi and attention gradients")
            if finite:
                joint = phi > 0 and attention > 0
                audit.check(row.get("task_gradient_phi_attention_same_probe_nonzero") is joint, f"task probe {index}: joint finite nonzero flag")
                joint_count += int(joint)
        audit.check(joint_count > 0 and task.get("any_observed_phi_attention_same_probe_nonzero") is True,
                    "actual same probe reaches both phi field and cross-attention")
        audit.check(joint_count == head.get("phi_attention_same_probe_count"), "loader actual joint gradient count")
        if training is not None:
            audit.check(joint_count == training.get("compiler_task_phi_attention_same_probe_count"), "training actual joint gradient count")
    if completion is not None:
        audit.check(completion.get("status") == "completed" and completion.get("head_sha256") == head.get("head_sha256")
                    and completion.get("audit_passed") is True, "formal training completion binds loaded head bytes")
        audit.check(completion.get("epochs") == 8 and completion.get("images_per_epoch") == 20000
                    and completion.get("smoke_only") is False and completion.get("complete_declared_train_subset") is True and completion.get("complete_original_train_split") is False,
                    "formal training completion records actual full eight epochs")
    audit.results["method_source_contract"] = {"formal_epoch": head.get("epoch"),
                    "training_images": training.get("images_per_epoch") if training is not None else None,
                    "expected_declared_training_images": 20000,
                    "configuration_sha256": head.get("configuration_sha256"), "head_sha256": head.get("head_sha256"),
                    "no_gt_forward": head.get("no_gt_forward"), "all_epochs_declared_subset_coverage_verified": head.get("all_epochs_declared_subset_coverage_verified")}

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


if __name__ == "__main__":
    raise SystemExit(main())

