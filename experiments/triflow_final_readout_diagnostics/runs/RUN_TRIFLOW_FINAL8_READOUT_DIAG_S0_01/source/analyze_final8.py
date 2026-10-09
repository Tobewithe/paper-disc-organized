"""CPU-only descriptive joins of returned epoch8 outputs; no inference or GT rematching."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import traceback

VERSION = "triflow_final8_descriptive_join_v1"
TRUST_BINS = ("[0,.25)", "[.25,.5)", "[.5,.75)", "[.75,.999999)", "[.999999,1]")
TRUST_THRESHOLD_FP32 = struct.unpack("f", struct.pack("f", .999999))[0]
ROOT_BINS = ("boundary_zero", "none", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1]")
DIAG_KEYS = ("trust_scale", "trust_saturated", "boundary_valid_count", "root_count", "neighbor_transition_count", "active_anchor_count")
IDENTITY_KEYS = ("image_id", "detection_index", "category_id", "model_class", "bbox", "score", "box_xyxy", "raw_input_box_xyxy", "raw_confidence")
SOURCE_FILES = ("COMPILER_IMAGES.jsonl", "SUMMARY.json", "paired_triflow_vs_baseline/INSTANCES.jsonl", "paired_triflow_vs_baseline/SUMMARY.json", "paired_triflow_vs_baseline/IMAGES.jsonl", "BASELINE_REPLAY_IMAGES.jsonl", "BASELINE_REPLAY_VERIFICATION.json", "ORIGINAL_BASELINE_RECEIPT.json", "baseline/COMPLETE.json", "triflow/COMPLETE.json", "source/evaluate_epoch_r3.py", "source/readout_support.py", "source/frozen_io.py", "source/triflow_model.py", "source/diagnostics_runtime.py", "source/diagnostics_contract_r3.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = Path(str(path)+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line, text in enumerate(f, 1):
            yield line, json.loads(text)


def rate(a, b):
    return a/b if b else None


def trust_bin(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        return "unknown"
    for upper, label in zip((.25, .5, .75, TRUST_THRESHOLD_FP32), TRUST_BINS):
        if value < upper:
            return label
    return TRUST_BINS[-1]


def root_bin(boundary, roots):
    if type(boundary) is not int or type(roots) is not int or not 0 <= roots <= boundary:
        return "unknown"
    if boundary == 0:
        return "boundary_zero"
    if roots == 0:
        return "none"
    fraction = roots/boundary
    return next(label for upper, label in zip((.25, .5, .75, 1), ROOT_BINS[2:]) if fraction <= upper)


def new_acc():
    return {"candidate_count": 0, "matched_count": 0, "baseline_success_count": 0, "baseline_failure_count": 0,
            "damage_count": 0, "repair_count": 0, "before_iou_sum": 0., "after_iou_sum": 0., "delta_iou_sum": 0.,
            "baseline_success_delta_sum": 0., "images": set(), "matched_images": set(), "gts": set()}


def add(acc, row):
    acc["candidate_count"] += 1
    acc["images"].add(row["image_id"])
    pair = row["paired"]
    if pair is None:
        return
    acc["matched_count"] += 1
    acc["matched_images"].add(row["image_id"])
    acc["gts"].add((row["image_id"], pair["annotation_id"]))
    for target, source in (("before_iou_sum", "baseline_mask_iou"), ("after_iou_sum", "method_mask_iou"), ("delta_iou_sum", "delta_mask_iou")):
        acc[target] += pair[source]
    acc["baseline_success_count"] += pair["baseline_success"]
    acc["baseline_failure_count"] += not pair["baseline_success"]
    acc["damage_count"] += pair["damage"]
    acc["repair_count"] += pair["repair"]
    if pair["baseline_success"]:
        acc["baseline_success_delta_sum"] += pair["delta_mask_iou"]


def finalize(acc):
    result = {key: value for key, value in acc.items() if key not in ("images", "matched_images", "gts")}
    result.update(images_with_candidates=len(acc["images"]), images_with_matched_candidates=len(acc["matched_images"]),
                  unique_associated_gt_count=len(acc["gts"]), matched_fraction_of_outputs=rate(acc["matched_count"], acc["candidate_count"]),
                  before_iou_mean=rate(acc["before_iou_sum"], acc["matched_count"]), after_iou_mean=rate(acc["after_iou_sum"], acc["matched_count"]),
                  delta_iou_mean=rate(acc["delta_iou_sum"], acc["matched_count"]),
                  baseline_success_delta_iou_mean=rate(acc["baseline_success_delta_sum"], acc["baseline_success_count"]),
                  damage_rate_of_baseline_success=rate(acc["damage_count"], acc["baseline_success_count"]),
                  repair_rate_of_baseline_failure=rate(acc["repair_count"], acc["baseline_failure_count"]))
    return result


def labels(cohort, diag, neighbor, own_raw):
    trust = trust_bin(diag.get("trust_scale")) if cohort == "refined" else "not_refined"
    root = root_bin(diag.get("boundary_valid_count"), diag.get("root_count")) if cohort == "refined" else "not_refined"
    saturated = ("true" if diag.get("trust_saturated") is True else "false" if diag.get("trust_saturated") is False else "unknown") if cohort == "refined" else "not_refined"
    root_presence = "none" if root == "none" else "boundary_zero" if root == "boundary_zero" else "some" if root in ROOT_BINS[2:] else root
    channel = "unknown" if cohort == "unknown_identity" else "not_refined"
    if cohort == "refined":
        boundary_on, anchor_on = diag["root_count"] > 0, diag["active_anchor_count"] > 0
        channel = "mixed" if boundary_on and anchor_on else "boundary_only" if boundary_on else "anchor_only" if anchor_on else "no_constraints"
    return {"cohort": cohort, "trust_bin": trust, "root_bin": root, "neighbor_valid_slots": str(neighbor.get("valid_slots")) if neighbor.get("valid_slots") is not None else "unknown",
            "neighbor_raw_type": neighbor["raw_type"], "own_raw_type": own_raw,
            "natural_channel": channel,
            "cross_trust_root_neighbor": "saturated="+saturated+"|root="+root_presence+"|neighbors="+neighbor["raw_type"]}


def analyze(source, protocol, run):
    source, protocol, run = source.resolve(), protocol.resolve(), run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if any((run/name).exists() for name in ("SUMMARY.json", "CANDIDATES_JOINED.jsonl", "SOURCE_LOCK.json")):
        raise FileExistsError("Analysis history is immutable; use another Run")
    files = {name: {"sha256_before": digest(source/name), "bytes": (source/name).stat().st_size} for name in SOURCE_FILES}
    protocol_hash = digest(protocol)
    originals = {name: digest(Path(__file__).with_name(name)) for name in ("analyze_final8.py", "verify_final8_analysis.py")}
    archived = run/"source"
    archived.mkdir(exist_ok=True)
    for name, sha in originals.items():
        target = archived/name
        if target.exists() and digest(target) != sha:
            raise ValueError("Executed analysis source differs from archive")
        if not target.exists():
            shutil.copy2(Path(__file__).with_name(name), target)
    origin = json.loads((source/"SUMMARY.json").read_text(encoding="utf-8"))
    pair_summary = json.loads((source/"paired_triflow_vs_baseline/SUMMARY.json").read_text(encoding="utf-8"))
    if origin.get("status") != "complete" or origin.get("evaluation_epoch") != 8 or origin.get("image_count") != 5000 or origin.get("paired", {}).get("triflow") != pair_summary or not pair_summary.get("paired_readout_valid"):
        raise ValueError("Complete authenticated epoch8 paired readout required")
    ids = origin["configuration"]["image_ids"]
    pairs, replay = defaultdict(dict), {}
    for _, row in jsonl(source/"paired_triflow_vs_baseline/INSTANCES.jsonl"):
        iid, index = row["image_id"], row["detection_index"]
        if index in pairs[iid]:
            raise ValueError("Duplicate paired detection identity")
        pairs[iid][index] = row
    for _, row in jsonl(source/"BASELINE_REPLAY_IMAGES.jsonl"):
        if row["image_id"] in replay:
            raise ValueError("Duplicate replay image identity")
        replay[row["image_id"]] = row
    if set(replay) != set(ids):
        raise ValueError("Baseline replay identity set differs")
    overall, grouped, gt_records = new_acc(), defaultdict(lambda: defaultdict(new_acc)), {}
    for label in ("refined", "unsupported_roi", "unrefined", "unknown_identity"):
        grouped["cohort"][label]
    for label in TRUST_BINS+("not_refined", "unknown"):
        grouped["trust_bin"][label]
    for label in ROOT_BINS+("not_refined", "unknown"):
        grouped["root_bin"][label]
    for label in ("no_constraints", "anchor_only", "boundary_only", "mixed", "not_refined", "unknown"):
        grouped["natural_channel"][label]
    total = Counter()
    unknown_reasons = Counter()
    baseline_files, method_files, image_rows, seen = [], [], [], set()
    chain = []
    with (run/"CANDIDATES_JOINED.jsonl").open("x", encoding="utf-8") as output:
        for _, compiler in jsonl(source/"COMPILER_IMAGES.jsonl"):
            iid = compiler["image_id"]
            if iid in seen or iid not in replay:
                raise ValueError("Compiler image identities are not unique/expected")
            seen.add(iid)
            base_path, method_path = source/"baseline/images"/f"{iid:012d}.json", source/"triflow/images"/f"{iid:012d}.json"
            base_bytes, method_bytes = base_path.read_bytes(), method_path.read_bytes()
            base_sha, method_sha = hashlib.sha256(base_bytes).hexdigest(), hashlib.sha256(method_bytes).hexdigest()
            baseline_files.append({"image_id": iid, "path": base_path.relative_to(source).as_posix(), "sha256_before": base_sha, "bytes": len(base_bytes)})
            method_files.append({"image_id": iid, "path": method_path.relative_to(source).as_posix(), "sha256_before": method_sha, "bytes": len(method_bytes)})
            chain.append(base_sha)
            base, method = json.loads(base_bytes), json.loads(method_bytes)
            detections, revised = base["detections"], method["detections"]
            image_errors = []
            if base_sha != replay[iid]["source_image_cache_sha256"]:
                image_errors.append("baseline_cache_hash_does_not_match_actual_replay")
            if base.get("image_id") != iid or method.get("image_id") != iid or len(detections) != len(revised) or len(detections) != len(compiler["eligible_output_rows"]):
                image_errors.append("image_or_output_cardinality_mismatch")
            if compiler.get("gt_used_in_forward") is not False:
                image_errors.append("deployment_gt_exclusion_not_recorded")
            if compiler.get("box_class_confidence_order_exact") is not True or compiler.get("unselected_coefficients_exact") is not True:
                image_errors.append("compiler_native_identity_parity_not_verified")
            mappings = {m["native_output_row"]: m for m in compiler["mappings"]}
            unsupported = {m["native_output_row"]: m for m in compiler["unsupported_roi"]}
            diag_by_row = {}
            for chunk in compiler["chunks"]:
                for local, native in enumerate(chunk["native_output_rows"]):
                    if native in diag_by_row:
                        raise ValueError("Compiler diagnostic native row duplicated")
                    diag_by_row[native] = {key: chunk["diagnostics"][key][local] for key in DIAG_KEYS if key in chunk["diagnostics"]}
            raw_counts = Counter(m["raw_index"] for m in list(mappings.values())+list(unsupported.values()))
            if len(mappings) != compiler["refined_count"] or set(mappings) != set(compiler["executable_rows"]) or set(mappings) != set(diag_by_row):
                image_errors.append("refined_mapping_chunk_cardinality_mismatch")
            if set(mappings)&set(unsupported) or set(mappings)|set(unsupported) != set(compiler["selected_rows"]):
                image_errors.append("selected_supported_partition_mismatch")
            image_acc, image_cohorts, image_diag = new_acc(), Counter(), Counter()
            for index, detection in enumerate(detections):
                errors = list(image_errors)
                if detection.get("detection_index") != index or detection.get("image_id") != iid:
                    errors.append("baseline_index_image_identity_mismatch")
                if index >= len(revised) or any(detection.get(key) != revised[index].get(key) for key in IDENTITY_KEYS):
                    errors.append("method_baseline_box_class_confidence_identity_mismatch")
                native = compiler["eligible_output_rows"][index] if index < len(compiler["eligible_output_rows"]) else None
                mapping, unsup = mappings.get(native), unsupported.get(native)
                cohort = "refined" if mapping else "unsupported_roi" if unsup else "unrefined"
                raw = mapping["raw_index"] if mapping else unsup["raw_index"] if unsup else None
                if mapping and mapping.get("post_conf_detection_index") != index or unsup and unsup.get("post_conf_detection_index") != index:
                    errors.append("explicit_postconf_index_mapping_mismatch")
                diag = diag_by_row.get(native, {})
                neighbor = {"valid_slots": None, "raw_type": "unknown_not_refined", "raw_indices": None, "output_rows": None}
                if mapping:
                    valid, nr, no = mapping["neighbor_valid"], mapping["predicted_neighbor_raw_indices"], mapping["predicted_neighbor_output_rows"]
                    if len(valid) != 2 or len(nr) != 2 or len(no) != 2 or any(type(v) is not bool for v in valid) or any(v and (type(nr[j]) is not int or nr[j] == raw or type(no[j]) is not int or no[j] < 0) for j, v in enumerate(valid)):
                        errors.append("invalid_predicted_neighbor_identity_slots")
                    else:
                        chosen = [nr[j] for j, v in enumerate(valid) if v]
                        kind = "no_neighbor" if not chosen else "one_neighbor" if len(chosen) == 1 else "duplicate" if len(set(chosen)) < len(chosen) else "distinct"
                        neighbor = {"valid_slots": len(chosen), "raw_type": kind, "raw_indices": nr, "output_rows": no}
                    if any(key not in diag for key in DIAG_KEYS) or trust_bin(diag.get("trust_scale")) == "unknown" or root_bin(diag.get("boundary_valid_count"), diag.get("root_count")) == "unknown" or type(diag.get("trust_saturated")) is not bool:
                        errors.append("missing_or_invalid_actual_compiler_diagnostics")
                pair = pairs.get(iid, {}).pop(index, None)
                if pair and (pair["category_id"] != detection["category_id"] or pair["baseline_score"] != detection["raw_confidence"]):
                    errors.append("paired_baseline_category_score_identity_mismatch")
                if pair and cohort in ("unsupported_roi", "unrefined") and (pair["delta_mask_iou"] != 0 or pair["damage"] or pair["repair"]):
                    errors.append("nonexecuted_candidate_has_nonzero_paired_change")
                if mapping and diag.get("root_count") == 0 and diag.get("active_anchor_count") == 0:
                    total["natural_no_constraints_instances"] += 1
                    total["natural_no_constraints_nonzero_coefficient_change"] += mapping["coefficient_max_abs_change"] != 0
                    total["natural_no_constraints_matched"] += pair is not None
                    total["natural_no_constraints_nonzero_paired_delta"] += bool(pair and pair["delta_mask_iou"] != 0)
                    if mapping["coefficient_max_abs_change"] != 0 or pair and (pair["delta_mask_iou"] != 0 or pair["damage"] or pair["repair"]):
                        errors.append("natural_no_constraints_zero_update_invariant_failed")
                own_raw = "unknown_unselected" if raw is None else "duplicate_selected_raw" if raw_counts[raw] > 1 else "unique_selected_raw"
                if errors:
                    unknown_reasons.update(set(errors))
                    cohort = "unknown_identity"
                    diag = {}
                    neighbor = {"valid_slots": None, "raw_type": "unknown_identity", "raw_indices": None, "output_rows": None}
                    own_raw = "unknown_identity"
                row = {"image_id": iid, "detection_index": index, "native_output_row": native, "raw_index": raw,
                       "cohort": cohort, "identity_verified": not errors, "unknown_reasons": sorted(set(errors)),
                       "baseline_identity": {key: detection.get(key) for key in ("category_id", "model_class", "bbox", "box_xyxy", "score", "raw_confidence")},
                       "diagnostics": diag, "neighbor": neighbor, "own_raw_multiplicity_selected": raw_counts[raw] if raw is not None else None,
                       "coefficient_max_abs_change": mapping["coefficient_max_abs_change"] if mapping else None,
                       "raw_identity_scope": "selected_native_outputs_only" if raw is not None else "unselected_native_raw_id_not_exported",
                       "paired": pair}
                row["strata"] = labels(cohort, diag, neighbor, own_raw)
                output.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
                add(overall, row); add(image_acc, row)
                image_cohorts[cohort] += 1
                total[cohort] += 1
                for family, label in row["strata"].items():
                    add(grouped[family][label], row)
                if cohort == "refined":
                    for key in ("boundary_valid_count", "root_count", "neighbor_transition_count", "active_anchor_count"):
                        total[key] += diag[key]; image_diag[key] += diag[key]
                    total["trust_saturated_instances"] += diag["trust_saturated"]
                    image_diag["trust_saturated_instances"] += diag["trust_saturated"]
                if pair:
                    gt_key = (iid, pair["annotation_id"])
                    gt = gt_records.setdefault(gt_key, {"image_id": iid, "annotation_id": pair["annotation_id"], "category_id": pair["category_id"], "associated_candidate_count": 0,
                        "baseline_success_candidate_count": 0, "baseline_failure_candidate_count": 0, "damaged_candidate_count": 0, "repaired_candidate_count": 0,
                        "cohort_counts": Counter(), "detection_indices": [], "known_raw_indices": []})
                    gt["associated_candidate_count"] += 1
                    gt["baseline_success_candidate_count"] += pair["baseline_success"]
                    gt["baseline_failure_candidate_count"] += not pair["baseline_success"]
                    gt["damaged_candidate_count"] += pair["damage"]
                    gt["repaired_candidate_count"] += pair["repair"]
                    gt["cohort_counts"][cohort] += 1
                    gt["detection_indices"].append(index)
                    if raw is not None:
                        gt["known_raw_indices"].append(raw)
            if pairs.get(iid):
                raise ValueError("Paired source contains detection indices outside baseline output set")
            image_rows.append({"image_id": iid, **finalize(image_acc), "cohort_counts": dict(image_cohorts),
                "cohort_fractions_of_image_outputs": {key: rate(image_cohorts[key], len(detections)) for key in ("refined", "unsupported_roi", "unrefined", "unknown_identity")},
                "compiler_counts": dict(image_diag), "has_damage": image_acc["damage_count"] > 0, "has_repair": image_acc["repair_count"] > 0})
    if seen != set(ids) or any(pairs.values()):
        raise ValueError("Not all source images/matched records were processed")
    original_chain = json.loads((source/"BASELINE_REPLAY_VERIFICATION.json").read_text())["baseline_image_cache_sha256_chain"]
    actual_chain = hashlib.sha256("".join(chain).encode()).hexdigest()
    if actual_chain != original_chain:
        raise ValueError("Actual baseline per-image identity chain differs from source replay receipt")
    with (run/"IMAGES.jsonl").open("x", encoding="utf-8") as f:
        for row in image_rows:
            f.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
    gt_labels = Counter()
    with (run/"GT_ASSOCIATIONS.jsonl").open("x", encoding="utf-8") as f:
        for key in sorted(gt_records):
            row = gt_records[key]
            row["known_raw_unique_count"] = len(set(row["known_raw_indices"]))
            row["known_raw_repeated_associations"] = len(row["known_raw_indices"])-row["known_raw_unique_count"]
            row["duplicate_candidate_association"] = row["associated_candidate_count"] > 1
            row["any_damage_label"] = row["damaged_candidate_count"] > 0
            row["all_associations_damage_label"] = row["damaged_candidate_count"] == row["associated_candidate_count"]
            row["any_repair_label"] = row["repaired_candidate_count"] > 0
            row["all_associations_repair_label"] = row["repaired_candidate_count"] == row["associated_candidate_count"]
            for field in ("duplicate_candidate_association", "any_damage_label", "all_associations_damage_label", "any_repair_label", "all_associations_repair_label"):
                gt_labels[field] += row[field]
            f.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
    for name, evidence in files.items():
        evidence["sha256_after"] = digest(source/name)
        if evidence["sha256_before"] != evidence["sha256_after"]:
            raise RuntimeError("Analysis changed source bytes: " + name)
    for collection in (baseline_files, method_files):
        for evidence in collection:
            evidence["sha256_after"] = digest(source/evidence["path"])
            if evidence["sha256_after"] != evidence["sha256_before"]:
                raise RuntimeError("Source image output changed during CPU analysis")
    if protocol_hash != digest(protocol):
        raise RuntimeError("Locked descriptive protocol changed during analysis")
    locks = {"source_run": str(source), "protocol": str(protocol), "protocol_sha256": protocol_hash,
        "analysis_source_sha256": originals, "files": files, "baseline_image_files": baseline_files, "method_image_files": method_files,
        "baseline_identity_chain_sha256": actual_chain, "all_before_after_sha256_equal": True}
    write_json(run/"SOURCE_LOCK.json", locks)
    summary = {"version": VERSION, "status": "complete" if not unknown_reasons else "complete_with_unknown_identities", "source_run": str(source),
        "protocol_sha256": protocol_hash, "source_lock_sha256": digest(run/"SOURCE_LOCK.json"), "inference_executed": False, "gt_rematched": False,
        "scope": "All baseline exported outputs and their previously fixed paired same-class GT associations; descriptive strata only",
        "join_reliability_passed": not unknown_reasons, "unknown_reasons_candidate_occurrences": dict(unknown_reasons),
        "overall": finalize(overall), "cohort_counts": {key: total[key] for key in ("refined", "unsupported_roi", "unrefined", "unknown_identity")},
        "strata": {family: {label: dict(finalize(acc), fraction_supplied_5000_images_with_candidates=len(acc["images"])/5000,
            fraction_supplied_5000_images_with_matched_candidates=len(acc["matched_images"])/5000) for label, acc in sorted(groups.items())} for family, groups in sorted(grouped.items())},
        "compiler_counts_reconstructed": {key: total[key] for key in ("boundary_valid_count", "root_count", "neighbor_transition_count", "active_anchor_count", "trust_saturated_instances")},
        "natural_channel_invariants": {key: total[key] for key in ("natural_no_constraints_instances", "natural_no_constraints_nonzero_coefficient_change", "natural_no_constraints_matched", "natural_no_constraints_nonzero_paired_delta")},
        "natural_channel_scope": "This solve's naturally occurring single-channel subset; not a full-cohort branch-disable ablation or general causal claim",
        "source_expected": {"paired_statistics": pair_summary["statistics"], "unique_associated_gt_count": pair_summary["unique_matched_gt_instances"],
            "selected_instances": origin["selected_instances"], "refined_instances": origin["refined_instances"], "unsupported_roi_instances": origin["unsupported_roi_instances"], "compiler_counts": origin["compiler_counts"]},
        "image_labels": {"images": len(image_rows), "with_matched_candidates": sum(row["matched_count"] > 0 for row in image_rows), "with_damage": sum(row["has_damage"] for row in image_rows), "with_repair": sum(row["has_repair"] for row in image_rows)},
        "gt_association_labels": {"associated_gt_count": len(gt_records), **dict(gt_labels), "unit": "labels among candidate-associated GT only; not GT success/damage/repair rates"},
        "bins": {"trust": list(TRUST_BINS), "root_fraction": list(ROOT_BINS), "threshold_selection": "frozen descriptive bins; no threshold tuning",
            "nominal_saturation_boundary": .999999, "effective_saturation_boundary_fp32": TRUST_THRESHOLD_FP32,
            "saturation_group_authority": "recorded source trust_saturated bool; trust bins use original FP32 boundary representation"},
        "unknown": ["unexported native raw IDs for unrefined outputs", "complete raw geometry/five-state recovery", "unassociated GT", "causal explanation", "AP or intervention gains within diagnostic bins"],
        "completed_at": now()}
    write_json(run/"SUMMARY.json", summary)
    write_json(run/"ANALYSIS_COMPLETE.json", {"status": "completed", "summary_sha256": digest(run/"SUMMARY.json"),
        "artifact_sha256": {name: digest(run/name) for name in ("CANDIDATES_JOINED.jsonl", "IMAGES.jsonl", "GT_ASSOCIATIONS.jsonl", "SOURCE_LOCK.json")}})
    print(json.dumps({"status": summary["status"], "overall": summary["overall"], "join_reliability_passed": summary["join_reliability_passed"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        analyze(args.source_run, args.protocol, args.run)
    except Exception as error:
        if args.run.exists():
            write_json(args.run/"ANALYSIS_FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc(), "failed_at": now()})
        raise


if __name__ == "__main__":
    main()
