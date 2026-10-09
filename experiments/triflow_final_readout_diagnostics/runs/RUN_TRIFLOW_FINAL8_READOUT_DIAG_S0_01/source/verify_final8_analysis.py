"""Independent CPU reconstruction of final8 descriptive joins and every aggregate."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import traceback

VERSION = "triflow_final8_analysis_verify_v1"
TRUST = ("[0,.25)", "[.25,.5)", "[.5,.75)", "[.75,.999999)", "[.999999,1]")
ROOT = ("boundary_zero", "none", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1]")
THRESHOLD = struct.unpack("f", struct.pack("f", .999999))[0]
DIAGNOSTICS = ("trust_scale", "trust_saturated", "boundary_valid_count", "root_count", "neighbor_transition_count", "active_anchor_count")
IDENTITY = ("image_id", "detection_index", "category_id", "model_class", "bbox", "score", "box_xyxy", "raw_input_box_xyxy", "raw_confidence")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def lines(path):
    with Path(path).open(encoding="utf-8") as f:
        for s in f:
            yield json.loads(s)


def require(condition, text):
    if not condition:
        raise ValueError(text)


def compare(actual, expected, path="root"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), path+": dictionary keys differ")
        for key in expected:
            compare(actual[key], expected[key], path+"."+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), path+": list shape differs")
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, path+"["+str(i)+"]")
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11), path+": numeric aggregate differs")
    else:
        require(type(actual) is type(expected) and actual == expected, path+": scalar differs")


def accumulator():
    return {"candidate_count": 0, "matched_count": 0, "baseline_success_count": 0, "baseline_failure_count": 0,
        "damage_count": 0, "repair_count": 0, "before_iou_sum": 0., "after_iou_sum": 0., "delta_iou_sum": 0., "baseline_success_delta_sum": 0.,
        "image_ids": set(), "matched_image_ids": set(), "gt_ids": set()}


def accumulate(a, row):
    a["candidate_count"] += 1
    a["image_ids"].add(row["image_id"])
    p = row["paired"]
    if p is None:
        return
    a["matched_count"] += 1
    a["matched_image_ids"].add(row["image_id"])
    a["gt_ids"].add((row["image_id"], p["annotation_id"]))
    for key, source in (("before_iou_sum", "baseline_mask_iou"), ("after_iou_sum", "method_mask_iou"), ("delta_iou_sum", "delta_mask_iou")):
        a[key] += p[source]
    a["baseline_success_count"] += int(p["baseline_success"])
    a["baseline_failure_count"] += int(not p["baseline_success"])
    a["damage_count"] += int(p["damage"])
    a["repair_count"] += int(p["repair"])
    a["baseline_success_delta_sum"] += p["delta_mask_iou"] if p["baseline_success"] else 0


def finish(a):
    def div(x, y):
        return x/y if y else None
    r = {key: value for key, value in a.items() if key not in ("image_ids", "matched_image_ids", "gt_ids")}
    r.update(images_with_candidates=len(a["image_ids"]), images_with_matched_candidates=len(a["matched_image_ids"]),
        unique_associated_gt_count=len(a["gt_ids"]), matched_fraction_of_outputs=div(a["matched_count"], a["candidate_count"]),
        before_iou_mean=div(a["before_iou_sum"], a["matched_count"]), after_iou_mean=div(a["after_iou_sum"], a["matched_count"]),
        delta_iou_mean=div(a["delta_iou_sum"], a["matched_count"]), baseline_success_delta_iou_mean=div(a["baseline_success_delta_sum"], a["baseline_success_count"]),
        damage_rate_of_baseline_success=div(a["damage_count"], a["baseline_success_count"]), repair_rate_of_baseline_failure=div(a["repair_count"], a["baseline_failure_count"]))
    return r


def classify(row):
    cohort, d, n = row["cohort"], row["diagnostics"], row["neighbor"]
    trust, root, sat = "not_refined", "not_refined", "not_refined"
    if cohort == "refined":
        value = d["trust_scale"]
        trust = TRUST[-1]
        for end, label in zip((.25, .5, .75, THRESHOLD), TRUST):
            if value < end:
                trust = label
                break
        boundary, count = d["boundary_valid_count"], d["root_count"]
        require(type(boundary) is int and type(count) is int and 0 <= count <= boundary, "actual roots must lie within valid-boundary count")
        root = "boundary_zero" if not boundary else "none" if not count else next(label for end, label in zip((.25, .5, .75, 1), ROOT[2:]) if count/boundary <= end)
        sat = "true" if d["trust_saturated"] else "false"
    presence = "some" if root in ROOT[2:] else root
    own = "unknown_identity" if cohort == "unknown_identity" else "unknown_unselected" if row["raw_index"] is None else "duplicate_selected_raw" if row["own_raw_multiplicity_selected"] > 1 else "unique_selected_raw"
    channel = "unknown" if cohort == "unknown_identity" else "not_refined"
    if cohort == "refined":
        root_on, anchor_on = d["root_count"] > 0, d["active_anchor_count"] > 0
        channel = "mixed" if root_on and anchor_on else "boundary_only" if root_on else "anchor_only" if anchor_on else "no_constraints"
    return {"cohort": cohort, "trust_bin": trust, "root_bin": root, "neighbor_valid_slots": str(n["valid_slots"]) if n["valid_slots"] is not None else "unknown",
        "neighbor_raw_type": n["raw_type"], "own_raw_type": own,
        "natural_channel": channel,
        "cross_trust_root_neighbor": "saturated="+sat+"|root="+presence+"|neighbors="+n["raw_type"]}


def verify(run, source):
    run, source = run.resolve(), source.resolve()
    summary = json.loads((run/"SUMMARY.json").read_text())
    lock = json.loads((run/"SOURCE_LOCK.json").read_text())
    complete = json.loads((run/"ANALYSIS_COMPLETE.json").read_text())
    require(lock["source_run"] == str(source) == summary["source_run"], "actual source Run identity differs")
    require(lock["all_before_after_sha256_equal"] is True, "analysis source immutability not established")
    require(sha(run/"SUMMARY.json") == complete["summary_sha256"] and sha(run/"SOURCE_LOCK.json") == summary["source_lock_sha256"], "summary/source-lock bytes not bound")
    require(sha(lock["protocol"]) == lock["protocol_sha256"] == summary["protocol_sha256"], "frozen protocol bytes differ")
    for name, evidence in lock["files"].items():
        require(evidence["sha256_before"] == evidence["sha256_after"] == sha(source/name), "immutable source digest differs: "+name)
    for name, value in complete["artifact_sha256"].items():
        require(sha(run/name) == value, "actual analysis artifact changed: "+name)
    require(sha(Path(__file__)) == lock["analysis_source_sha256"]["verify_final8_analysis.py"], "actual independent verifier source differs")
    for name, value in lock["analysis_source_sha256"].items():
        require(sha(run/"source"/name) == value, "actual archived analysis source differs")
    expected = json.loads((source/"SUMMARY.json").read_text())
    paired = json.loads((source/"paired_triflow_vs_baseline/SUMMARY.json").read_text())
    require(expected["paired"]["triflow"] == paired and paired["paired_readout_valid"] is True, "actual paired source receipt differs")
    expected_pair = {}
    for p in lines(source/"paired_triflow_vs_baseline/INSTANCES.jsonl"):
        key = p["image_id"], p["detection_index"]
        require(key not in expected_pair, "source matched key duplicated")
        expected_pair[key] = p
    for collection in (lock["baseline_image_files"], lock["method_image_files"]):
        require(len(collection) == 5000 and len({e["image_id"] for e in collection}) == 5000, "source image lock cardinality differs")
        for evidence in collection:
            require(sha(source/evidence["path"]) == evidence["sha256_before"] == evidence["sha256_after"], "actual baseline/method source file changed")
    chain = hashlib.sha256("".join(e["sha256_before"] for e in lock["baseline_image_files"]).encode()).hexdigest()
    replay = json.loads((source/"BASELINE_REPLAY_VERIFICATION.json").read_text())
    require(chain == lock["baseline_identity_chain_sha256"] == replay["baseline_image_cache_sha256_chain"], "original baseline identity chain differs")
    aggregate, groups, gt = accumulator(), defaultdict(lambda: defaultdict(accumulator)), {}
    for value in ("refined", "unsupported_roi", "unrefined", "unknown_identity"):
        groups["cohort"][value]
    for value in TRUST+("not_refined", "unknown"):
        groups["trust_bin"][value]
    for value in ROOT+("not_refined", "unknown"):
        groups["root_bin"][value]
    for value in ("no_constraints", "anchor_only", "boundary_only", "mixed", "not_refined", "unknown"):
        groups["natural_channel"][value]
    joined = iter(lines(run/"CANDIDATES_JOINED.jsonl"))
    actual_images = {row["image_id"]: row for row in lines(run/"IMAGES.jsonl")}
    require(len(actual_images) == 5000, "analysis image cardinality differs")
    cohorts, compiler_counts, reasons, invariants = Counter(), Counter(), Counter(), Counter()
    visited, matched_seen, with_damage, with_repair, with_matched = set(), 0, 0, 0, 0
    for compiler in lines(source/"COMPILER_IMAGES.jsonl"):
        iid = compiler["image_id"]
        require(iid not in visited, "source compiler image duplicated")
        visited.add(iid)
        base = json.loads((source/"baseline/images"/f"{iid:012d}.json").read_text())["detections"]
        method = json.loads((source/"triflow/images"/f"{iid:012d}.json").read_text())["detections"]
        mapping = {m["native_output_row"]: m for m in compiler["mappings"]}
        unsupported = {m["native_output_row"]: m for m in compiler["unsupported_roi"]}
        counts = Counter(m["raw_index"] for m in list(mapping.values())+list(unsupported.values()))
        diag = {}
        for chunk in compiler["chunks"]:
            for position, native in enumerate(chunk["native_output_rows"]):
                require(native not in diag, "diagnostic native-row key duplicated")
                diag[native] = {key: chunk["diagnostics"][key][position] for key in DIAGNOSTICS if key in chunk["diagnostics"]}
        image_aggregate, image_cohorts, image_diagnostics = accumulator(), Counter(), Counter()
        for index, detection in enumerate(base):
            row = next(joined)
            require(row["image_id"] == iid and row["detection_index"] == index == detection["detection_index"], "joined true detection-index cardinality/order differs")
            key = iid, index
            pair = expected_pair.pop(key, None)
            compare(row["paired"], pair, "original_fixed_paired_row")
            compare(row["baseline_identity"], {k: detection.get(k) for k in ("category_id", "model_class", "bbox", "box_xyxy", "score", "raw_confidence")}, "baseline_identity")
            native = compiler["eligible_output_rows"][index]
            require(row["native_output_row"] == native, "native row/postconf index explicit bridge differs")
            m, u = mapping.get(native), unsupported.get(native)
            raw = m["raw_index"] if m else u["raw_index"] if u else None
            require(row["coefficient_max_abs_change"] == (m["coefficient_max_abs_change"] if m else None), "actual coefficient change magnitude differs")
            if m and diag[native].get("root_count") == 0 and diag[native].get("active_anchor_count") == 0:
                invariants["natural_no_constraints_instances"] += 1
                invariants["natural_no_constraints_nonzero_coefficient_change"] += m["coefficient_max_abs_change"] != 0
                invariants["natural_no_constraints_matched"] += pair is not None
                invariants["natural_no_constraints_nonzero_paired_delta"] += bool(pair and pair["delta_mask_iou"] != 0)
            require(row["raw_index"] == raw and row["own_raw_multiplicity_selected"] == (counts[raw] if raw is not None else None), "actual selected raw identity/multiplicity differs")
            require(row["identity_verified"] is (not row["unknown_reasons"]), "unknown identities not transparently marked")
            if row["identity_verified"]:
                require(len(base) == len(method) == len(compiler["eligible_output_rows"]) and all(detection.get(k) == method[index].get(k) for k in IDENTITY), "baseline/method complete identity parity differs")
                require(row["cohort"] == ("refined" if m else "unsupported_roi" if u else "unrefined"), "actual executed/unsupported/unselected cohort differs")
                if m:
                    require(m["post_conf_detection_index"] == index, "mapping explicit detection index differs")
                    compare(row["diagnostics"], diag[native], "actual_chunk_diagnostics")
                    chosen = [r for r, valid in zip(m["predicted_neighbor_raw_indices"], m["neighbor_valid"]) if valid]
                    neighbor_kind = "no_neighbor" if len(chosen) == 0 else "one_neighbor" if len(chosen) == 1 else "duplicate" if len(set(chosen)) == 1 else "distinct"
                    compare(row["neighbor"], {"valid_slots": len(chosen), "raw_type": neighbor_kind, "raw_indices": m["predicted_neighbor_raw_indices"], "output_rows": m["predicted_neighbor_output_rows"]}, "actual_neighbors")
                else:
                    require(row["diagnostics"] == {}, "unexecuted candidate has fabricated diagnostic values")
                    if pair:
                        require(pair["delta_mask_iou"] == 0 and pair["damage"] is False and pair["repair"] is False, "unexecuted paired candidate changed")
            else:
                require(row["cohort"] == "unknown_identity" and row["diagnostics"] == {}, "unreliable identity was assigned mechanistic strata")
            reasons.update(row["unknown_reasons"])
            compare(row["strata"], classify(row), "independent_stratum_classification")
            if pair:
                matched_seen += 1
                require(pair["baseline_score"] == detection["raw_confidence"] and pair["category_id"] == detection["category_id"], "paired joins guessed score/class identity")
                b, a = pair["baseline_mask_iou"], pair["method_mask_iou"]
                require(0 <= b <= 1 and 0 <= a <= 1 and math.isclose(a-b, pair["delta_mask_iou"], abs_tol=1e-14), "actual paired IoU/delta differs")
                require(pair["baseline_success"] is (b >= .75) and pair["damage"] is (b >= .75 and a < .75) and pair["repair"] is (b < .75 and a >= .75), "candidate success/damage/repair labels differ")
                gkey = iid, pair["annotation_id"]
                entry = gt.setdefault(gkey, {"image_id": iid, "annotation_id": pair["annotation_id"], "category_id": pair["category_id"], "associated_candidate_count": 0,
                    "baseline_success_candidate_count": 0, "baseline_failure_candidate_count": 0, "damaged_candidate_count": 0, "repaired_candidate_count": 0,
                    "cohort_counts": Counter(), "detection_indices": [], "known_raw_indices": []})
                entry["associated_candidate_count"] += 1
                entry["baseline_success_candidate_count"] += int(pair["baseline_success"])
                entry["baseline_failure_candidate_count"] += int(not pair["baseline_success"])
                entry["damaged_candidate_count"] += int(pair["damage"])
                entry["repaired_candidate_count"] += int(pair["repair"])
                entry["cohort_counts"][row["cohort"]] += 1
                entry["detection_indices"].append(index)
                if raw is not None:
                    entry["known_raw_indices"].append(raw)
            accumulate(aggregate, row); accumulate(image_aggregate, row)
            cohorts[row["cohort"]] += 1; image_cohorts[row["cohort"]] += 1
            for family, label in row["strata"].items():
                accumulate(groups[family][label], row)
            if row["cohort"] == "refined":
                for field in ("boundary_valid_count", "root_count", "neighbor_transition_count", "active_anchor_count"):
                    compiler_counts[field] += row["diagnostics"][field]; image_diagnostics[field] += row["diagnostics"][field]
                compiler_counts["trust_saturated_instances"] += row["diagnostics"]["trust_saturated"]
                image_diagnostics["trust_saturated_instances"] += row["diagnostics"]["trust_saturated"]
        expected_image = {"image_id": iid, **finish(image_aggregate), "cohort_counts": dict(image_cohorts),
            "cohort_fractions_of_image_outputs": {k: image_cohorts[k]/len(base) if base else None for k in ("refined", "unsupported_roi", "unrefined", "unknown_identity")},
            "compiler_counts": dict(image_diagnostics), "has_damage": image_aggregate["damage_count"] > 0, "has_repair": image_aggregate["repair_count"] > 0}
        compare(actual_images[iid], expected_image, "image_aggregate")
        with_damage += expected_image["has_damage"]; with_repair += expected_image["has_repair"]; with_matched += image_aggregate["matched_count"] > 0
    require(next(joined, None) is None and not expected_pair, "joined candidate/matched cardinality not complete")
    require(visited == set(expected["configuration"]["image_ids"]) == set(actual_images), "all supplied5000 image identities differ")
    compare(summary["overall"], finish(aggregate), "overall")
    require(aggregate["candidate_count"] == replay["detections"], "all baseline output source count differs")
    compare(summary["cohort_counts"], {k: cohorts[k] for k in ("refined", "unsupported_roi", "unrefined", "unknown_identity")}, "cohorts")
    rebuilt = {family: {label: dict(finish(a), fraction_supplied_5000_images_with_candidates=len(a["image_ids"])/5000,
        fraction_supplied_5000_images_with_matched_candidates=len(a["matched_image_ids"])/5000) for label, a in sorted(values.items())} for family, values in sorted(groups.items())}
    compare(summary["strata"], rebuilt, "all_stratum_counts_denominators_means_and_rates")
    compare(summary["compiler_counts_reconstructed"], dict(compiler_counts), "actual_compiler_counts")
    compare(summary["natural_channel_invariants"], {k: invariants[k] for k in ("natural_no_constraints_instances", "natural_no_constraints_nonzero_coefficient_change", "natural_no_constraints_matched", "natural_no_constraints_nonzero_paired_delta")}, "natural_channel_actual_zero_update_invariants")
    require(invariants["natural_no_constraints_nonzero_coefficient_change"] == 0 and invariants["natural_no_constraints_nonzero_paired_delta"] == 0, "natural no-constraint update invariant actually failed")
    for k, v in compiler_counts.items():
        require(v == expected["compiler_counts"][k], "reconstructed compiler source total differs: "+k)
    require(cohorts["refined"] == expected["refined_instances"] and cohorts["unsupported_roi"] == expected["unsupported_roi_instances"]
        and cohorts["refined"]+cohorts["unsupported_roi"] == expected["selected_instances"], "actual native cohort source totals differ")
    stats = paired["statistics"]
    compare(summary["source_expected"], {"paired_statistics": stats, "unique_associated_gt_count": paired["unique_matched_gt_instances"],
        "selected_instances": expected["selected_instances"], "refined_instances": expected["refined_instances"],
        "unsupported_roi_instances": expected["unsupported_roi_instances"], "compiler_counts": expected["compiler_counts"]}, "actual_locked_source_summary")
    count_pairs = (("matched_count", "matched_detection_count"), ("baseline_success_count", "baseline_success_count"), ("baseline_failure_count", "baseline_failure_count"), ("damage_count", "damage_count"), ("repair_count", "repair_count"))
    for actual, target in count_pairs:
        require(aggregate[actual] == stats[target], "actual paired source count differs: "+target)
    for actual, target in (("delta_iou_mean", "mean_mask_iou_delta"), ("before_iou_mean", "baseline_mean_mask_iou"), ("after_iou_mean", "method_mean_mask_iou"),
                          ("damage_rate_of_baseline_success", "damage_rate_of_baseline_success"), ("repair_rate_of_baseline_failure", "repair_rate_of_baseline_failure"), ("baseline_success_delta_iou_mean", "baseline_success_mean_iou_delta")):
        compare(summary["overall"][actual], stats[target]["value"], "actual_source_mean_or_rate."+target)
    require(len(gt) == paired["unique_matched_gt_instances"], "unique associatedGT cardinality differs")
    labels = Counter()
    actual_gt = iter(lines(run/"GT_ASSOCIATIONS.jsonl"))
    for key in sorted(gt):
        record = gt[key]
        record["cohort_counts"] = dict(record["cohort_counts"])
        record["known_raw_unique_count"] = len(set(record["known_raw_indices"]))
        record["known_raw_repeated_associations"] = len(record["known_raw_indices"])-record["known_raw_unique_count"]
        record["duplicate_candidate_association"] = record["associated_candidate_count"] > 1
        record["any_damage_label"] = record["damaged_candidate_count"] > 0
        record["all_associations_damage_label"] = record["damaged_candidate_count"] == record["associated_candidate_count"]
        record["any_repair_label"] = record["repaired_candidate_count"] > 0
        record["all_associations_repair_label"] = record["repaired_candidate_count"] == record["associated_candidate_count"]
        for k in ("duplicate_candidate_association", "any_damage_label", "all_associations_damage_label", "any_repair_label", "all_associations_repair_label"):
            labels[k] += record[k]
        compare(next(actual_gt), record, "unique_GT_candidate_association_record")
    require(next(actual_gt, None) is None, "associatedGT rows duplicated")
    compare(summary["gt_association_labels"], {"associated_gt_count": len(gt), **dict(labels), "unit": "labels among candidate-associated GT only; not GT success/damage/repair rates"}, "GT_labels_not_GT_rates")
    compare(summary["image_labels"], {"images": len(visited), "with_matched_candidates": with_matched, "with_damage": with_damage, "with_repair": with_repair}, "image_labels")
    compare(summary["unknown_reasons_candidate_occurrences"], dict(reasons), "unknown_reasons")
    compare(summary["bins"], {"trust": list(TRUST), "root_fraction": list(ROOT), "threshold_selection": "frozen descriptive bins; no threshold tuning",
        "nominal_saturation_boundary": .999999, "effective_saturation_boundary_fp32": THRESHOLD,
        "saturation_group_authority": "recorded source trust_saturated bool; trust bins use original FP32 boundary representation"}, "frozen_descriptive_bins_and_fp32_boundary")
    require(summary["join_reliability_passed"] is (not reasons) and summary["inference_executed"] is False and summary["gt_rematched"] is False, "read-only scope/reliability differs")
    for name, evidence in lock["files"].items():
        require(sha(source/name) == evidence["sha256_before"], "source bytes changed during independent verification")
    return {"audit_version": VERSION, "status": "passed", "passed": True, "run": str(run), "source_run": str(source),
        "candidate_rows": aggregate["candidate_count"], "matched_rows": matched_seen, "images": len(visited), "unique_associated_gt": len(gt),
        "source_compiler_counts": dict(compiler_counts), "source_paired_counts": {target: stats[target] for _, target in count_pairs},
        "all_summary_strata_reaggregated": True, "actual_join_identities_verified": not reasons, "source_bytes_unchanged": True,
        "summary_sha256": sha(run/"SUMMARY.json"), "joined_sha256": sha(run/"CANDIDATES_JOINED.jsonl"),
        "scope": "CPU source identities and descriptive aggregation; no inference, GT rematching, causal or AP-bin claim",
        "verified_at": datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        report = verify(args.run, args.source_run)
    except Exception as error:
        report = {"audit_version": VERSION, "status": "failed", "passed": False, "error": repr(error), "traceback": traceback.format_exc()}
    target = args.out or args.run/"VERIFICATION.json"
    with target.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
