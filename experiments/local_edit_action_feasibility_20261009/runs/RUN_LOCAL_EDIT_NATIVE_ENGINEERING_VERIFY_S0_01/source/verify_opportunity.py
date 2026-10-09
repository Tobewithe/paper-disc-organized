"""Independent CPU reaggregation of stored F receipts, masks and pixel ledgers.

Uses only the standard library and NumPy. It does not import an inference,
action-core or evaluator module, rerun a model, decode GT, or redo matching.
Native input/support assertions are checked as recorded receipt coverage;
stored original RLE, sparse edits and all finite-oracle arithmetic are checked
independently. Register a separate verification Run before invoking this CLI.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

VERSION = "local_edit_independent_verify_v1"
SEED, BOOTSTRAPS, GATE = 20261009, 5000, .002
POOL_SHA = "ef3453e65b80a2d07997c882b0624a88f6d8161fcd32e2d629cdff3285ba3c63"
CAL_SHA = "1b75dfea2ca011dc43aaa8a7b863900c3bae913472b849f76934361366d9d8ce"
WEIGHT_SHA = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
GROUPS = ("G_old", "G_eq", "G_eq_plus_L")
GLOBAL_IDS = ("zero", "smooth", "global_025", "global_05", "global_075", "global_10", "global_expand_025")
BINDING_KEYS = ("image_id", "detection_index", "native_output_row", "raw_index", "baseline_input_sha256",
                "hidden_query_sha256", "coefficient_sha256")
PIXEL_KEYS = ("pixel_TP", "pixel_FP", "pixel_FN", "mask_iou", "coverage", "purity")
CLUSTER_COLUMNS = ("labelable", "local_delta_sum", "direction_delta_sum", "baseline_success", "baseline_failure",
                   "old_iou_sum", "equal_iou_sum", "local_iou_sum", "old_repair", "equal_repair", "local_repair", "local_additional_repair")
SOURCE_KEYS = {"run_opportunity", "native_opportunity", "local_actions", "frozen_io", "native_mask_adapter", "protocol",
               "vendor_ops", "vendor_nms", "vendor_augment", "vendor_segment_validator", "vendor_head",
               "mask_calibration", "local_features", "risk_calibration", "portable_risk"}
CHECK_KEYS = ("native_input_parity_candidates", "native_original_parity_candidates", "identity_bound_candidates",
              "zero_exact_actions", "support_protected_actions", "single_region_actions", "dtype_checked_actions",
              "empty_actions_retained", "cross_identity_rejection_candidates", "changed_baseline_rejection_candidates",
              "forced_empty_export_probe_candidates")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((str(array.dtype), tuple(array.shape))).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def equal(actual, expected, label):
    """Recursive comparison: exact topology/booleans and small float tolerance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label+": dict keys differ")
        for key in expected:
            equal(actual[key], expected[key], label+"."+key)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label+": list shape differs")
        for index, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, label+f"[{index}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (float, int)) and not isinstance(actual, bool) and math.isfinite(actual) and
                math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label+": numeric value differs")
    else:
        require(actual == expected and (not isinstance(expected, bool) or isinstance(actual, bool)), label+": value differs")


def integer(value, label, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum, label+": invalid integer")
    return value


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n")


def jsonlines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for number, text in enumerate(handle, 1):
            require(bool(text.strip()), f"{path}:{number}: empty record")
            yield json.loads(text)


def candidate_key(record):
    return record["image_id"], record["detection_index"]


def binding(record):
    return {key: record[key] for key in BINDING_KEYS}


def locked_document(item, label):
    require(Path(item["path"]).is_file() and sha(item["path"]) == item["sha256"], label+": current bytes differ")
    return read(item["path"])


def completed_run(run, mode):
    meta, summary, complete = (read(run/name) for name in ("run.json", "SUMMARY.json", "COMPLETE.json"))
    require(meta.get("status") == "completed" and meta.get("return_code") == 0 and
            meta.get("artifact_completeness") == "complete" and meta.get("missing_outputs") == [], "Source Run not exit0/complete")
    require(summary.get("passed") is True and summary.get("mode") == mode and
            summary.get("status") == mode+"_complete", "Source summary mode/completion differs")
    require(complete.get("status") == summary["status"] and complete.get("summary_sha256") == sha(run/"SUMMARY.json"), "Completion receipt differs")
    require(meta.get("metrics_source", {}).get("sha256") == sha(run/"SUMMARY.json"), "Runner metrics-source SHA differs")
    require(not (run/"FAILURE.json").exists(), "Source Run retains a failure receipt")
    for artifact in meta.get("artifacts", []):
        require(artifact.get("exists") and Path(artifact["path"]).is_file() and sha(artifact["path"]) == artifact["sha256"], "Runner final artifact bytes differ")
    frozen = summary.get("frozen_integrity", {})
    require(frozen.get("passed") is True and frozen.get("initial_state_sha256") == frozen.get("final_state_sha256"), "Frozen-model receipt differs")
    for key in ("all_state_exact", "all_params_requires_grad_false", "all_bn_eval", "all_gradients_none"):
        require(frozen.get(key) is True, "Frozen receipt assertion absent: "+key)
    require(frozen.get("base_weights_sha256") == WEIGHT_SHA and summary.get("zero_native_input_original_parity_passed") is True,
            "Recorded weight/native zero assertion differs")
    require(summary.get("source_unchanged") is True and summary.get("GT_used_for_action_generation") is False, "Recorded source/action-GT invariant differs")
    return meta, summary


def provenance(run, meta, summary):
    inputs = read(run/"EXECUTION_INPUTS.json")
    configuration = inputs["configuration"]
    require(inputs["configuration_sha256"] == canonical_sha(configuration) == summary["configuration_sha256"], "Configuration SHA differs")
    declared, source = read(run/"DECLARED_SOURCE_LOCK.json"), read(run/"SOURCE_LOCK.json")
    require(set(source["files"]) == SOURCE_KEYS and source["files"] == declared["files"], "Declared/archive execution source set differs")
    require(sha(run/"DECLARED_SOURCE_LOCK.json") == configuration["declared_source_lock_sha256"], "Declared source-lock SHA differs")
    require(sha(run/"SOURCE_LOCK.json") == summary["source_lock_sha256"], "Source-lock receipt SHA differs")
    require(configuration["source_sha256"] == {key: item["sha256"] for key, item in source["files"].items()}, "Configuration execution-source map differs")
    original_inputs = {str(Path(item["path"]).resolve()): item for item in meta.get("inputs", [])}
    snapshots = {str(Path(item["path"]).resolve()): item for item in meta.get("snapshots", [])}
    watched = {}
    for key, item in source["files"].items():
        filename = key.removeprefix("vendor_")+".py" if key.startswith("vendor_") else "PROTOCOL.md" if key == "protocol" else key+".py"
        archived = run/"source"/filename
        require(archived.is_file() and sha(archived) == item["sha256"], "Archived execution source differs: "+key)
        original = str(Path(item["path"]).resolve())
        require(original_inputs.get(original, {}).get("sha256") == item["sha256"], "Runner initial source identity absent: "+key)
        capture = snapshots.get(original, {})
        require(capture.get("revision") == item["sha256"] and sha(run/capture["snapshot"]) == item["sha256"], "Runner before-source snapshot differs: "+key)
        require(sha(original) == item["sha256"], "Current source changed after F: "+key)
        watched[original] = item["sha256"]
        watched[str(archived.resolve())] = item["sha256"]
    panel = read(run/"PANEL.json")
    panel_sha = configuration["panel_sha256"]
    require(sha(run/"PANEL.json") == panel_sha, "Archived panel differs")
    command = meta.get("command", [])
    def command_value(flag):
        require(flag in command and command.index(flag)+1 < len(command), "Runner command missing "+flag)
        return command[command.index(flag)+1]
    equal(command_value("--panel-sha256"), panel_sha, "runner panel SHA")
    equal(command_value("--mode"), configuration["mode"], "runner mode")
    for flag, expected in (("--panel", panel_sha), ("--source-lock", configuration["declared_source_lock_sha256"])):
        original = str(Path(command_value(flag)).resolve())
        require(original_inputs.get(original, {}).get("sha256") == expected and sha(original) == expected, "Original panel/source lock differs: "+flag)
        snapshot = snapshots.get(original, {})
        require(snapshot.get("revision") == expected and sha(run/snapshot["snapshot"]) == expected, "Initial panel/source-lock snapshot differs")
        watched[original] = expected
    for filename in ("PANEL.json", "DECLARED_SOURCE_LOCK.json", "SOURCE_LOCK.json", "EXECUTION_INPUTS.json",
                     "SUMMARY.json", "COMPLETE.json", "run.json", "CANDIDATES.jsonl", "ACTIONS.jsonl", "IMAGES.jsonl", "GT_ASSOCIATIONS.jsonl"):
        watched[str((run/filename).resolve())] = sha(run/filename)
    for filename in ("OPPORTUNITY_STATISTICS.json", "ACTION_CONTROLS.json"):
        if (run/filename).exists():
            watched[str((run/filename).resolve())] = sha(run/filename)
    for path in (run/"sparse").glob("*.npz"):
        watched[str(path.resolve())] = sha(path)
    return panel, configuration, watched


def panel_replay(panel, watched):
    require(panel.get("schema_version") == "local_edit_opportunity_panel_v1" and panel.get("split") == "train2017", "Panel schema/split differs")
    images = panel["images"]
    ids = [item["image_id"] for item in images]
    require(len(ids) == len(set(ids)) == 512 and ids == sorted(ids), "Fixed panel must contain sorted unique512 IDs")
    equal(panel["engineering_image_ids"], ids[:4], "engineering first4")
    population, exclusion = panel["population"], panel["calibration_exclusion"]
    equal(population["source"]["sha256"], POOL_SHA, "original20k pool SHA")
    equal(exclusion["source"]["sha256"], CAL_SHA, "original calibration split SHA")
    original = locked_document(population["source"], "original20k pool")
    eligible = locked_document(population, "eligible pool")
    calibration = locked_document(exclusion["source"], "original calibration split")
    excluded = locked_document(exclusion, "fixed calibration exclusion list")
    require(isinstance(original, list) and len(original) == len(set(original)) == 20000, "Original20k ID universe differs")
    fit, selection = calibration["fit_ids"], calibration["selection_ids"]
    require(len(fit) == len(set(fit)) == 1500 and len(selection) == len(set(selection)) == 500 and not set(fit)&set(selection), "Original fit/select uniqueness or sizes differ")
    equal(excluded, sorted(set(fit)|set(selection)), "original fit1500/select500 union")
    equal(sorted(exclusion["image_ids"]), excluded, "inline exclusion IDs")
    require(len(exclusion["image_ids"]) == len(set(exclusion["image_ids"])) == 2000, "Inline exclusions duplicate")
    equal(eligible, sorted(set(original)-set(excluded)), "eligible20k-minus-calibration")
    require(len(eligible) == 19638 and len(set(original)&set(fit)) == 277 and len(set(original)&set(selection)) == 85, "Eligible pool/intersection sizes differ")
    selection_definition = panel["selection"]
    require(selection_definition["seed"] == SEED and selection_definition["image_count"] == 512 and selection_definition["kind"] == "uniform_without_replacement", "Sampling declaration differs")
    require(selection_definition["population_count"] == len(eligible) and selection_definition["population_sha256"] == canonical_sha(eligible), "Eligible population fingerprint differs")
    replay = sorted(int(iid) for iid in np.random.default_rng(SEED).choice(np.asarray(eligible, dtype=np.int64), 512, replace=False))
    equal(ids, replay, "independent uniform512 replay")
    require(not set(ids)&set(excluded), "Panel overlaps independent calibration")
    for item in (population, population["source"], exclusion, exclusion["source"], panel["annotations"], panel["weights"]):
        require(Path(item["path"]).is_file() and sha(item["path"]) == item["sha256"], "Original input hash differs")
        watched[str(Path(item["path"]).resolve())] = item["sha256"]
    require(panel["weights"]["sha256"] == WEIGHT_SHA, "Official checkpoint SHA differs")
    for item in images:
        integer(item["image_id"], "panel image ID")
        require(Path(item["path"]).is_file() and sha(item["path"]) == item["sha256"], "Panel image bytes differ")
        watched[str(Path(item["path"]).resolve())] = item["sha256"]
    return {item["image_id"]: item for item in images}


def rle_intervals(rle, shape):
    """Decode official COCO compressed counts into foreground intervals only."""
    equal(rle["size"], shape, "RLE original size")
    encoded = rle["counts"]
    require(isinstance(encoded, str), "Stored RLE must use an ASCII counts string")
    counts, cursor = [], 0
    while cursor < len(encoded):
        value, shift, more = 0, 0, True
        while more:
            require(cursor < len(encoded), "Truncated COCO RLE count")
            code = ord(encoded[cursor])-48
            cursor += 1
            require(0 <= code <= 63, "Invalid COCO RLE ASCII code")
            value |= (code & 31) << shift
            more = bool(code & 32)
            shift += 5
            if not more and code & 16:
                value |= -1 << shift
        if len(counts) > 2:
            value += counts[-2]
        require(value >= 0, "Negative reconstructed COCO RLE count")
        counts.append(value)
    size = int(shape[0])*int(shape[1])
    require(sum(counts) == size, "COCO RLE counts do not cover the original image")
    start, foreground = 0, []
    for index, count in enumerate(counts):
        if index % 2 and count:
            foreground.append((start, start+count))
        start += count
    return foreground, sum(end-start for start, end in foreground)


def intersection_size(left, right):
    first = second = total = 0
    while first < len(left) and second < len(right):
        a, b = left[first], right[second]
        total += max(0, min(a[1], b[1])-max(a[0], b[0]))
        if a[1] < b[1]:
            first += 1
        else:
            second += 1
    return total


def zero_rle_sha(intervals, shape):
    flat = np.zeros(int(shape[0])*int(shape[1]), dtype=np.bool_)
    for start, end in intervals:
        flat[start:end] = True
    return array_sha(flat.reshape(shape, order="F"))


def sparse_connected(indices, width=640):
    unseen = set(int(index) for index in indices)
    pending = [unseen.pop()]
    while pending:
        index = pending.pop()
        row, column = divmod(index, width)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0 or not (0 <= row+dr < 640 and 0 <= column+dc < 640):
                    continue
                neighbor = (row+dr)*width+column+dc
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    pending.append(neighbor)
    return not unseen


def check_pixels(record, labelable, gt_area, original_area=None):
    if not labelable:
        require(all(record.get(key) is None for key in PIXEL_KEYS), "Unknown pixel labels were converted to a measured value")
        return None
    tp, fp, fn = (integer(record[key], key) for key in PIXEL_KEYS[:3])
    require(tp+fn == gt_area and gt_area > 0, "Pixel TP+FN differs from fixed original GT area")
    if original_area is not None:
        require(tp+fp == original_area, "Pixel TP+FP differs from actual original RLE area")
    expected = {"mask_iou": tp/(tp+fp+fn), "coverage": tp/(tp+fn), "purity": tp/(tp+fp) if tp+fp else None}
    for key, value in expected.items():
        equal(record[key], value, "pixel-derived "+key)
    return expected["mask_iou"]


def independent_bootstrap(clusters):
    matrix = np.asarray(clusters, dtype=np.float64)
    totals = matrix.sum(axis=0)
    rng = np.random.default_rng(SEED)
    resampled = np.empty((BOOTSTRAPS, matrix.shape[1]), dtype=np.float64)
    # Different batching from inference evaluator, identical fixed RNG stream.
    for offset in range(0, BOOTSTRAPS, 73):
        count = min(73, BOOTSTRAPS-offset)
        selected = rng.integers(0, len(matrix), size=(count, len(matrix)))
        resampled[offset:offset+count] = matrix[selected].sum(axis=1)
    result = {"bootstrap_samples": BOOTSTRAPS, "bootstrap_seed": SEED,
              "bootstrap_unit": "paired supplied image clusters; ratio of candidate sums; no-match images retained",
              "images": len(matrix), "images_with_labelable_candidates": int(np.count_nonzero(matrix[:, 0])),
              "labelable_matched_candidates": int(totals[0]), "baseline_success_count": int(totals[3]), "baseline_failure_count": int(totals[4])}
    ratios = {"local_marginal_mask_iou": (1, 0), "equal_direction_vs_old_mask_iou": (2, 0), "old_oracle_mean_mask_iou": (5, 0),
              "equal_oracle_mean_mask_iou": (6, 0), "local_oracle_mean_mask_iou": (7, 0), "old_oracle_repair_rate": (8, 4),
              "equal_oracle_repair_rate": (9, 4), "local_oracle_repair_rate": (10, 4), "local_additional_repair_rate": (11, 4)}
    for name, (numerator, denominator) in ratios.items():
        valid = resampled[:, denominator] > 0
        distribution = resampled[valid, numerator]/resampled[valid, denominator]
        result[name] = {"value": float(totals[numerator]/totals[denominator]) if totals[denominator] else None,
                        "ci95": np.quantile(distribution, [.025, .975]).tolist() if len(distribution) else None,
                        "valid_bootstrap_draws": int(valid.sum())}
    return result


def independent_gate(statistics):
    limits = statistics["local_marginal_mask_iou"]["ci95"]
    if limits is None:
        decision = "insufficient_evidence"
    elif limits[0] >= GATE:
        decision = "invest_in_separate_method_comparison"
    elif limits[1] < GATE:
        decision = "stop_current_configuration"
    else:
        decision = "insufficient_evidence"
    return {"decision": decision, "threshold": GATE, "metric": "best(G_eq+L)-best(G_eq), matched candidate mean",
            "rule": "lower95CI>=.002 invest; upper95CI<.002 stop; otherwise insufficient",
            "meaning": "artificial finite-action route investment gate; not AP prediction, deployment evidence, or method upper bound"}


def three_value_any(values):
    return True if True in values else None if None in values else False


def three_value_all(values):
    return False if False in values else None if None in values else True


def verify_records(source, output, panel_images, config, summary, mode, watched):
    image_records = list(jsonlines(source/"IMAGES.jsonl"))
    ids = config["image_ids"]
    equal([image["image_id"] for image in image_records], ids, "full ordered image cluster scope")
    equal(ids, sorted(panel_images) if mode == "opportunity" else sorted(panel_images)[:4], "fixed mode image scope")
    image_map = {image["image_id"]: image for image in image_records}
    require(len(image_map) == len(ids), "Duplicate image record")
    clusters = {iid: np.zeros(12, dtype=np.float64) for iid in ids}
    counts = {iid: defaultdict(int) for iid in ids}
    checks = defaultdict(int, {key: 0 for key in CHECK_KEYS})
    gt_groups, controls = {}, {}
    sparse_cache, sparse_used, candidate_sequence = {}, defaultdict(set), []
    action_groups = iter(itertools.groupby(jsonlines(source/"ACTIONS.jsonl"), candidate_key))
    association_rows = iter(jsonlines(source/"GT_ASSOCIATIONS.jsonl"))
    unknown_reasons, original_change_sums, truncation = defaultdict(int), defaultdict(int), defaultdict(int)
    with (output/"REAGGREGATED_CANDIDATES.jsonl").open("w", encoding="utf-8") as candidate_output:
        for candidate in jsonlines(source/"CANDIDATES.jsonl"):
            key = candidate_key(candidate)
            iid, detection_index = key
            require(iid in image_map and detection_index == counts[iid]["candidates"], "Candidate scope/order/detection index is incomplete")
            counts[iid]["candidates"] += 1
            candidate_sequence.append(key)
            integer(candidate["native_output_row"], "native row")
            integer(candidate["raw_index"], "raw index")
            require(candidate["native_output_row"] < 300 and candidate["raw_confidence"] > .001,
                    "Candidate is outside declared normal native filter scope")
            previous = counts[iid].get("previous_native_row", -1)
            require(candidate["native_output_row"] > previous, "Native post-conf row order changed")
            counts[iid]["previous_native_row"] = candidate["native_output_row"]
            equal(candidate["source_image_sha256"], panel_images[iid]["sha256"], "candidate image identity")
            equal(candidate["input_shape"], [640, 640], "candidate input grid")
            for digest_key in ("baseline_input_sha256", "baseline_original_sha256", "hidden_query_sha256", "coefficient_sha256", "input_sha256"):
                require(isinstance(candidate[digest_key], str) and len(candidate[digest_key]) == 64, "Missing candidate digest binding")
            group_key, grouped = next(action_groups, (None, None))
            equal(group_key, key, "ACTION candidate identity/order")
            actions = list(grouped)
            action_ids = [action["action_id"] for action in actions]
            require(7 <= len(actions) <= 23 and len(set(action_ids)) == len(action_ids), "Finite action count/unique IDs differ")
            equal(action_ids[:7], GLOBAL_IDS, "fixed global+zero order")
            equal(action_ids, candidate["action_ids"], "candidate action IDs")
            equal(candidate["action_count"], len(actions), "candidate action count")
            assoc = candidate["association"]
            equal(binding(assoc), binding(candidate), "GT fixed candidate binding")
            labelable = assoc["labelable"]
            require(isinstance(labelable, bool), "GT labelable flag invalid")
            if mode == "engineering":
                require(assoc["matched"] is None and labelable is False and assoc["annotation_id"] is None and assoc["unknown_reason"] == "engineering_no_GT", "Engineering parsed/measured GT labels")
            else:
                require(isinstance(assoc["matched"], bool), "Formal matched state is unknown")
                if assoc["matched"]:
                    counts[iid]["matched"] += 1
                    integer(assoc["annotation_id"], "GT annotation ID", 1)
                    require(assoc["box_iou"] >= .5 and assoc["gt_iscrowd"] == 0 and assoc["category_id"] == candidate["category_id"], "Stored fixed GT association eligibility differs")
                else:
                    require(assoc["annotation_id"] is None and not labelable, "Unmatched detection has a GT label")
            if labelable:
                require(assoc["matched"] is True and assoc["unknown_reason"] is None, "Labelable state lacks a reliable stored association")
                counts[iid]["labelable"] += 1
                gt_area = integer(assoc["gt_original_pixel_area"], "GT original pixel area", 1)
            else:
                gt_area = None
                require(assoc["baseline_success"] is None and bool(assoc["unknown_reason"]), "Unknown association lost its reason")
                unknown_reasons[assoc["unknown_reason"]] += 1
            baseline_iou = check_pixels(assoc, labelable, gt_area, candidate["baseline_original_area"] if labelable else None)
            baseline_success = None if baseline_iou is None else baseline_iou >= .75
            equal(assoc["baseline_success"], baseline_success, "baseline success threshold")
            best = {group: None for group in GROUPS}
            base_intervals, base_area = rle_intervals(actions[0]["segmentation"], candidate["original_shape"])
            require(zero_rle_sha(base_intervals, candidate["original_shape"]) == candidate["baseline_original_sha256"], "Stored original zero RLE differs from bound original baseline SHA")
            equal(base_area, candidate["baseline_original_area"], "zero original area")
            equal(candidate["baseline_empty"], base_area == 0, "baseline empty flag")
            sign_actions = {"remove": [], "add": []}
            empties = 0
            for action_position, action in enumerate(actions):
                equal(binding(action), binding(candidate), "action immutable candidate binding")
                equal(action["annotation_id"], assoc["annotation_id"], "action fixed GT identity")
                equal(action["labelable"], labelable, "action label availability")
                equal(action["baseline_success"], baseline_success, "action baseline success")
                intervals, area = rle_intervals(action["segmentation"], candidate["original_shape"])
                changed = area+base_area-2*intersection_size(intervals, base_intervals)
                equal(area, action["original_area"], "independent RLE area")
                equal(changed, action["original_changed_area"], "independent original symmetric change area")
                equal(action["original_changed_image_fraction"], changed/(candidate["original_shape"][0]*candidate["original_shape"][1]), "original changed image fraction")
                equal(action["original_changed_baseline_area_ratio"], changed/base_area if base_area else None, "original changed baseline area ratio")
                equal(action["original_empty"], area == 0, "original empty action")
                equal(action["input_empty"], action["input_area"] == 0, "input empty action")
                integer(action["input_area"], "input foreground area")
                integer(action["input_changed_area"], "input changed area")
                require(action["input_area"] <= 640*640, "Input area exceeds grid")
                for seconds in action["timing"].values():
                    require(isinstance(seconds, (float, int)) and math.isfinite(seconds) and seconds >= 0, "Invalid actual action timing")
                if action_position < 6:
                    equal(action["available_in"], GROUPS, "G_old membership")
                elif action_position == 6:
                    equal(action["available_in"], GROUPS[1:], "equal global expansion membership")
                else:
                    equal(action["available_in"], [GROUPS[2]], "local-only membership")
                if action_position == 0:
                    require(action["family"] == "zero" and action["tau"] == 0 and action["input_changed_area"] == changed == 0, "Zero action altered its baseline")
                    sparse = action["sparse"]
                    equal(sparse["binding"], binding(candidate), "zero bound identity")
                    require(sparse["area"] == sparse["sign"] == 0 and sparse["bbox"] is None, "Zero sparse metadata differs")
                    require(sparse["indices_sha256"] == array_sha(np.empty(0, dtype=np.int64)), "Zero sparse digest differs")
                elif action_position < 7:
                    require(action["family"] == "global" and action["sparse"] is None, "Global action acquired sparse edit")
                    if action_position == 1:
                        area32 = np.float32(base_area)
                        normalized = np.float32(area32/np.float32(2304))
                        expected_tau = float(np.float32(np.float32(.75)/np.float32(np.float32(1)+np.float32(normalized*normalized))))
                        require(math.isclose(action["tau"], expected_tau, rel_tol=0, abs_tol=1e-7), "Smooth threshold differs from FP32 formula")
                    else:
                        equal(action["tau"], (.25, .5, .75, 1.0, -.25)[action_position-2], "global threshold")
                    if action["tau"] > 0:
                        require(area <= base_area and intersection_size(intervals, base_intervals) == area, "Positive global original action expanded baseline")
                    else:
                        require(area >= base_area and intersection_size(intervals, base_intervals) == base_area, "Negative global original action contracted baseline")
                else:
                    require(action["family"] == "local", "Finite extra action is not local")
                    sparse = action["sparse"]
                    equal(sparse["binding"], binding(candidate), "local immutable binding")
                    equal(sparse["shape"], [640, 640], "local sparse input shape")
                    operation = sparse["operation"]
                    require(operation in sign_actions and sparse["sign"] == (-1 if operation == "remove" else 1), "Local edit sign differs")
                    equal(action["tau"], .25 if operation == "remove" else -.25, "local generating threshold")
                    artifact = sparse["indices_artifact"]
                    relative = Path(artifact["path"])
                    require(relative == Path("sparse")/f"{iid:012d}.npz", "Sparse artifact moved across image")
                    if iid not in sparse_cache:
                        path = source/relative
                        require(watched.get(str(path.resolve())) == sha(path), "Sparse artifact bytes changed after initial hash")
                        sparse_cache[iid] = np.load(path, allow_pickle=False)
                    indices = sparse_cache[iid][artifact["key"]]
                    sparse_used[iid].add(artifact["key"])
                    require(indices.dtype == np.int64 and indices.ndim == 1 and indices.size > 0 and indices[0] >= 0 and indices[-1] < 640*640 and np.all(indices[1:] > indices[:-1]), "Sparse indices invalid/duplicated/unsorted")
                    require(array_sha(indices) == sparse["indices_sha256"], "Sparse indices digest differs")
                    rows, columns = np.divmod(indices, 640)
                    bbox = [int(rows.min()), int(columns.min()), int(rows.max())+1, int(columns.max())+1]
                    equal(sparse["bbox"], bbox, "sparse pixel bbox")
                    equal(sparse["area"], int(indices.size), "sparse edit area")
                    equal(action["input_changed_area"], int(indices.size), "single-region input change area")
                    require(sparse_connected(indices), "Sparse edit has more than one 8-connected region")
                    digest = hashlib.sha256(f"{operation}:640:640:".encode("ascii")+indices.astype("<i8", copy=False).tobytes()).hexdigest()
                    require(action["action_id"] == sparse["action_id"] == operation+"-"+digest, "Core sparse action ID differs")
                    sign_actions[operation].append({"area": int(indices.size), "bbox": bbox, "first_flat_index": int(indices[0])})
                    checks["single_region_actions"] += 1
                    counts[iid]["local_actions"] += 1
                iou = check_pixels(action, labelable, gt_area, area if labelable else None)
                repair = None if iou is None else not baseline_success and iou >= .75
                damage = None if iou is None else baseline_success and iou < .75
                equal(action["repair"], repair, "action repair threshold")
                equal(action["damage"], damage, "action damage threshold")
                if iou is not None:
                    for group in action["available_in"]:
                        if best[group] is None or iou > best[group]["mask_iou"]:
                            best[group] = {"action_id": action["action_id"], "mask_iou": iou}
                    control_name = action["action_id"] if action["family"] != "local" else "local_"+action["sparse"]["operation"]
                    control = controls.setdefault(control_name, {"labelable_action_count": 0, "baseline_success_action_count": 0,
                                                                "baseline_failure_action_count": 0, "damage_action_count": 0, "repair_action_count": 0})
                    for name, value in (("labelable_action_count", 1), ("baseline_success_action_count", int(baseline_success)),
                                        ("baseline_failure_action_count", int(not baseline_success)), ("damage_action_count", int(damage)), ("repair_action_count", int(repair))):
                        control[name] += value
                empties += int(area == 0)
                counts[iid]["actions"] += 1
                counts[iid]["empty"] += int(area == 0)
                checks["support_protected_actions"] += 1
                checks["dtype_checked_actions"] += 1
                checks["empty_actions_retained"] += int(area == 0)
                original_change_sums[action["family"]] += changed
            equal(candidate["original_empty_actions"], empties, "candidate empty count")
            for operation, selected in sign_actions.items():
                inventory = candidate["components"][operation]
                before = integer(inventory["components_before_cap"], "pre-cap component count")
                equal(inventory["components_kept"], min(8, before), "top8 kept count")
                equal(inventory["components_truncated"], max(0, before-8), "top8 truncated count")
                equal(len(selected), inventory["components_kept"], "local action/component count")
                equal(selected, inventory["selected"], "independent sparse selected components")
                equal(selected, sorted(selected, key=lambda item: (-item["area"], item["bbox"][0], item["bbox"][1], item["first_flat_index"])), "stable selected component sorting")
                require(inventory["pixels_before_cap"] == sum(item["area"] for item in selected)+inventory["pixels_truncated"], "Truncated pixel accounting differs")
                truncation[operation+"_components"] += inventory["components_truncated"]
                truncation[operation+"_pixels"] += inventory["pixels_truncated"]
            if labelable:
                old, eq, local = (best[group]["mask_iou"] for group in GROUPS)
                require(local >= eq >= old >= baseline_iou, "Nested oracle/zero inclusion failed")
                local_delta, direction_delta = local-eq, eq-old
                opportunity = {"local_marginal_mask_iou": local_delta, "equal_direction_vs_old_mask_iou": direction_delta,
                               "best_actions": best, "zero_increment_retained": local_delta == 0,
                               "oracle_repair": {group: not baseline_success and best[group]["mask_iou"] >= .75 for group in GROUPS},
                               "oracle_damage": {group: baseline_success and best[group]["mask_iou"] < .75 for group in GROUPS}}
                clusters[iid] += [1, local_delta, direction_delta, int(baseline_success), int(not baseline_success), old, eq, local,
                                  int(not baseline_success and old >= .75), int(not baseline_success and eq >= .75),
                                  int(not baseline_success and local >= .75), int(not baseline_success and eq < .75 and local >= .75)]
                counts[iid]["zero_local_increment"] += int(local_delta == 0)
            else:
                opportunity = {"local_marginal_mask_iou": None, "equal_direction_vs_old_mask_iou": None, "best_actions": best,
                               "zero_increment_retained": None, "oracle_repair": None, "oracle_damage": None}
            equal(candidate["opportunity"], opportunity, "independently rebuilt finite-oracle opportunity")
            independent_association = next(association_rows, None)
            equal(independent_association, assoc | opportunity, "GT association supplement")
            if assoc["matched"] is True:
                gt_key = (iid, assoc["annotation_id"])
                group = gt_groups.setdefault(gt_key, {"image_id": iid, "annotation_id": assoc["annotation_id"], "detection_indices": [], "labels": defaultdict(list)})
                group["detection_indices"].append(detection_index)
                labels = {"baseline_success": baseline_success,
                          "local_margin_positive": opportunity["local_marginal_mask_iou"] > 0 if labelable else None,
                          "local_additional_repair": not baseline_success and best["G_eq"]["mask_iou"] < .75 and best["G_eq_plus_L"]["mask_iou"] >= .75 if labelable else None}
                for label, value in labels.items():
                    group["labels"][label].append(value)
                for finite_group in GROUPS:
                    group["labels"][finite_group+"_oracle_repair"].append(opportunity["oracle_repair"][finite_group] if labelable else None)
            for name in ("native_input_parity_candidates", "native_original_parity_candidates", "identity_bound_candidates", "zero_exact_actions"):
                checks[name] += 1
            if mode == "engineering":
                for name in ("cross_identity_rejection_candidates", "changed_baseline_rejection_candidates", "forced_empty_export_probe_candidates"):
                    checks[name] += 1
            line(candidate_output, binding(candidate) | {"annotation_id": assoc["annotation_id"], "labelable": labelable,
                                                        "baseline_success": baseline_success, "action_count": len(actions), "opportunity": opportunity})
    require(next(action_groups, None) is None and next(association_rows, None) is None, "Trailing actions/associations lack candidate identity")
    for iid, archived in sparse_cache.items():
        equal(set(archived.files), sparse_used[iid], "all sparse artifact keys used")
        archived.close()
    require({int(path.stem) for path in (source/"sparse").glob("*.npz")} == set(sparse_cache), "Unreferenced sparse artifact image exists")
    # Three recorded receipt checks are probes, not inferred from the masks.
    for key in ("cross_identity_rejection_candidates", "changed_baseline_rejection_candidates", "forced_empty_export_probe_candidates"):
        checks.setdefault(key, 0)
    equal(dict(checks), summary["engineering_checks"], "recorded engineering assertion coverage")
    image_checks = defaultdict(int)
    with (output/"REAGGREGATED_IMAGES.jsonl").open("w", encoding="utf-8") as image_output:
        for iid in ids:
            record, count = image_map[iid], counts[iid]
            require(count["candidates"] == record["native_candidate_count"] and count["candidates"] <= 300, "Native candidate count differs from per-image receipt")
            equal(record["source_image_sha256"], panel_images[iid]["sha256"], "image receipt hash")
            equal(record["action_count"], count["actions"], "image action count")
            equal(record["original_empty_actions"], count["empty"], "image empty count")
            for key, value in record["checks"].items():
                image_checks[key] += value
            expected_checks = {key: count["candidates"] for key in CHECK_KEYS[:4]}
            expected_checks.update({"support_protected_actions": count["actions"], "single_region_actions": count["local_actions"],
                                    "dtype_checked_actions": count["actions"], "empty_actions_retained": count["empty"],
                                    "cross_identity_rejection_candidates": count["candidates"] if mode == "engineering" else 0,
                                    "changed_baseline_rejection_candidates": count["candidates"] if mode == "engineering" else 0,
                                    "forced_empty_export_probe_candidates": count["candidates"] if mode == "engineering" else 0})
            equal(record["checks"], expected_checks, "image native assertion coverage")
            if mode == "opportunity":
                equal(record["matched_candidate_count"], count["matched"], "image matched count")
                equal(record["labelable_candidate_count"], count["labelable"], "image labelable count")
                equal(record["unmatched_candidate_count"], count["candidates"]-count["matched"], "image unknown scope")
                equal(record["cluster_columns"], CLUSTER_COLUMNS, "cluster column definition")
                equal(record["cluster_sums"], clusters[iid].tolist(), "independent image cluster sums")
            else:
                require(all(record[key] is None for key in ("matched_candidate_count", "labelable_candidate_count", "unmatched_candidate_count", "cluster_sums", "cluster_columns")), "Engineering image inferred GT labels")
            line(image_output, {"image_id": iid, "native_candidates": count["candidates"], "matched_candidates": count["matched"] if mode == "opportunity" else None,
                                "labelable_candidates": count["labelable"] if mode == "opportunity" else None,
                                "no_labelable_match": count["labelable"] == 0 if mode == "opportunity" else None,
                                "zero_local_increments": count["zero_local_increment"] if mode == "opportunity" else None,
                                "cluster_sums": clusters[iid].tolist() if mode == "opportunity" else None})
    equal(dict(image_checks), summary["engineering_checks"], "per-image check totals")
    total_candidates = sum(counts[iid]["candidates"] for iid in ids)
    total_actions = sum(counts[iid]["actions"] for iid in ids)
    total_matches = sum(counts[iid]["matched"] for iid in ids)
    total_labelable = sum(counts[iid]["labelable"] for iid in ids)
    equal(summary["image_count"], len(ids), "summary image count")
    equal(summary["native_candidates"], total_candidates, "summary native candidates")
    equal(summary["actions"], total_actions, "summary action count")
    equal(summary["empty_actions_preserved_count"], sum(counts[iid]["empty"] for iid in ids), "summary empty actions")
    statistics = independent_bootstrap([clusters[iid].tolist() for iid in ids]) if mode == "opportunity" else None
    if mode == "opportunity":
        for key, expected in (("matched_candidates", total_matches), ("labelable_matched_candidates", total_labelable),
                              ("unmatched_candidates", total_candidates-total_matches), ("matched_but_unlabelable_candidates", total_matches-total_labelable)):
            equal(summary[key], expected, "summary "+key)
        # Source summary counts only valid GT instances, rather than all matched
        # but invalid annotations; retain both sets explicitly in this verifier.
        valid_unique_gt = sum(any(value is not None for value in group["labels"]["baseline_success"]) for group in gt_groups.values())
        equal(summary["unique_matched_GT_instances"], valid_unique_gt, "summary valid unique GT instances")
        equal(summary["statistics"], statistics, "independent clustered bootstrap")
        equal(summary["investment_gate"], independent_gate(statistics), "independent .002 gate")
        equal(read(source/"OPPORTUNITY_STATISTICS.json"), statistics | {"investment_gate": independent_gate(statistics)}, "separate opportunity statistics")
        equal(read(source/"ACTION_CONTROLS.json")["controls"], controls, "independent individual action repair/damage controls")
        require(summary["GT_parsed"] is True and config["GT_parsed"] is True, "Formal GT parse receipt absent")
    else:
        require(summary["GT_parsed"] is False and config["GT_parsed"] is False and summary["statistics"] is None and summary["investment_gate"] is None,
                "Engineering summary fabricated opportunity statistics")
        for key in ("matched_candidates", "labelable_matched_candidates", "unmatched_candidates", "matched_but_unlabelable_candidates", "unique_matched_GT_instances"):
            require(summary[key] is None, "Engineering summary GT count must stay unknown")
    gt_any_all_counts = defaultdict(lambda: defaultdict(int))
    with (output/"GT_GROUPS.jsonl").open("w", encoding="utf-8") as gt_output:
        for key in sorted(gt_groups):
            group = gt_groups[key]
            collapsed = {}
            for label, values in group["labels"].items():
                collapsed[label] = {"any": three_value_any(values), "all": three_value_all(values),
                                    "true_count": values.count(True), "false_count": values.count(False), "unknown_count": values.count(None)}
                for operation in ("any", "all"):
                    gt_any_all_counts[label][operation+"_"+str(collapsed[label][operation]).lower()] += 1
            line(gt_output, {"image_id": group["image_id"], "annotation_id": group["annotation_id"],
                             "detection_indices": group["detection_indices"], "matched_detection_count": len(group["detection_indices"]),
                             "fixed_label_any_all": collapsed})
    return {"image_count": len(ids), "native_candidates": total_candidates, "actions": total_actions,
            "matched_candidates": total_matches if mode == "opportunity" else None,
            "labelable_matched_candidates": total_labelable if mode == "opportunity" else None,
            "all_matched_unique_GT_count": len(gt_groups) if mode == "opportunity" else None,
            "unique_GT_any_all": {key: dict(value) for key, value in gt_any_all_counts.items()} if mode == "opportunity" else None,
            "images_without_labelable_matches": sum(counts[iid]["labelable"] == 0 for iid in ids) if mode == "opportunity" else None,
            "zero_local_increments_retained": sum(counts[iid]["zero_local_increment"] for iid in ids) if mode == "opportunity" else None,
            "unknown_reasons": dict(unknown_reasons), "original_changed_area_sums_by_family": dict(original_change_sums),
            "component_truncation_totals": dict(truncation), "recorded_engineering_assertion_counts": dict(checks),
            "statistics": statistics, "investment_gate": independent_gate(statistics) if statistics is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study-root", "run-id", "input-run-id"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--mode", required=True, choices=("engineering", "opportunity"))
    parser.add_argument("--engineering-run-id", help="Required in formal mode: prior complete native engineering Run")
    args = parser.parse_args()
    if args.mode == "opportunity" and not args.engineering_run_id:
        parser.error("Formal verification requires --engineering-run-id")
    study = Path(args.study_root).resolve()
    require(study == Path(__file__).resolve().parents[1], "Verifier Study root differs")
    for value in (args.run_id, args.input_run_id, args.engineering_run_id):
        if value is not None:
            require(value and Path(value).name == value, "Run ID must be one path component")
    output, source = study/"runs"/args.run_id, study/"runs"/args.input_run_id
    require(output != source and (output/"run.json").is_file(), "Register a separate verification Run first")
    require(not any((output/name).exists() for name in ("SUMMARY.json", "VERIFICATION.json", "FAILURE.json", "REAGGREGATED_CANDIDATES.jsonl")), "Verifier history exists; retry requires a new Run ID")
    started, began = now(), time.perf_counter()
    try:
        script_hash = sha(__file__)
        (output/"source").mkdir(exist_ok=True)
        shutil.copy2(__file__, output/"source"/"verify_opportunity.py")
        meta, summary = completed_run(source, args.mode)
        panel, configuration, watched = provenance(source, meta, summary)
        panel_images = panel_replay(panel, watched)
        require(configuration["GT_used_for_action_generation"] is False, "Configuration GT action-generation declaration differs")
        if args.mode == "opportunity":
            engineering = study/"runs"/args.engineering_run_id
            _, evidence = completed_run(engineering, "engineering")
            engineering_input = read(engineering/"EXECUTION_INPUTS.json")["configuration"]
            require(evidence["image_count"] == 4 and engineering_input["image_ids"] == sorted(panel_images)[:4] and
                    engineering_input["panel_sha256"] == configuration["panel_sha256"] and engineering_input["source_sha256"] == configuration["source_sha256"],
                    "Prior native engineering evidence is from another panel/source")
            for key in ("native_input_parity_candidates", "native_original_parity_candidates", "identity_bound_candidates", "zero_exact_actions",
                        "cross_identity_rejection_candidates", "changed_baseline_rejection_candidates", "forced_empty_export_probe_candidates"):
                require(evidence["engineering_checks"][key] == evidence["native_candidates"] > 0, "Prior engineering assertion coverage incomplete")
            for filename in ("run.json", "SUMMARY.json", "COMPLETE.json", "EXECUTION_INPUTS.json"):
                watched[str((engineering/filename).resolve())] = sha(engineering/filename)
        dump(output/"VERIFICATION_INPUTS.json", {"verifier_version": VERSION, "verifier_sha256": script_hash,
                                                 "source_run_id": args.input_run_id, "engineering_run_id": args.engineering_run_id,
                                                 "mode": args.mode, "before_sha256": watched, "started_at": started})
        result = verify_records(source, output, panel_images, configuration, summary, args.mode, watched)
        after = {path: sha(path) for path in watched}
        require(after == watched and sha(__file__) == script_hash, "Source/panel/input/verifier bytes changed during independent verification")
        report = {"status": "verification_complete", "passed": True, "mode": args.mode, "verifier_version": VERSION,
                  "source_run_id": args.input_run_id, "source_run_status": meta["status"], "source_run_return_code": meta["return_code"],
                  "source_summary_sha256": sha(source/"SUMMARY.json"), "source_panel_sha256": configuration["panel_sha256"],
                  "source_protocol_sha256": configuration["source_sha256"]["protocol"], "verifier_sha256": script_hash,
                  "checks": {"source_snapshots_initial_and_current_exact": True, "panel_uniform512_and_calibration_semantics_exact": True,
                             "all_native_candidate_and_action_identity_coverage": True, "all_unknown_labels_preserved": True,
                             "stored_original_RLE_and_sparse_actions_checked": True, "stored_pixel_ledger_arithmetic_checked": True,
                             "nested_groups_with_single_zero_max23": True, "candidate_denominator_zero_increments_preserved": True,
                             "paired_image_cluster_bootstrap_and_gate_checked": True if args.mode == "opportunity" else None},
                  "fresh_model_native_decode_rerun": False, "fresh_GT_decode_rerun": False, "GT_matching_rerun": False,
                  "native_receipt_scope": "Original native bytes asserted by source execution; checker verifies assertion coverage and bound stored zero RLE, not a second forward",
                  "GT_verification_scope": "Existing fixed association and original pixel TP/FP/FN ledgers only; no original annotation-mask recomputation",
                  "dropped_component_scope": "Pre-cap counts and pixel totals internally checked; selected sparse components independently connected/sorted; omitted components not regenerated",
                  "statistics_units": "IoU/coverage/purity fractions 0..1; original pixel counts; candidate ratios with paired image clusters; unique GT any/all separate",
                  "results": result, "hashes_unchanged": True, "watched_file_count": len(watched), "after_sha256": after,
                  "timing": {"total_elapsed_seconds": time.perf_counter()-began}, "started_at": started, "completed_at": now(),
                  "limitations": ["Recorded matching validity cannot prove fresh GT decode or source-association correctness",
                                  "Normal native output scope and GT oracle opportunity do not establish complete raw geometry, AP, RGB learnability, or deployment safety",
                                  "Positive oracle zero-damage is a nested-set construction property",
                                  "Engineering unknown labels and unmeasured opportunity statistics remain null"]}
        dump(output/"VERIFICATION.json", report)
        dump(output/"SUMMARY.json", {key: report[key] for key in ("status", "passed", "mode", "source_run_id", "source_summary_sha256", "checks", "results", "fresh_model_native_decode_rerun", "fresh_GT_decode_rerun", "GT_matching_rerun", "timing", "completed_at")})
        dump(output/"COMPLETE.json", {"status": "verification_complete", "summary_sha256": sha(output/"SUMMARY.json"),
                                     "verification_sha256": sha(output/"VERIFICATION.json"), "completed_at": now()})
        print(f"VERIFIED {args.mode} {args.input_run_id}: candidates={result['native_candidates']} actions={result['actions']}", flush=True)
        return 0
    except Exception as exc:
        dump(output/"FAILURE.json", {"status": "failed", "passed": False, "source_run_id": args.input_run_id,
                                   "error": repr(exc), "traceback": traceback.format_exc(), "time": now()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
