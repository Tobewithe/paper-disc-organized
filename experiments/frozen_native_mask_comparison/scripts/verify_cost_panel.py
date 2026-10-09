"""Independent CPU audit/reaggregation of a fully returned individual-cost Run.

No benchmark modules, Torch, forward, GT, GPU, SSH or original wall-clock
measurement are executed. Source clocks are reaggregated from actual samples;
retained RLE is decoded losslessly and compared with recorded binary hashes.
Fresh binary/native execution assertions remain source-receipt evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
import shutil
import sys
import time
import traceback

import numpy as np

VERSION = "independent_individual_cost_verify_v1"
ARMS = ("baseline", "RCMC_first64", "multi_local_first64", "global_minus025_first64", "TriFlow_final8")
BENCH_SHA = "8b6ccf6d770cdacc10ed43f625b59c15fb37f8df6ec5554692cb4a44be09bcd6"
LAPTOP_PROTOCOL_SHA = "2211ae66a47560c81f658791e1a02680dbcccba5326c9288392c943be96277b0"
DESKTOP_PROTOCOL_SHA = "85ba9fb8b0534d1f34723ece2d8f1809ba9bb8ee446afe4757a1c68e7fc6c13f"
LIST_SHA = "b20742148d06ff75864eb0ffe47cb3a3e010e8913dafdcd46ceeb1ca498d09db"
WEIGHTS_SHA = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
RESPONSE_SHA = "eb2fca8b9bbb0192d62723e3da086c303729752772aa1a357ea64f61f70888b9"
MULTI_SHA = "71874d3f23e6cbe2179ce0962c751b7afd04b9e29581a0c3227515ca48b76987"
CORE_SHA = "1b96e0767ab1985875cdfec33cdd6e98a666ecd3ee5caefc7dbca9e80c35691e"
RUNTIME_SHA = "c24bc853ff17c7f4a0be3bc847a8b6733283b2aa06bc16388ab1b84fca3380b3"
HEAD_SHA = "5c47c39dd5912999af8d8a11e0052a9bcb291c55413eb63b3c4bdc3c054707c8"
HEAD_STATE_SHA = "9bbe280e1de3c61a92a2c4595aeb3b5e855665a105112b39ae9646a5602772e8"
SOURCES = {"benchmark_individual_cost.py", "frozen_io.py", "native_mask_adapter.py", "portable_risk.py",
           "risk_calibration.py", "local_features.py", "mask_calibration.py"}
ENDPOINT = "preloaded original RGB ndarray through preprocessing/native forward/method to all original binary masks on CPU"
EXCLUDES = ["disk read/RGB prep", "model/gate/head loading", "warmup", "RLE/identity audit", "JSON writing", "COCO/GT"]
GPU_SCOPE = "reset immediately before this deployment endpoint; includes resident own arm model/assets"
CPU_SCOPE = "entire fresh arm-repeat process, including initialization/warmup/prior audit; not reset or deployment-only"
CHILD_CPU_SCOPE = "fresh process entire lifetime including init/warmup/audit"
AUDIT_FIELDS = ("reference_read_seconds", "rle_records_encoding_seconds", "identity_rle_parity_seconds",
                "input_tensor_audit_seconds", "output_write_seconds")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def equal(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label+": keys differ")
        for key in expected:
            equal(actual[key], expected[key], label+"."+key)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label+": length differs")
        for index, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, label+f"[{index}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (float, int)) and not isinstance(actual, bool) and math.isfinite(actual) and
                math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label+": number differs")
    else:
        require(actual == expected and (not isinstance(expected, bool) or isinstance(actual, bool)), label+": value differs")


def integer(value, label, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum, label+": invalid integer")
    return value


def seconds(value, label):
    require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0,
            label+": invalid measured nonnegative seconds")
    return float(value)


def description(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.size > 0 and np.isfinite(values).all(), "Cannot describe absent/nonfinite clocks")
    return {"n": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "std_population": float(np.std(values, ddof=0)), "p10": float(np.quantile(values, .1)),
            "p90": float(np.quantile(values, .9)), "min": float(values.min()), "max": float(values.max())}


def jsonlines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for text in handle:
            require(bool(text.strip()), "Empty raw sample record")
            yield json.loads(text)


def watch(path, watched, expected=None):
    path = Path(path).resolve()
    require(path.is_file(), "Retained evidence missing: "+str(path))
    actual = sha(path)
    require(expected is None or actual == expected, "Retained evidence SHA differs: "+str(path))
    if str(path) in watched:
        require(watched[str(path)] == actual, "Evidence changed during verification: "+str(path))
    watched[str(path)] = actual
    return actual


def source_relative(value, source_directory):
    """Rebase a declared remote path without guessing a filesystem location."""
    value, root = PureWindowsPath(value), PureWindowsPath(source_directory)
    require(value.is_absolute() and root.is_absolute(), "Recorded source paths must be absolute")
    try:
        relative = value.relative_to(root)
    except ValueError as exc:
        raise AssertionError("Recorded artifact is outside its declared source Run") from exc
    require(".." not in relative.parts, "Recorded artifact escapes source Run")
    return Path(*relative.parts)


def transferred_source(source, watched):
    transfer = read(source/"transfer.json")
    watch(source/"transfer.json", watched)
    watch(source/"manifest.sha256", watched)
    require(transfer.get("source_host") == "28358lan" and transfer.get("all_manifest_sha256_verified") is True,
            "Cost source must be an actually verified 28358lan return")
    require(transfer.get("transfer_status") == "complete_scientific_consumption", "Partial transport cannot establish a complete scientific cost package")
    require(transfer.get("run_id") == source.name, "Transfer Run identity differs")
    records = {}
    for text in (source/"manifest.sha256").read_text(encoding="utf-8-sig").splitlines():
        if not text.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", text)
        require(match is not None, "Malformed transfer SHA manifest record")
        digest, name = match.groups()
        relative = Path(name.replace("\\", "/"))
        require(not relative.is_absolute() and ".." not in relative.parts and name not in records, "Invalid/duplicate manifest path")
        resolved = (source/relative).resolve()
        require(resolved.is_relative_to(source.resolve()), "Manifest path escapes source Run")
        watch(resolved, watched, digest.lower())
        records[str(relative).replace("\\", "/")] = digest.lower()
    require(len(records) == transfer.get("verified_files"), "Transfer verified-file count differs")
    return transfer, records


def require_manifest(path, source, manifest):
    name = str(Path(path).resolve().relative_to(source.resolve())).replace("\\", "/")
    require(name in manifest, "Required original evidence is absent from transfer manifest: "+name)


def source_contract(source, watched, mode):
    transfer, manifest = transferred_source(source, watched)
    meta, summary, inputs, complete = (read(source/name) for name in ("run.json", "SUMMARY.json", "COST_INPUTS.json", "COST_COMPLETE.json"))
    for name in ("run.json", "SUMMARY.json", "COST_INPUTS.json", "COST_COMPLETE.json", "COST_SAMPLES.json", "PAIRED_COST_INCREMENTS.json"):
        require_manifest(source/name, source, manifest)
        watch(source/name, watched)
    watch(source/"COST_SOURCE_PACKAGE_RECEIPT.json", watched)
    require(meta.get("status") == "completed" and meta.get("return_code") == 0 and
            meta.get("artifact_completeness") == "complete" and meta.get("missing_outputs") == [], "Original runner is not exit0/complete")
    require(meta.get("run_id") == source.name and summary.get("status") == "completed" and summary.get("passed") is True,
            "Original complete Run/summary identity differs")
    require(not (source/"COST_FAILURE.json").exists(), "Source cost failure receipt exists")
    require(complete.get("status") == "completed" and complete.get("summary_sha256") == sha(source/"SUMMARY.json"), "Original cost completion SHA differs")
    for key in ("all_output_parity_passed", "all_isolation_passed"):
        require(summary.get(key) is True and complete.get(key) is True, "Source complete assertion missing: "+key)
    require(summary.get("source_unchanged") is True and summary.get("gt_used") is False and
            summary.get("ap_measured") is False and summary.get("confidence_intervals") is None, "Cost source scientific scope differs")
    engineering = mode == "engineering"
    equal(summary["engineering"], engineering, "mode")
    equal(summary["engineering_passed"], True if engineering else None, "actual engineering gate state")
    equal(inputs["engineering"], engineering, "input mode")
    equal(summary["arm_names"], ARMS, "five fixed arms")
    equal(inputs["arm_names"], ARMS, "input arms")
    equal(inputs["endpoint"], ENDPOINT, "deployment endpoint")
    equal(inputs["deployment_excludes"], EXCLUDES, "deployment exclusions")
    count, repeats = (4, 1) if engineering else (32, 3)
    equal(inputs["measured_images"], count, "measured images")
    equal(inputs["warmup_images"], 4, "warmup image count")
    equal(inputs["repeat_blocks"], repeats, "repeat blocks")
    equal(summary["image_count_per_arm_repeat"], count, "summary measured images")
    equal(summary["repeat_blocks"], repeats, "summary repeat blocks")
    equal(summary["sample_count"], len(ARMS)*count*repeats, "sample cardinality")
    equal(summary["child_processes"], len(ARMS)*repeats, "fresh child cardinality")
    equal(inputs["version"], "frozen_individual_cost_v1", "locked laptop implementation version")
    binding = inputs["source_binding"]
    require(canonical_sha(binding) == inputs["source_binding_sha256"] == summary["source_binding_sha256"], "Execution source-binding fingerprint differs")
    require(set(binding["source_sha256"]) == SOURCES and binding["source_sha256"]["benchmark_individual_cost.py"] == BENCH_SHA,
            "Exact locked laptop source set differs")
    require(binding["protocol_sha256"] == LAPTOP_PROTOCOL_SHA and binding["images_list_sha256"] == LIST_SHA,
            "Locked laptop protocol/list differs")
    for key, digest in (("weights_sha256", WEIGHTS_SHA), ("response_sha256", RESPONSE_SHA), ("multi_sha256", MULTI_SHA),
                        ("core_sha256", CORE_SHA), ("runtime_sha256", RUNTIME_SHA), ("head_sha256", HEAD_SHA)):
        equal(binding[key], digest, "locked "+key)
    for name, digest in binding["source_sha256"].items():
        path = source/"source"/name
        require_manifest(path, source, manifest)
        watch(path, watched, digest)
    watch(source/"source"/"COST_PANEL_PROTOCOL.md", watched, LAPTOP_PROTOCOL_SHA)
    watch(source/"assets"/"response.json", watched, RESPONSE_SHA)
    watch(source/"assets"/"multi_local.json", watched, MULTI_SHA)
    for name, key in (("triflow_model.py", "core_sha256"), ("diagnostics_runtime.py", "runtime_sha256"),
                      ("HEAD_PROVENANCE.json", "head_provenance_sha256")):
        watch(source/"source"/"triflow_reference"/name, watched, binding[key])
    head = read(source/"source"/"triflow_reference"/"HEAD_PROVENANCE.json")
    require(head.get("snapshot_sha256") == HEAD_SHA and head.get("loaded_state_sha256") == HEAD_STATE_SHA and head.get("evaluation_epoch") == 8,
            "Archived actual final8 head provenance differs")
    for record in meta.get("artifacts", []):
        relative = source_relative(record["path"], transfer["source_directory"])
        require(record.get("exists") is True, "Original recorder observed a missing output")
        watch(source/relative, watched, record["sha256"])
    for record in meta.get("snapshots", []):
        require(record["revision"] == watch(source/record["snapshot"], watched), "Original before-source snapshot SHA differs")
    if not engineering:
        gate = read(source/"ENGINEERING_GATE.json")
        require_manifest(source/"ENGINEERING_GATE.json", source, manifest)
        watch(source/"ENGINEERING_GATE.json", watched)
        require(gate.get("passed") is True and isinstance(gate.get("sha256"), str) and len(gate["sha256"]) == 64,
                "Formal source lacks a passed actual engineering gate")
        remote_study = PureWindowsPath(transfer["source_directory"]).parents[1]
        try:
            gate_relative = PureWindowsPath(gate["path"]).relative_to(remote_study)
        except ValueError as exc:
            raise AssertionError("Engineering gate source is outside this declared Study") from exc
        gate_local = source.parents[1]/Path(*gate_relative.parts)
        watch(gate_local, watched, gate["sha256"])
        original_engineering = read(gate_local)
        require(original_engineering.get("passed") is True and original_engineering.get("engineering_passed") is True and
                original_engineering.get("source_binding_sha256") == inputs["source_binding_sha256"] and
                original_engineering.get("image_count_per_arm_repeat") == 4 and original_engineering.get("arm_names") == list(ARMS),
                "Formal gate does not reference the actual same-source completed engineering receipt")
    return meta, summary, inputs, transfer, manifest, count, repeats


def panel_sources(args, inputs, watched, count):
    listing = Path(args.images_list).resolve()
    watch(listing, watched, LIST_SHA)
    original_paths = [PureWindowsPath(text.strip()) for text in listing.read_text(encoding="utf-8-sig").splitlines() if text.strip()]
    require(len(original_paths) == len(set(original_paths)) == 5000, "Locked original5000 source list is incomplete")
    panel_ids = [int(path.stem) for path in original_paths[:32]]
    equal(inputs["image_ids"], panel_ids, "original-order first32 panel IDs")
    equal([int(PureWindowsPath(value).stem) for value in inputs["image_paths"]], panel_ids, "source image path identity/order")
    equal([item["image_id"] for item in inputs["image_files"]], panel_ids, "source JPEG fingerprints order")
    require(len(set(inputs["image_paths"])) == 32, "Duplicate panel path")
    for item, path in zip(inputs["image_files"], inputs["image_paths"]):
        require(PureWindowsPath(item["path"]) == PureWindowsPath(path) and item["bytes"] > 0 and len(item["sha256"]) == 64,
                "Source JPEG fingerprint/path contract differs")
        local_jpeg = Path(args.images_root)/PureWindowsPath(path).name
        watch(local_jpeg, watched, item["sha256"])
        require(local_jpeg.stat().st_size == item["bytes"], "Returned original JPEG size differs")
    native = Path(args.native_reference).resolve()
    watch(native/"SUMMARY.json", watched, inputs["source_binding"]["native_reference_summary_sha256"])
    watch(native/"BASELINE_PARITY.json", watched, inputs["source_binding"]["native_reference_parity_sha256"])
    native_summary, native_parity = read(native/"SUMMARY.json"), read(native/"BASELINE_PARITY.json")
    require(native_summary.get("image_count") == 5000 and native_summary.get("passed") is True and native_parity.get("passed") is True,
            "Actual native source metadata is not the complete parity-passing5000 reference")
    fingerprints = {}
    path = native/"BASELINE_PARITY_IMAGES.jsonl"
    watch(path, watched)
    for record in jsonlines(path):
        iid = record["image_id"]
        if iid in panel_ids:
            require(iid not in fingerprints and len(record["input_sha256"]) == 64, "Duplicate/invalid native input fingerprint")
            fingerprints[iid] = record["input_sha256"]
    require(set(fingerprints) == set(panel_ids), "Native original panel input tensor fingerprints incomplete")
    return panel_ids[:count], fingerprints


def decode_rle(rle):
    """Official compressed COCO counts -> a C-contiguous uint8 original mask."""
    height, width = rle["size"]
    integer(height, "RLE height", 1)
    integer(width, "RLE width", 1)
    require(isinstance(rle["counts"], str), "Retained masks must have ASCII compressed RLE counts")
    encoded, cursor, counts = rle["counts"], 0, []
    while cursor < len(encoded):
        value, shift, more = 0, 0, True
        while more:
            require(cursor < len(encoded), "Truncated RLE count")
            code = ord(encoded[cursor])-48
            cursor += 1
            require(0 <= code <= 63, "Invalid compressed RLE character")
            value |= (code & 31) << shift
            more = bool(code & 32)
            shift += 5
            if not more and code & 16:
                value |= -1 << shift
        if len(counts) > 2:
            value += counts[-2]
        require(value >= 0, "Negative reconstructed RLE count")
        counts.append(value)
    require(sum(counts) == height*width, "RLE does not cover declared original image")
    flat, offset = np.zeros(height*width, dtype=np.uint8), 0
    for index, length in enumerate(counts):
        if index % 2:
            flat[offset:offset+length] = 1
        offset += length
    return np.ascontiguousarray(flat.reshape((height, width), order="F"))


def loaded_assets(arm):
    return {"baseline": {}, "global_minus025_first64": {}, "RCMC_first64": {"response": RESPONSE_SHA},
            "multi_local_first64": {"multi_local": MULTI_SHA},
            "TriFlow_final8": {"head": HEAD_SHA, "head_state": HEAD_STATE_SHA, "triflow_model.py": CORE_SHA,
                               "diagnostics_runtime.py": RUNTIME_SHA, "diagnostic_scope": "minimal"}}[arm]


def method_contract(arm, sample):
    method = sample["method_work"]
    rows = integer(sample["native_rows"], "normal native rows")
    require(rows <= 300, "Normal output exceeds max_det300")
    n = integer(method["method_target_rows"], "actual method target rows")
    require(n == method["supported_first64"] and n <= method["selected_first64"] <= min(64, rows), "First64/protoROI count contract differs")
    if arm == "baseline":
        require(method["selected_first64"] == n == 0, "Baseline executed a method subset")
    else:
        equal(method["selected_first64"], min(64, rows), "method first64 budget")
    expected = {"baseline": (0, 0, 0, 0), "global_minus025_first64": (n, 0, 0, 0),
                "RCMC_first64": (n, n, 0, n), "multi_local_first64": (5*n, 5*n, 5*n, 5*n),
                "TriFlow_final8": (0, 0, 0, 0)}[arm]
    equal(tuple(method[key] for key in ("action_rows", "five_feature_rows", "eighteen_feature_rows", "gate_rows")), expected,
          "isolated method work counts")
    if arm == "TriFlow_final8":
        equal(method["diagnostic_scope"], "minimal", "TriFlow deployment diagnostics")
    return rows


def verify_all(args, source, output, inputs, summary, transfer, manifest, ids, fingerprints, watched, repeats):
    reconstructed, children, process_rows = [], [], []
    reference_cache, rgb_by_image, native_rows_by_image = {}, {}, {}
    signed = {arm: [] for arm in ARMS}
    final_order = [(repeat, arm, position) for repeat in range(repeats)
                   for position, arm in enumerate(ARMS[repeat:]+ARMS[:repeat])]
    require(len(summary["children"]) == len(final_order), "Original observed child list is incomplete")
    previous_finish = None
    reference_panel = Path(args.reference_panel).resolve()
    for child_index, (repeat, arm, position) in enumerate(final_order):
        folder = source/"measurements"/f"repeat{repeat}"/arm
        for name in ("PROCESS_OBSERVED.json", "CHILD_COMPLETE.json", "COST_SAMPLES.jsonl", "RAW_RGB_INPUTS.json"):
            require_manifest(folder/name, source, manifest)
            watch(folder/name, watched)
        require(not (folder/"CHILD_FAILURE.json").exists(), "A child failure is retained in an alleged complete cost Run")
        observed, receipt = read(folder/"PROCESS_OBSERVED.json"), read(folder/"CHILD_COMPLETE.json")
        require(observed["arm"] == receipt["arm"] == arm and observed["repeat_index"] == receipt["repeat_index"] == repeat,
                "Fresh child identity differs")
        require(observed["order_position"] == position and observed["return_code"] == 0 and observed["pid"] == receipt["pid"] > 0,
                "Actual child order/exit/PID differs")
        start, finish = dt.datetime.fromisoformat(observed["started_at"]), dt.datetime.fromisoformat(observed["finished_at"])
        require(finish >= start and (previous_finish is None or start >= previous_finish), "Child processes were not serial in declared cyclic order")
        previous_finish = finish
        require(receipt["status"] == "completed" and receipt["passed"] is True and receipt["all_output_parity_passed"] is True and
                receipt["all_isolation_passed"] is True and receipt["source_binding_sha256"] == observed["source_binding_sha256"] == inputs["source_binding_sha256"],
                "Child complete/source/parity/isolation receipt differs")
        equal(receipt["image_count"], len(ids), "child full measured panel")
        equal(receipt["engineering"], args.mode == "engineering", "child mode")
        equal(receipt["loaded_assets"], loaded_assets(arm), "only own required assets loaded")
        require(PureWindowsPath(receipt["interpreter"]) == PureWindowsPath(inputs["interpreter"]) ==
                PureWindowsPath("C:/Users/28358/anaconda3/envs/pytorch/python.exe"), "Actual child interpreter differs from laptop protocol")
        require(receipt["runtime"]["torch"] == "2.5.1" and receipt["runtime"]["tf32"] is False and
                receipt["runtime"]["cudnn_tf32"] is False and receipt["runtime"]["threads"] == 4,
                "Actual recorded laptop runtime/FP32 guard differs")
        require(receipt["gt_used"] is False and receipt["ap_measured"] is False and receipt["scientific_scale_claimed"] is False,
                "Child cost-only scientific scope differs")
        equal(receipt["head_unchanged"], True if arm == "TriFlow_final8" else None, "head unchanged receipt")
        frozen = receipt["frozen_integrity"]
        require(frozen["passed"] is True and frozen["initial_state_sha256"] == frozen["final_state_sha256"] and
                frozen["base_weights_sha256"] == WEIGHTS_SHA and frozen["extracted_images"] == len(ids)+4,
                "Recorded official frozen model integrity/forward count differs")
        for key in ("all_state_exact", "all_params_requires_grad_false", "all_bn_eval", "all_gradients_none"):
            require(frozen[key] is True, "Frozen child assertion absent: "+key)
        require(watch(folder/"COST_SAMPLES.jsonl", watched) == receipt["samples_sha256"], "Child sample bytes differ from complete receipt")
        samples = list(jsonlines(folder/"COST_SAMPLES.jsonl"))
        equal([sample["image_id"] for sample in samples], ids, "per-child original measured order")
        raw_rgb = read(folder/"RAW_RGB_INPUTS.json")["images"]
        equal([record["image_id"] for record in raw_rgb], ids, "preloaded RGB panel order")
        for record in raw_rgb:
            require(len(record["rgb_shape"]) == 3 and record["rgb_shape"][2] == 3 and len(record["rgb_bytes_sha256"]) == 64,
                    "Raw RGB shape/hash declaration differs")
            if record["image_id"] in rgb_by_image:
                equal(record, rgb_by_image[record["image_id"]], "recorded same RGB across all arms/repeats")
            else:
                rgb_by_image[record["image_id"]] = record
        equal([record["image_id"] for record in receipt["warmup_samples"]], ids[:4], "fixed warmup first4")
        warm_sum = sum(seconds(record["seconds"], "warmup endpoint") for record in receipt["warmup_samples"])
        require(seconds(receipt["warmup_seconds"], "complete warmup wall")+1e-9 >= warm_sum,
                "Warmup wall is shorter than its measured endpoints")
        phase = {key: seconds(receipt[key], key) for key in ("dependency_import_seconds", "image_read_and_rgb_prepare_seconds",
                 "weight_and_own_asset_initialization_seconds", "warmup_seconds", "total_child_seconds")}
        process_wall = seconds(observed["process_wall_seconds"], "actual observed process wall")
        equal(receipt["process_cpu_peak_scope"], CHILD_CPU_SCOPE, "whole fresh-process CPU peak scope")
        phase["measured_deployment_seconds_sum"] = 0.0
        phase.update({key: 0.0 for key in AUDIT_FIELDS})
        for sample in samples:
            require(sample["arm"] == arm and sample["repeat_index"] == repeat, "Per-sample arm/repeat differs")
            iid = sample["image_id"]
            rows = method_contract(arm, sample)
            if iid in native_rows_by_image:
                equal(rows, native_rows_by_image[iid], "same normal native candidate scope across all arms/repeats")
            else:
                native_rows_by_image[iid] = rows
            for key in ("isolated_method_and_assets", "all_rows_identity_and_rle_exact", "native_input_tensor_exact"):
                require(sample[key] is True, "Source per-image actual assertion failed: "+key)
            equal(sample["native_input_tensor_sha256"], fingerprints[iid], "recorded native tensor vs original source fingerprint")
            seconds(sample["deployment_seconds"], "measured deployment endpoint")
            equal(sample["cuda_peak_scope"], GPU_SCOPE, "reset endpoint GPU peak scope")
            equal(sample["cpu_peak_scope"], CPU_SCOPE, "cumulative process CPU peak scope")
            memory = sample["memory_after_endpoint"]
            for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "cpu_current_rss_bytes"):
                integer(memory[key], "memory "+key)
            require(memory["cuda_peak_reserved_bytes"] >= memory["cuda_peak_allocated_bytes"], "CUDA reserved peak below allocated")
            equal(sample["cpu_rss_after_endpoint_bytes"], memory["cpu_current_rss_bytes"], "sample endpoint RSS after")
            integer(sample["cpu_rss_before_endpoint_bytes"], "sample RSS before")
            if memory["cpu_process_cumulative_peak_bytes"] is not None:
                require(memory["cpu_process_cumulative_peak_bytes"] >= memory["cpu_current_rss_bytes"], "OS cumulative peak below sampled RSS")
            phase["measured_deployment_seconds_sum"] += sample["deployment_seconds"]
            for key in AUDIT_FIELDS:
                phase[key] += seconds(sample[key], key)
            saved = folder/"outputs"/f"{iid:012d}.json"
            require_manifest(saved, source, manifest)
            watch(saved, watched)
            output_rows = read(saved)
            require(output_rows["image_id"] == iid and len(output_rows["detections"]) == rows, "Stored whole normal output coverage differs")
            reference = reference_panel/arm/"images"/f"{iid:012d}.json"
            if not reference.is_file() and arm != "TriFlow_final8":
                reference = Path(args.native_reference)/arm/"images"/f"{iid:012d}.json"
            watch(reference, watched, sample["reference_sha256"])
            cached = reference_cache.setdefault((arm, iid), read(reference))
            require(cached.get("image_id") == iid and output_rows["detections"] == cached.get("detections"),
                    "Full stored detection identity/RLE differs from the original frozen reference")
            if "original_shape" in cached:
                equal(cached["original_shape"], rgb_by_image[iid]["rgb_shape"][:2], "reference original shape")
            expected_remote = (PureWindowsPath(inputs["triflow_reference"])/"triflow" if arm == "TriFlow_final8"
                               else PureWindowsPath(inputs["reference_run"])/arm)/"images"/f"{iid:012d}.json"
            require(PureWindowsPath(sample["reference_file"]) == expected_remote, "Source reference cache path points to another arm/image")
            binary_digest = hashlib.sha256()
            for index, detection in enumerate(output_rows["detections"]):
                require(detection["image_id"] == iid and detection["detection_index"] == index, "Stored candidate identity/order changed")
                binary = decode_rle(detection["segmentation"])
                equal(list(binary.shape), rgb_by_image[iid]["rgb_shape"][:2], "all masks on original pixel grid")
                binary_digest.update(binary.tobytes(order="C"))
            require(binary_digest.hexdigest() == sample["output_mask_binary_sha256"],
                    "Losslessly reconstructed saved RLE mask bytes differ from recorded output binary signature")
            reconstructed.append(sample)
        cpu_peak = receipt["process_final_memory"]["cpu_process_cumulative_peak_bytes"]
        if cpu_peak is not None:
            integer(cpu_peak, "whole fresh child cumulative CPU peak")
            require(cpu_peak >= max(sample["cpu_rss_after_endpoint_bytes"] for sample in samples), "Final OS peak below endpoint samples")
        children.append(observed | {"child_complete_sha256": sha(folder/"CHILD_COMPLETE.json"),
                                  "initialization_seconds": receipt["weight_and_own_asset_initialization_seconds"],
                                  "dependency_import_seconds": receipt["dependency_import_seconds"],
                                  "image_read_seconds": receipt["image_read_and_rgb_prepare_seconds"],
                                  "warmup_seconds": receipt["warmup_seconds"], "cpu_process_peak_bytes": cpu_peak})
        process_rows.append({"arm": arm, "repeat_index": repeat, "order_position": position, "pid": observed["pid"],
                             "recorded_phase_seconds": phase, "observed_process_wall_seconds": process_wall,
                             "outside_child_wall_residual_seconds": process_wall-phase["total_child_seconds"],
                             "pure_startup_seconds": None, "startup_scope": "residual also includes before/after child recording, final receipt writing, shutdown and supervision; no isolated startup timer",
                             "cpu_fresh_process_peak_bytes": cpu_peak, "cpu_peak_scope": CHILD_CPU_SCOPE,
                             "postwarmup_RSS_one_point_bytes": receipt["postwarmup_rss_bytes"],
                             "postwarmup_RSS_scope": "one postwarmup sample; no continuous deployment RSS peak measurement",
                             "runtime": receipt["runtime"], "all_stored_outputs_verified": True})
    equal(summary["children"], children, "independent actual child receipts/order/phase reconstruction")
    equal(read(source/"COST_SAMPLES.json"), {"samples": reconstructed}, "aggregate samples vs each original child raw ledger")
    lookup = {(sample["repeat_index"], sample["image_id"], sample["arm"]): sample for sample in reconstructed}
    require(len(lookup) == len(ARMS)*len(ids)*repeats, "Missing/duplicate arm/image/repeat samples")
    statistics, costs = {}, {}
    for arm in ARMS:
        samples = [sample for sample in reconstructed if sample["arm"] == arm]
        signed[arm] = [{"repeat_index": sample["repeat_index"], "image_id": sample["image_id"],
                        "signed_deployment_increment_seconds": sample["deployment_seconds"]-lookup[(sample["repeat_index"], sample["image_id"], "baseline")]["deployment_seconds"]}
                       for sample in samples]
        statistics[arm] = {"deployment_seconds": description([sample["deployment_seconds"] for sample in samples]),
                           "paired_signed_increment_seconds": description([item["signed_deployment_increment_seconds"] for item in signed[arm]]),
                           "repeat_mean_seconds": [description([sample["deployment_seconds"] for sample in samples if sample["repeat_index"] == repeat])["mean"] for repeat in range(repeats)],
                           "gpu_peak_allocated_bytes": max(sample["memory_after_endpoint"]["cuda_peak_allocated_bytes"] for sample in samples),
                           "gpu_peak_reserved_bytes": max(sample["memory_after_endpoint"]["cuda_peak_reserved_bytes"] for sample in samples),
                           "cpu_fresh_process_peak_bytes_per_repeat": [child["cpu_process_peak_bytes"] for child in children if child["arm"] == arm]}
        costs[arm] = {"deployment_seconds": statistics[arm]["deployment_seconds"], "signed_increment_seconds": statistics[arm]["paired_signed_increment_seconds"],
                      "negative_signed_increment_samples": sum(item["signed_deployment_increment_seconds"] < 0 for item in signed[arm]),
                      "recorded_outside_deployment_seconds": {key: description([sample[key] for sample in samples]) for key in AUDIT_FIELDS},
                      "gpu_endpoint_peak": {"allocated_bytes": statistics[arm]["gpu_peak_allocated_bytes"], "reserved_bytes": statistics[arm]["gpu_peak_reserved_bytes"], "scope": GPU_SCOPE},
                      "cpu_fresh_process_peak_bytes_per_repeat": statistics[arm]["cpu_fresh_process_peak_bytes_per_repeat"], "cpu_peak_scope": CHILD_CPU_SCOPE}
    equal(summary["statistics"], statistics, "independent pooled mean/median/populationstd/p10/p90/paired increment/memory aggregation")
    equal(read(source/"PAIRED_COST_INCREMENTS.json"), signed, "all signed per-image/repeat baseline differences; negatives retained")
    dump(output/"REAGGREGATED_COST.json", {"arms": costs, "statistics": statistics, "paired_signed_increments": signed,
                                           "clock_units": "seconds", "confidence_intervals": None})
    dump(output/"PROCESS_PHASES.json", {"children": process_rows, "endpoint": ENDPOINT,
                                        "phase_unit": "seconds; source recorded clocks, independently grouped not remeasured"})
    return {"arms": costs, "statistics": statistics, "image_count_per_arm_repeat": len(ids), "repeat_blocks": repeats,
            "fresh_child_processes": len(children), "sample_count": len(reconstructed), "retained_mask_rows_compared": sum(sample["native_rows"] for sample in reconstructed),
            "all_sample_signed_increments_checked": True, "all_child_phase_and_memory_scopes_checked": True,
            "all_saved_output_RLE_and_recorded_binary_signatures_checked": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study-root", "run-id", "input-run-id", "reference-panel", "native-reference", "images-list"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--images-root", help="Local retained original32 JPEG bytes; defaults to Study/assets/cost_panel/images")
    parser.add_argument("--mode", required=True, choices=("engineering", "formal"))
    args = parser.parse_args()
    study = Path(args.study_root).resolve()
    if args.images_root is None:
        args.images_root = str(study/"assets"/"cost_panel"/"images")
    require(study == Path(__file__).resolve().parents[1], "Checker Study root differs")
    for value in (args.run_id, args.input_run_id):
        require(value and Path(value).name == value, "Run ID must be one path component")
    source, output = study/"runs"/args.input_run_id, study/"runs"/args.run_id
    require(source != output and (output/"run.json").is_file(), "Register a distinct CPU verification Run with the original runner first")
    require(not any((output/name).exists() for name in ("SUMMARY.json", "VERIFICATION.json", "FAILURE.json", "VERIFICATION_INPUTS.json")), "Checker history exists; retry requires a new Run ID")
    began, started, watched = time.perf_counter(), now(), {}
    try:
        checker_sha = sha(__file__)
        (output/"source").mkdir(exist_ok=True)
        shutil.copy2(__file__, output/"source"/"verify_cost_panel.py")
        meta, summary, inputs, transfer, manifest, count, repeats = source_contract(source, watched, args.mode)
        ids, fingerprints = panel_sources(args, inputs, watched, count)
        dump(output/"VERIFICATION_INPUTS.json", {"version": VERSION, "source_run_id": source.name,
                                                 "mode": args.mode, "checker_sha256": checker_sha,
                                                 "source_binding_sha256": inputs["source_binding_sha256"],
                                                 "before_sha256": watched, "started_at": started})
        result = verify_all(args, source, output, inputs, summary, transfer, manifest, ids, fingerprints, watched, repeats)
        after = {path: sha(path) for path in watched}
        require(after == watched and sha(__file__) == checker_sha, "Returned inputs/reference/checker bytes changed during verification")
        report = {"version": VERSION, "status": "verification_complete", "passed": True, "mode": args.mode,
                  "source_run_id": source.name, "source_return_code": meta["return_code"], "source_summary_sha256": sha(source/"SUMMARY.json"),
                  "checker_sha256": checker_sha, "source_binding_sha256": inputs["source_binding_sha256"], "results": result,
                  "source_profile": "laptop Torch2.5.1; five isolated serial arms; fixed original panel", "deployment_endpoint": ENDPOINT,
                  "fresh_forward_rerun": False, "wall_clock_remeasured": False, "fresh_original_RGB_decode": False,
                  "raw_memory_remeasured": False, "source_GPU_or_process_execution_trace_rerun": False,
                  "stored_binary_evidence_scope": "retained lossless output RLE reconstructed to uint8 mask bytes, exact reference RLE/identity and recorded binary SHA; no fresh native tensor/binary forward comparison",
                  "input_evidence_scope": "original list/panel order and retained JPEG/RGB/input tensor fingerprints plus original native input ledger; RGB not redecoded",
                  "isolation_evidence_scope": "locked executable source, own-asset and method-row receipts, child PID/order/serial timestamp coverage; no new execution trace",
                  "memory_scope": "GPU per-endpoint reset peak includes resident model/assets and warm cache; reserved is retained capacity, not marginal/minimum memory. CPU Windows peak covers complete fresh arm-repeat process including init/warmup/audit; RSS endpoints and postwarm sample are not an exact deployment peak",
                  "pure_process_startup_seconds": None, "startup_scope": "process wall and child wall separately retained; residual includes launch/recording/receipt writing/teardown and supervision, cannot isolate startup",
                  "confidence_intervals": None, "GT_used": False, "AP_measured": False, "hashes_unchanged": True,
                  "manifest_verified_file_count": len(manifest), "watched_file_count": len(watched), "after_sha256": after,
                  "verification_CPU_wall_seconds": time.perf_counter()-began, "started_at": started, "completed_at": now(),
                  "limitations": ["Fixed first32 source images and registered laptop endpoint; engineering first4 is a contract check, not formal cost evidence",
                                  "Three cyclic rotations mitigate drift and do not fully balance arm positions",
                                  "Stored clocks/OS counters are reaggregated; timing and memory measurement authenticity beyond source receipts is not freshly established",
                                  "Original head/model state and byte parity assertions refer to sealed producer receipts; model/GPU/GT/COCO are not rerun",
                                  "No seven-arm joint time division, TriFlow709.739ms different diagnostic boundary, or cross-device timing pool used"]}
        dump(output/"VERIFICATION.json", report)
        dump(output/"SUMMARY.json", {key: report[key] for key in ("version", "status", "passed", "mode", "source_run_id", "source_summary_sha256", "checker_sha256", "source_binding_sha256", "results", "fresh_forward_rerun", "wall_clock_remeasured", "memory_scope", "pure_process_startup_seconds", "confidence_intervals", "hashes_unchanged", "verification_CPU_wall_seconds", "completed_at")})
        dump(output/"COMPLETE.json", {"status": "verification_complete", "summary_sha256": sha(output/"SUMMARY.json"),
                                     "verification_sha256": sha(output/"VERIFICATION.json"), "completed_at": now()})
        print(f"VERIFIED {source.name} mode={args.mode} samples={result['sample_count']} children={result['fresh_child_processes']}", flush=True)
        return 0
    except Exception as exc:
        dump(output/"FAILURE.json", {"status": "failed", "passed": False, "source_run_id": source.name, "error": repr(exc),
                                   "traceback": traceback.format_exc(), "time": now(), "wall_clock_remeasured": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
