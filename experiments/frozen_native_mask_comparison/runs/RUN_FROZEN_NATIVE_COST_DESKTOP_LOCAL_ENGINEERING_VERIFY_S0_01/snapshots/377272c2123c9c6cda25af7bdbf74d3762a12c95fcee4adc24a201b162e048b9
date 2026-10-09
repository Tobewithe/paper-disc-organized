"""Independent CPU audit of the separate desktop32-reference cost profile.

Reuses pinned, data-only RLE/statistics checks from the sealed laptop checker;
three explicit environment/reference adaptations do not alter arithmetic or
parity criteria. No model, GPU, GT, original wall-clock or memory is rerun.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import re
import shutil
import time
import traceback

OLD_CHECKER_SHA = "a2e5796fa3c9e3ac6e673957bfa73c84b3befcc49dac88147d6bcf526ec8f268"
PROFILE_SHA = "b5872d1ab27a17df1b5bd2caf433fdf4fb87ba455b4ebf3ae91b7cf6945cf0be"
PROTOCOL_SHA = "f67d773459da9786f312d49acb77a8745ce83ded405e3ac0a6994e83f95b79a0"
MAP_SHA = "ab5fcb3c11956ceccd55779a342fea5d9c4800d5c5b33d7841aefc1c28fdc56b"
BINDING_SHA = "31ea3a93a2a913e16d44476f0cc00cbd9d0faa7ab81ff0eedadf5e666db2ecae"
TORCH = "2.9.1+cu128"
GPU = "NVIDIA GeForce RTX 5060 Ti"
INTERPRETER = "C:/Dpan/envsfiles/CondaData/envs/pytorch/python.exe"
VERSION = "independent_desktop32_cost_verify_v1"


def load_pinned_checks():
    path = Path(__file__).with_name("verify_cost_panel.py")
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest() != OLD_CHECKER_SHA:
        raise ValueError("Sealed laptop data-only checker dependency changed")
    spec = importlib.util.spec_from_file_location("pinned_cost_data_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == "verify_all")

    class ExplicitProfile(ast.NodeTransformer):
        def __init__(self):
            self.interpreters = self.versions = self.lookups = 0

        def visit_Constant(self, node):
            if node.value == "C:/Users/28358/anaconda3/envs/pytorch/python.exe":
                node.value = INTERPRETER
                self.interpreters += 1
            elif node.value == "2.5.1":
                node.value = TORCH
                self.versions += 1
            return node

        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "expected_remote":
                node.value = ast.parse('PureWindowsPath(inputs["reference_run"])/arm/"images"/f"{iid:012d}.json"').body[0].value
                self.lookups += 1
            return self.generic_visit(node)

    profile = ExplicitProfile()
    function = profile.visit(function)
    if (profile.interpreters, profile.versions, profile.lookups) != (1, 1, 1):
        raise ValueError("Expected three explicit desktop environment/reference adaptations differ")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), module.__dict__)
    return module


C = load_pinned_checks()


def local_seal(run, watched):
    seal = C.read(run/"LOCAL_ARTIFACT_SEAL.json")
    C.watch(run/"LOCAL_ARTIFACT_SEAL.json", watched)
    C.watch(run/"manifest.sha256", watched, seal["manifest_sha256"])
    C.require(seal.get("run_id") == run.name and seal.get("source_kind") == "local_artifact_hash_seal" and
              seal.get("execution_host") == "desktop" and seal.get("transfer_required") is False and
              seal.get("original_run_json_preserved_verbatim") is True, "Local seal/source identity differs")
    records = {}
    for text in (run/"manifest.sha256").read_text(encoding="utf-8-sig").splitlines():
        if not text.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", text)
        C.require(match is not None, "Malformed local manifest record")
        digest, name = match.groups()
        relative = Path(name.replace("\\", "/"))
        target = (run/relative).resolve()
        C.require(not relative.is_absolute() and ".." not in relative.parts and target.is_relative_to(run.resolve()), "Local manifest path escapes Run")
        normalized = str(relative).replace("\\", "/")
        C.require(normalized not in records, "Duplicate local manifest path")
        C.watch(target, watched, digest)
        records[normalized] = digest
    C.require(len(records) == seal["files"], "Local sealed file count differs")
    return seal, records


def actual_completed(run, watched, completion_name):
    meta, summary, complete = (C.read(run/name) for name in ("run.json", "SUMMARY.json", completion_name))
    for name in ("run.json", "SUMMARY.json", completion_name):
        C.watch(run/name, watched)
    C.require(meta.get("status") == "completed" and meta.get("return_code") == 0 and
              meta.get("artifact_completeness") == "complete" and meta.get("missing_outputs") == [] and
              summary.get("status") == "completed" and summary.get("passed") is True, "Original actual runner not completed0/complete")
    C.require(complete["summary_sha256"] == C.sha(run/"SUMMARY.json"), "Original complete summary fingerprint differs")
    C.require(meta.get("metrics_source", {}).get("sha256") == C.sha(run/"SUMMARY.json"), "Original runner metric fingerprint differs")
    for item in meta.get("artifacts", []):
        C.require(item.get("exists") is True, "Original runner missing artifact")
        path = Path(item["path"]).resolve()
        C.require(path.is_relative_to(run.resolve()), "Original runner artifact outside this desktop Run")
        C.watch(path, watched, item["sha256"])
    for item in meta.get("snapshots", []):
        C.watch(run/item["snapshot"], watched, item["revision"])
    return meta, summary


def reference_contract(reference, inputs, watched):
    _, manifest = local_seal(reference, watched)
    meta, summary = actual_completed(reference, watched, "COMPLETE.json")
    recorded = C.read(reference/"INPUTS.json")
    C.watch(reference/"INPUTS.json", watched)
    C.watch(reference/"SUMMARY.json", watched, inputs["source_binding"]["native_reference_summary_sha256"])
    C.watch(reference/"BASELINE_PARITY.json", watched, inputs["source_binding"]["native_reference_parity_sha256"])
    parity = C.read(reference/"BASELINE_PARITY.json")
    C.require(summary["image_count"] == 32 and summary["diagnostic_only"] is False and
              summary["complete_common_baseline_exact"] is True and summary["isolated_cost_executor_used"] is False and
              summary["actual_torch"] == TORCH and summary["actual_gpu"] == GPU and
              summary["gt_used"] is False and summary["ap_measured"] is False and
              summary["cross_device_quality_equivalence_claimed"] is False, "Original-path desktop32 reference scope differs")
    C.equal(summary["original_paths_used"], ["FrozenYOLO.extract", "prepare_native", "NativeMaskAdapter.decode_arms",
                                           "original evaluate_epoch_r3.refine_coefficients/detection_records"], "Original independent reference paths")
    C.require(recorded["isolated_cost_executor_used"] is False and recorded["diagnostic_only"] is False and
              recorded["actual_torch"] == TORCH and recorded["actual_gpu"] == GPU and recorded["protocol_sha256"] == PROTOCOL_SHA,
              "Independent reference input/environment declaration differs")
    C.equal(recorded["image_ids"], inputs["image_ids"], "Same actual desktop32 reference panel order")
    C.require(parity["passed"] is True and parity["images"] == 32 and parity["common_original_paths_exact"] is True and
              parity["cross_device_identity_claimed"] is False, "Complete common original-path baseline receipt absent")
    frozen = summary["frozen_integrity"]
    C.require(frozen["passed"] is True and frozen["initial_state_sha256"] == frozen["final_state_sha256"] and
              frozen["base_weights_sha256"] == C.WEIGHTS_SHA and frozen["extracted_images"] == 32, "Reference frozen official model receipt differs")
    for name, digest in recorded["source_sha256"].items():
        C.watch(reference/"source"/name, watched, digest)
        C.require(name not in ("benchmark_individual_cost.py", "benchmark_desktop_local_cost.py"), "Isolated executor appears in reference-generator source set")
    C.watch(reference/"source"/"COST_PANEL_DESKTOP_LOCAL_REFERENCE_PROTOCOL.md", watched, PROTOCOL_SHA)
    head = C.read(reference/"HEAD_PROVENANCE.json")
    C.watch(reference/"HEAD_PROVENANCE.json", watched)
    C.require(head["head_sha256"] == C.HEAD_SHA and head["loaded_state_sha256"] == C.HEAD_STATE_SHA and
              head["core_sha256"] == C.CORE_SHA and head["runtime_sha256"] == C.RUNTIME_SHA and
              head["diagnostic_scope"] == "full" and head["no_gt_forward"] is True, "Original final8 reference head/source differs")
    for name, digest in (("triflow_model.py", C.CORE_SHA), ("diagnostics_runtime.py", C.RUNTIME_SHA),
                         ("evaluate_epoch_r3.py", head["evaluator_sha256"])):
        C.watch(reference/"source"/"triflow"/name, watched, digest)
    fingerprints, native_rows = {}, {}
    C.watch(reference/"BASELINE_PARITY_IMAGES.jsonl", watched)
    for row in C.jsonlines(reference/"BASELINE_PARITY_IMAGES.jsonl"):
        C.require(row["image_id"] not in fingerprints and row["native_replay_exact"] is True and
                  row["complete_common_baseline_identity_rle_exact"] is True, "Recorded common native identity/parity coverage differs")
        fingerprints[row["image_id"]] = row["input_sha256"]
        native_rows[row["image_id"]] = row["native_rows"]
    C.equal(list(fingerprints), inputs["image_ids"], "Complete actual reference input tensor order")
    C.require(sum(native_rows.values()) == summary["native_rows"], "Reference native row totals differ")
    return fingerprints, manifest


def cost_contract(source, args, watched):
    seal, manifest = local_seal(source, watched)
    meta, summary = actual_completed(source, watched, "COST_COMPLETE.json")
    inputs = C.read(source/"COST_INPUTS.json")
    C.watch(source/"COST_INPUTS.json", watched)
    engineering = args.mode == "engineering"
    count, repeats = (4, 1) if engineering else (32, 3)
    C.require(summary["engineering"] is engineering and inputs["engineering"] is engineering and
              summary["engineering_passed"] is (True if engineering else None), "Actual desktop cost mode/gate differs")
    for key, expected in (("arm_names", list(C.ARMS)), ("image_count_per_arm_repeat", count), ("repeat_blocks", repeats),
                          ("sample_count", 5*count*repeats), ("child_processes", 5*repeats)):
        C.equal(summary[key], expected, "Desktop summary "+key)
    for key, expected in (("arm_names", list(C.ARMS)), ("measured_images", count), ("warmup_images", 4), ("repeat_blocks", repeats)):
        C.equal(inputs[key], expected, "Desktop input "+key)
    C.require(summary["source_unchanged"] is True and summary["all_output_parity_passed"] is True and
              summary["all_isolation_passed"] is True and summary["gt_used"] is False and summary["ap_measured"] is False and
              summary["confidence_intervals"] is None, "Desktop source completion/scientific scope differs")
    C.require(inputs["version"] == summary["version"] == "frozen_individual_cost_desktop_local_v1" and
              inputs["cross_device_quality_equivalence_claimed"] is False and inputs["desktop_full_ap_measured"] is False,
              "Desktop profile or quality scope differs")
    C.equal(inputs["endpoint"], C.ENDPOINT, "Same deploy endpoint")
    C.equal(inputs["deployment_excludes"], ["disk/RGB prep", "model loading", "warmup", "RLE/audit", "writing", "COCO/GT"], "Actual desktop deploy exclusions")
    C.require(Path(inputs["interpreter"]).resolve() == Path(INTERPRETER).resolve(), "Actual desktop interpreter differs")
    C.require(Path(inputs["reference_run"]).resolve() == Path(args.reference_run).resolve(), "Wrong same-machine reference Run")
    binding = inputs["source_binding"]
    C.require(C.canonical_sha(binding) == inputs["source_binding_sha256"] == summary["source_binding_sha256"] == BINDING_SHA,
              "Exact desktop source/reference binding differs")
    C.require(set(binding["source_sha256"]) == C.SOURCES | {"benchmark_desktop_local_cost.py"} and
              binding["source_sha256"]["benchmark_individual_cost.py"] == C.BENCH_SHA and
              binding["source_sha256"]["benchmark_desktop_local_cost.py"] == PROFILE_SHA and
              binding["protocol_sha256"] == PROTOCOL_SHA and binding["images_list_sha256"] == MAP_SHA, "Exact original algorithm/desktop profile lock differs")
    for name, digest in binding["source_sha256"].items():
        C.watch(source/"source"/name, watched, digest)
    C.watch(source/"source"/"COST_PANEL_DESKTOP_LOCAL_REFERENCE_PROTOCOL.md", watched, PROTOCOL_SHA)
    for key, digest in (("weights_sha256", C.WEIGHTS_SHA), ("response_sha256", C.RESPONSE_SHA), ("multi_sha256", C.MULTI_SHA),
                        ("head_sha256", C.HEAD_SHA), ("core_sha256", C.CORE_SHA), ("runtime_sha256", C.RUNTIME_SHA)):
        C.equal(binding[key], digest, "Original locked "+key)
    for name, digest in (("response.json", C.RESPONSE_SHA), ("multi_local.json", C.MULTI_SHA)):
        C.watch(source/"assets"/name, watched, digest)
    for name, key in (("triflow_model.py", "core_sha256"), ("diagnostics_runtime.py", "runtime_sha256"), ("HEAD_PROVENANCE.json", "head_provenance_sha256")):
        C.watch(source/"source"/"triflow_reference"/name, watched, binding[key])
    # These current local originals are available; validate without importing.
    command = meta["command"]
    for flag, expected in (("--weights", C.WEIGHTS_SHA), ("--protocol", PROTOCOL_SHA)):
        C.watch(command[command.index(flag)+1], watched, expected)
    vendor = Path(command[command.index("--vendor")+1])
    for name, digest in binding["vendor_source_sha256"].items():
        C.watch(vendor/name, watched, digest)
    if not engineering:
        gate = C.read(source/"ENGINEERING_GATE.json")
        C.watch(source/"ENGINEERING_GATE.json", watched)
        C.watch(gate["path"], watched, gate["sha256"])
        passed = C.read(gate["path"])
        C.require(gate["passed"] is True and passed["engineering_passed"] is True and
                  passed["source_binding_sha256"] == BINDING_SHA and passed["image_count_per_arm_repeat"] == 4,
                  "Original same-source desktop engineering gate absent")
    return meta, summary, inputs, manifest, count, repeats


def panel_contract(args, inputs, watched):
    C.watch(args.images_list, watched, MAP_SHA)
    C.watch(args.original_list, watched, C.LIST_SHA)
    paths = [Path(text.strip()).resolve() for text in Path(args.images_list).read_text(encoding="utf-8").splitlines() if text.strip()]
    original = [PureWindowsPath(text.strip()) for text in Path(args.original_list).read_text(encoding="utf-8").splitlines() if text.strip()]
    C.require(len(paths) == len(set(paths)) == len(original) == len(set(original)) == 5000, "Actual desktop identity-map/original list cardinality differs")
    C.equal([int(path.stem) for path in paths], [int(path.stem) for path in original], "Original full5000 identity order mapped to desktop")
    C.equal(inputs["image_ids"], [int(path.stem) for path in paths[:32]], "Original first32 panel")
    C.equal([Path(path).resolve() for path in inputs["image_paths"]], paths[:32], "Actual desktop panel paths")
    for item, path in zip(inputs["image_files"], paths[:32]):
        C.require(item["image_id"] == int(path.stem) and Path(item["path"]).resolve() == path and path.stat().st_size == item["bytes"], "Actual original JPEG identity/size differs")
        C.watch(path, watched, item["sha256"])


def resource_contract(source, inputs, summary, watched):
    before = C.read(source/"RESOURCES_BEFORE.json")
    C.watch(source/"RESOURCES_BEFORE.json", watched)
    def validate_sample(row):
        C.require(isinstance(row["time"], (float, int)) and row["time"] > 0 and
                  0 <= row["cpu_percent_since_last_sample"] <= 100 and row["system_available_memory_bytes"] > 0,
                  "Invalid retained physical resource sample")
        C.require(row["gpu_query_return_code"] == 0 and GPU in row["gpu_snapshot"], "Recorded physical desktop GPU sample differs")
        for process in row["python_processes"]:
            text = process["command"].lower()
            expected = "boundary" in text and "verify" not in text and "workbench" not in text
            C.equal(process["cpu_boundary_heavy_candidate"], expected, "Recorded finite Boundary-name preguard classification")
    validate_sample(before)
    pre_hits = sum(process["cpu_boundary_heavy_candidate"] for process in before["python_processes"])
    C.require(inputs["engineering"] or pre_hits == 0, "Formal desktop cost began while finite Boundary-heavy preguard had a hit")
    rows, runtime_hits = [], 0
    for child in summary["children"]:
        path = source/"measurements"/f"repeat{child['repeat_index']}"/child["arm"]/"RESOURCE_SAMPLES.json"
        C.watch(path, watched)
        record = C.read(path)
        C.require(record["sampling_interval_seconds"] == 2 and record["sampler_in_cost_child"] is False and
                  record["sampler_overhead_exactly_measured"] is False and len(record["samples"]) > 0,
                  "Actual finite parent-resource sampler contract differs")
        C.equal(record["scope"], "whole actual child lifetime including initialization, warmup, deployment and audit; not aligned to deployment-only intervals", "Resource sampling scope")
        timestamps = []
        for sample in record["samples"]:
            validate_sample(sample)
            timestamps.append(sample["time"])
            runtime_hits += sum(process["cpu_boundary_heavy_candidate"] for process in sample["python_processes"])
        C.require(timestamps == sorted(timestamps), "Resource observations are not ordered")
        rows.append({"arm": child["arm"], "repeat_index": child["repeat_index"], "samples": len(timestamps),
                     "recorded_cpu_percent_min": min(row["cpu_percent_since_last_sample"] for row in record["samples"]),
                     "recorded_cpu_percent_max": max(row["cpu_percent_since_last_sample"] for row in record["samples"]),
                     "sampling_interval_seconds": 2, "aligned_to_deployment_only": False})
    return {"before_heavy_candidate_hits": pre_hits, "during_resource_heavy_candidate_hits": runtime_hits, "child_resources": rows,
            "sampler_in_cost_child": False, "sampler_overhead_exactly_measured": False,
            "limitations": "finite process-name scan and two-second nominal system sampling do not prove absence of every workload or capture every peak; parent sampler overhead unquantified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study-root", "run-id", "input-run-id", "reference-run", "images-list", "original-list"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--mode", required=True, choices=("engineering", "formal"))
    args = parser.parse_args()
    study = Path(args.study_root).resolve()
    C.require(study == Path(__file__).resolve().parents[1], "Desktop checker Study root differs")
    for name in (args.run_id, args.input_run_id):
        C.require(name and Path(name).name == name, "Run ID must be one path component")
    source, output = study/"runs"/args.input_run_id, study/"runs"/args.run_id
    C.require(source != output and (output/"run.json").is_file(), "Register a distinct desktop CPU verification Run first")
    C.require(not any((output/name).exists() for name in ("SUMMARY.json", "VERIFICATION.json", "FAILURE.json", "VERIFICATION_INPUTS.json")), "Verification history exists; retry needs a new Run ID")
    watched, started, began = {}, C.now(), time.perf_counter()
    try:
        code_sha = C.sha(__file__)
        (output/"source").mkdir(exist_ok=True)
        shutil.copy2(__file__, output/"source"/"verify_desktop_cost_panel.py")
        shutil.copy2(Path(__file__).with_name("verify_cost_panel.py"), output/"source"/"verify_cost_panel.py")
        C.watch(Path(__file__).with_name("verify_cost_panel.py"), watched, OLD_CHECKER_SHA)
        meta, summary, inputs, manifest, count, repeats = cost_contract(source, args, watched)
        panel_contract(args, inputs, watched)
        fingerprints, _ = reference_contract(Path(args.reference_run).resolve(), inputs, watched)
        resources = resource_contract(source, inputs, summary, watched)
        args.reference_panel = args.native_reference = args.reference_run
        C.dump(output/"VERIFICATION_INPUTS.json", {"version": VERSION, "mode": args.mode, "source_run_id": source.name,
                                                   "checker_sha256": code_sha, "pinned_data_checker_sha256": OLD_CHECKER_SHA,
                                                   "actual_source_binding_sha256": BINDING_SHA, "before_sha256": watched})
        results = C.verify_all(args, source, output, inputs, summary, {"source_directory": str(source)}, manifest,
                               inputs["image_ids"][:count], fingerprints, watched, repeats)
        branch_counts = {"measured_outputs": 0, "native_empty_outputs": 0, "empty_mask_rows": 0, "unsupported_first64_rows": 0}
        for sample in C.read(source/"COST_SAMPLES.json")["samples"]:
            branch_counts["measured_outputs"] += 1
            branch_counts["native_empty_outputs"] += int(sample["native_rows"] == 0)
            if sample["arm"] != "baseline":
                branch_counts["unsupported_first64_rows"] += sample["method_work"]["selected_first64"]-sample["method_work"]["supported_first64"]
            frame = source/"measurements"/f"repeat{sample['repeat_index']}"/sample["arm"]/"outputs"/f"{sample['image_id']:012d}.json"
            for detection in C.read(frame)["detections"]:
                branch_counts["empty_mask_rows"] += int(not C.decode_rle(detection["segmentation"]).any())
        after = {path: C.sha(path) for path in watched}
        C.require(after == watched and C.sha(__file__) == code_sha, "Desktop source/reference/checker evidence changed while reading")
        report = {"version": VERSION, "status": "verification_complete", "passed": True, "mode": args.mode,
                  "source_run_id": source.name, "source_summary_sha256": C.sha(source/"SUMMARY.json"), "checker_sha256": code_sha,
                  "pinned_data_checker_sha256": OLD_CHECKER_SHA, "source_binding_sha256": BINDING_SHA,
                  "profile": "desktop same-machine original-path reference32; Torch2.9.1+cu128/RTX5060Ti", "results": results,
                  "resources": resources, "actual_branch_coverage": branch_counts,
                  "source_literal_limitations_scope": "source summary retains historical laptop wording; actual local interpreter/runtime/inputs/reference and new protocol define this profile",
                  "same_machine_reference_independent_original_paths_verified": True, "isolated_executor_used_to_generate_reference": False,
                  "cross_device_quality_equivalence_claimed": False, "laptop_AP_paired_with_desktop_cost": False, "desktop_full_AP_measured": False,
                  "fresh_forward_rerun": False, "wall_clock_remeasured": False, "GT_used": False, "GPU_used": False,
                  "binary_scope": "lossless retained RLE to recorded uint8 binary SHA; actual original/common-baseline forward assertions remain producer receipt evidence",
                  "memory_scope": "GPU per-endpoint reset peak includes resident/warm cache; reserved is retained capacity. CPU cumulative peak is whole fresh child including init/warm/audit; sampled RSS not an exact deployment peak",
                  "pure_startup_seconds": None, "startup_scope": "observed process wall minus child wall includes launch/recording/receipt writing/shutdown/supervision; isolated startup not measured",
                  "confidence_intervals": None, "hashes_unchanged": True, "watched_file_count": len(watched), "after_sha256": after,
                  "verification_CPU_wall_seconds": time.perf_counter()-began, "started_at": started, "completed_at": C.now(),
                  "limitations": ["Same desktop32 fixed implementation cost; no laptop-bit/quality equivalence, desktop full AP or boundary-quality confirmation",
                                  "Branches with zero actual observed counts remain unexercised; no synthetic parity coverage inferred from summaries",
                                  "Three cyclic arm rotations reduce drift but do not fully balance positions",
                                  "Resource sampler covers child lifetime, not deploy-only intervals, with unquantified background sampler overhead"]}
        C.dump(output/"VERIFICATION.json", report)
        C.dump(output/"SUMMARY.json", {key: report[key] for key in ("version", "status", "passed", "mode", "source_run_id", "source_summary_sha256", "checker_sha256", "pinned_data_checker_sha256", "source_binding_sha256", "profile", "results", "resources", "actual_branch_coverage", "cross_device_quality_equivalence_claimed", "laptop_AP_paired_with_desktop_cost", "fresh_forward_rerun", "wall_clock_remeasured", "hashes_unchanged", "verification_CPU_wall_seconds", "completed_at")})
        C.dump(output/"COMPLETE.json", {"status": "verification_complete", "summary_sha256": C.sha(output/"SUMMARY.json"),
                                       "verification_sha256": C.sha(output/"VERIFICATION.json"), "completed_at": C.now()})
        print(f"VERIFIED_DESKTOP {source.name} samples={results['sample_count']} branches={branch_counts}", flush=True)
        return 0
    except Exception as exc:
        C.dump(output/"FAILURE.json", {"status": "failed", "passed": False, "source_run_id": source.name, "error": repr(exc),
                                     "traceback": traceback.format_exc(), "time": C.now(), "fresh_forward_rerun": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
