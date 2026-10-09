"""Strengthened independent saved-evidence target-cost verification.

The original e9 checker remains unchanged. This CPU-only wrapper verifies
source/fit/fresh-process/memory receipts, then invokes its pinned data checker
in a separate output subdirectory. No GT, model, GPU or original clock runs.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

BASE_CHECKER_SHA = "e9b0d8bf58a5a7f211aec05e230467319e7aaea0a18082fc79295a64a43aafac"
HELPER_SHA = "a2e5796fa3c9e3ac6e673957bfa73c84b3befcc49dac88147d6bcf526ec8f268"
PROTOCOL_SHA = "cc0d3169e32fc1d42dbd8f1027861ef29fa75f18bb145addd73e6c77d6a94057"
BENCHMARK_SHA = "0fca24abdec1591a9d7b2aa49bfb794108fa3c8bef1956ea82afcb79b2045345"
WEIGHTS_SHA = "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"
PRODUCERS = {
    "target_common.py": "4de7929a7278038de66c0daacba17403217db71643d8b77ccd806658bb1f011f",
    "native_response.py": "386ccd5f5f6dbbfd2d472c626bff15fc2896754234cac3a9a23aa3372e78f024",
    "extract_targets.py": "c83044e8918086da30dbb1008e8962cb32be65c28ded4e190fdea003fe4d6392",
    "fit_target.py": "274143455b85fa6a25fa156aa69790dc58a0d0c79855c22ad8458aae678878d2",
    "infer_targets.py": "bffd38d255a8d98b60d83ed71eae1c8973a7fb9e14e2dfbc46c23c7e0c573734",
    "run_target_pipeline.py": "22d3a9d94e0ccb894e803e9aaa34c5c831e8973ce3d390dd7561da0289cd56a7",
    "preflight_targets.py": "dc4442326a0e83d8083d0e1a6508db3e9e000e5e4669d1e3d579eb56e31ed390",
    "collect_target_runs.py": "b29d63d3a7f4127111d9300673e4f9be7c9abda420bffba95b0a48c1f8f6951a",
}
FROZEN = {
    "frozen_io.py": "cff75eec547d9ec12eda5409bd273231ab99c16cdc6119d75f6d413098363293",
    "native_mask_adapter.py": "0a11d7cee6f73ec680ce7895f9501a2f87be5d97a8904d3fe6b3a62a958fa0c2",
    "risk_calibration.py": "b4d1fbae794e6e1de0a8d00bb99a721687224e5b530ac816638e113d3a09ae6f",
    "portable_risk.py": "b92dee9681af044ff5dd555a39f1b6c66d35314159be7ab26331ba6c321dccb9",
    "mask_calibration.py": "d238d586df78987b2288f1927599199e2a489ae855de48854d827246da40cac8",
    "local_features.py": "a0bdefe767f2fa38536e7c6261e32a1bc066238cc90bb7585de70490e8b0caf6",
}
GPU_SCOPE = "reset immediately before endpoint; includes resident own model and retained allocator cache"
CPU_SCOPE = "fresh process lifetime including imports/init/warmup/audit; OS cumulative peak not reset"
ARMS = ("baseline", "target_I", "target_H")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base-checker", "helper", "study", "input", "reference", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--mode", choices=("engineering", "formal"), required=True)
    a = p.parse_args()
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest(a.base_checker) != BASE_CHECKER_SHA or digest(a.helper) != HELPER_SHA:
        raise ValueError("Pinned independent data checker/helper changed")
    h = load_module(a.helper, "target_cost_v2_helper")
    q, read = h.require, h.read
    study, source, reference, out = [Path(x).resolve() for x in (a.study, a.input, a.reference, a.output)]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "VERIFICATION.json").exists() or (out / "BASE_CHECKER").exists():
        raise FileExistsError("Independent retry needs a new Run")
    watched = {}
    began = time.perf_counter()
    try:
        def watch(path, expected=None):
            return h.watch(path, watched, expected)
        watch(a.base_checker, BASE_CHECKER_SHA)
        watch(a.helper, HELPER_SHA)
        watch(__file__)

        def runner(root):
            m = read(root / "run.json")
            watch(root / "run.json")
            q(m["source_kind"] == "runner_observed" and m["status"] == "completed" and
              m["return_code"] == 0 and m["artifact_completeness"] == "complete", "Actual runner source gate differs")
            q(m["run_id"] == root.name, "Literal source Run identity differs")
            for s in m["snapshots"]:
                path = (root / s["snapshot"]).resolve()
                q(path.is_relative_to(root), "Source snapshot escapes Run")
                watch(path, s["revision"])
            for f in m["artifacts"]:
                q(f["exists"] is True, "Declared source artifact missing")
                path = root / h.source_relative(f["path"], m["locations"][0]["path"])
                watch(path, f["sha256"])
            return m

        def returned(root):
            transfer = read(root / "transfer.json")
            watch(root / "transfer.json")
            q(transfer["status"] == "complete_scientific_consumption" and
              transfer["all_manifest_sha256_verified"] is True and transfer["source_run_json_preserved"] is True,
              "Complete returned byte proof missing")
            listed = set()
            watch(root / "manifest.sha256")
            for line in (root / "manifest.sha256").read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                expected, relative = line.split(None, 1)
                path = (root / relative.strip().lstrip("*")).resolve()
                q(path.is_relative_to(root) and path not in listed, "Manifest escapes or duplicates source file")
                listed.add(path)
                watch(path, expected)
            actual = {p.resolve() for p in root.rglob("*") if p.is_file() and
                      p.name not in ("manifest.sha256", "transfer.json") and not p.name.endswith(".tmp")}
            q(listed == actual, "Original complete source manifest file set differs")
            return len(listed)

        def lock(root, summary):
            path = root / "SOURCE_LOCK.json"
            watch(path, summary["source_lock_sha256"])
            value = read(path)
            q(h.canonical_sha(value["vendor_python_files"]) == value["vendor_tree_sha256"], "Archived vendor tree identity differs")
            expected = {"study/" + k: v for k, v in PRODUCERS.items()}
            expected.update({"frozen/" + k: v for k, v in FROZEN.items()})
            expected["PROTOCOL.md"] = PROTOCOL_SHA
            for key, expected_sha in expected.items():
                q(value["files"][key]["sha256"] == expected_sha, "Fixed producer/frozen source differs: " + key)
            for key, info in value["files"].items():
                if info["snapshot"] is not None:
                    snapshot = (root / info["snapshot"]).resolve()
                    q(snapshot.is_relative_to(root), "Source lock snapshot escapes Run")
                    watch(snapshot, info["sha256"])
            return value

        metadata = runner(source)
        source_files = returned(source)
        inputs, summary = read(source / "COST_INPUTS.json"), read(source / "SUMMARY.json")
        binding = inputs["source_binding"]
        q(binding["script_sha256"] == BENCHMARK_SHA and binding["protocol_sha256"] == PROTOCOL_SHA and
          binding["weights_sha256"] == WEIGHTS_SHA and binding["producer_source_sha256"] == PRODUCERS,
          "Frozen production/cost/checkpoint binding differs")
        q(BENCHMARK_SHA in {s["revision"] for s in metadata["snapshots"]}, "Cost source not actually snapshotted")
        q(inputs["repeats"] == (1 if a.mode == "engineering" else 3), "Declared repeat count differs")
        runner(reference)
        reference_files = returned(reference)
        rs = read(reference / "SUMMARY.json")
        q(rs["status"] == "prediction_complete" and rs["arm_names"] == list(ARMS) and
          rs["image_count"] == len(rs["image_ids"]) == 5000 and rs["source_unchanged"] is True and
          rs["gt_parsed"] is False and rs["gt_used_in_inference"] is False, "Formal GT-free reference scope differs")
        rl = lock(reference, rs)
        config = read(reference / rl["files"]["config.json"]["snapshot"])
        q(rl["files"]["config.json"]["sha256"] == binding["config_sha256"] and
          config["producer_source_sha256"] == PRODUCERS, "Original execution config differs")
        q(any(info["path"].replace("\\", "/").lower() == config["val"]["images_list"].replace("\\", "/").lower() and
              info["sha256"] == h.LIST_SHA for info in rl["files"].values()), "Original val list source hash differs")
        reference_inputs = read(reference / "INPUTS.json")
        q(reference_inputs["model_sha256"] == binding["models_sha256"] and
          reference_inputs["protocol_sha256"] == PROTOCOL_SHA and
          reference_inputs["config_sha256"] == binding["config_sha256"], "Formal reference model/config binding differs")
        for arm in ARMS:
            ac = read(reference / arm / "COMPLETE.json")
            receipt = rs["prediction_receipts"][arm]
            watch(reference / arm / "COMPLETE.json", receipt["complete_sha256"])
            watch(reference / arm / "predictions.json", receipt["predictions_sha256"])
            q(ac["status"] == "prediction_complete" and ac["image_ids"] == rs["image_ids"] and
              ac["image_count"] == 5000 and ac["predictions_sha256"] == receipt["predictions_sha256"] and
              ac["gt_used"] is False and ac["empty_masks_preserved"] is True, "Full reference arm completion differs")
        fit_values = {}
        for arm in ARMS[1:]:
            model = study / h.source_relative(inputs["models"][arm], metadata["working_directory"])
            root = model.parent
            q(root == study / h.source_relative(reference_inputs["model_runs"][arm], metadata["working_directory"]),
              "Cost model path differs from actual formal reference estimator")
            runner(root)
            returned(root)
            fit, complete = read(root / "SUMMARY.json"), read(root / "COMPLETE.json")
            profile = binding["fit_profiles"][arm]
            q(reference_inputs["model_summary_sha256"][arm] == profile["summary_sha256"],
              "Formal reference estimator summary differs from cost profile")
            lock(root, fit)
            q(fit["status"] == "fit_complete" and fit["target"] == arm[-1] and fit["iterations"] == 100 and
              fit["training_rows"] > 0 and fit["features"] == 5 and fit["source_unchanged"] is True and
              fit["runtime"]["sklearn"] == "1.6.1" and fit["runtime"]["torch"] == "2.5.1",
              "Actual new20k fit scope differs")
            q(complete["passed"] and complete["summary_sha256"] == watch(root / "SUMMARY.json") and
              complete["model_sha256"] == fit["model_sha256"] == watch(model, binding["models_sha256"][arm]),
              "Actual estimator completion/model binding differs")
            for key in ("full_hgb_parameters", "portable_sign_mismatches", "portable_max_abs_error",
                        "training_feature_bytes_sha256", "supervised_binding_sha256", "weight_bytes_sha256"):
                h.equal(profile[key], fit[key], "Cost fit profile." + key)
            unit_sha = hashlib.sha256(np.ones(fit["training_rows"], dtype=np.float64).tobytes()).hexdigest()
            q(fit["weight_bytes_sha256"] == unit_sha and fit["sample_weight"] == "unit default, no sample_weight argument",
              "Default unit candidate weights differ")
            fit_values[arm] = fit
        for key in ("extraction_run", "extraction_summary_sha256", "extraction_complete_sha256", "full_hgb_parameters",
                    "training_feature_bytes_sha256", "supervised_binding_sha256", "weight_bytes_sha256",
                    "training_rows", "training_images", "runtime"):
            h.equal(fit_values["target_I"][key], fit_values["target_H"][key], "Identical fit budget." + key)
        extraction = study / h.source_relative(fit_values["target_I"]["extraction_run"], metadata["working_directory"])
        runner(extraction)
        returned(extraction)
        es, ec, ei = [read(extraction / name) for name in ("SUMMARY.json", "COMPLETE.json", "INPUTS.json")]
        el = lock(extraction, es)
        watch(extraction / "SUMMARY.json", fit_values["target_I"]["extraction_summary_sha256"])
        watch(extraction / "COMPLETE.json", fit_values["target_I"]["extraction_complete_sha256"])
        watch(extraction / "INPUTS.json", es["inputs_sha256"])
        q(es["passed"] is True and es["status"] == "extraction_complete" and es["engineering"] is False and
          es["images"] == 20000 and es["source_unchanged"] is True and ec["passed"] is True and
          ec["summary_sha256"] == h.sha(extraction / "SUMMARY.json"), "Actual full20k extraction completion differs")
        q(ei["image_ids"] == ei["original_image_ids"] and len(ei["image_ids"]) == len(set(ei["image_ids"])) == 20000,
          "Shared full20k extraction identity/order differs")
        q(any(info["sha256"] == "ef3453e65b80a2d07997c882b0624a88f6d8161fcd32e2d629cdff3285ba3c63"
              for info in el["files"].values()), "Original selected20k identity source lock differs")
        image_records = list(h.jsonlines(extraction / "IMAGES.jsonl"))
        q([v["image_id"] for v in image_records] == ei["image_ids"] and
          [v["position"] for v in image_records] == list(range(1, 20001)), "Actual full20k per-image receipt scope differs")
        q(sum(v["counts"]["supervised_rows"] for v in image_records) == es["totals"]["supervised_rows"] ==
          fit_values["target_I"]["training_rows"], "Shared valid training row denominator differs")
        q(es["supervised_binding_sha256"] == fit_values["target_I"]["supervised_binding_sha256"],
          "Full20k extraction row identity differs from fitted rows")

        def memory(value, label):
            for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "cpu_current_rss_bytes"):
                h.integer(value[key], label + "." + key)
            peak = value["cpu_process_cumulative_peak_bytes"]
            if peak is not None:
                h.integer(peak, label + ".cpu_process_cumulative_peak_bytes")
                q(peak >= value["cpu_current_rss_bytes"], "Lifetime CPU peak below current RSS")
            q(value["cuda_peak_reserved_bytes"] >= value["cuda_peak_allocated_bytes"], "CUDA reserved peak below allocated peak")
            return value

        children, memory_arms, child_commands = [], {arm: [] for arm in ARMS}, set()
        repeats = 1 if a.mode == "engineering" else 3
        count = 4 if a.mode == "engineering" else 32
        for repeat in range(repeats):
            for arm in ARMS[repeat:] + ARMS[:repeat]:
                child = source / "measurements" / f"repeat{repeat}" / arm
                rc = read(child / "CHILD_COMPLETE.json")
                process_root = child.parent / (arm + "_process")
                started, observed = read(process_root / "PROCESS_STARTED.json"), read(process_root / "PROCESS_OBSERVED.json")
                q(rc["pid"] == started["pid"] == observed["pid"],
                  "Fresh child's retained process identity differs")
                h.integer(rc["pid"], "Observed scientific child PID", minimum=1)
                command = started["command"]
                def argument(flag):
                    q(command.count(flag) == 1, "Actual child command flag absent/duplicated")
                    return command[command.index(flag)+1]
                q(argument("--arm") == arm and argument("--repeat") == str(repeat) and
                  study / h.source_relative(argument("--child-output"), metadata["working_directory"]) == child and
                  argument("--protocol-sha256") == PROTOCOL_SHA and
                  argument("--cost-protocol-sha256") == binding["cost_protocol_sha256"],
                  "Retained actual own child command differs")
                command_key = (started["started_at"], tuple(command))
                q(command_key not in child_commands, "Fresh process start receipt repeated")
                child_commands.add(command_key)
                q(rc["gt_used"] is False and rc["ap_measured"] is False and
                  rc["original_method_path"] == ("baseline" if arm == "baseline" else "RCMC_first64"),
                  "Own necessary-path/GT-free child differs")
                frozen = rc["frozen_integrity"]
                q(frozen["passed"] is True and frozen["base_weights_sha256"] == WEIGHTS_SHA and
                  frozen["initial_state_sha256"] == frozen["final_state_sha256"] == rs["frozen_integrity"]["final_state_sha256"] and
                  all(frozen[k] is True for k in ("all_state_exact", "all_params_requires_grad_false", "all_bn_eval", "all_gradients_none")) and
                  frozen["extracted_images"] == count + 4, "Retained frozen official source integrity differs")
                for key in ("dependency_seconds", "image_read_rgb_seconds", "initialization_seconds", "warmup_seconds", "total_child_seconds"):
                    h.seconds(rc[key], key)
                h.seconds(observed["process_wall_seconds"], "Fresh process wall")
                q(observed["process_wall_seconds"] >= rc["total_child_seconds"], "Child clock exceeds observed process lifetime")
                h.equal(rc["warmup_seconds"], sum(h.seconds(v["seconds"], "Warmup image") for v in rc["warmup_samples"]), "Warmup sum")
                memory(rc["postwarm_memory"], "postwarm")
                final = memory(rc["process_final_memory"], "final")
                samples = list(h.jsonlines(child / "COST_SAMPLES.jsonl"))
                measured_stage_seconds = sum(rc[k] for k in ("dependency_seconds", "image_read_rgb_seconds", "initialization_seconds", "warmup_seconds"))
                cpu_previous = rc["postwarm_memory"]["cpu_process_cumulative_peak_bytes"]
                for item in samples:
                    work = item["method_work"]
                    for field in ("selected_first64", "supported_first64", "method_target_rows", "action_rows", "five_feature_rows", "eighteen_feature_rows", "gate_rows"):
                        h.integer(work[field], "Actual method work." + field)
                    q(work["selected_first64"] == (0 if arm == "baseline" else min(64, item["native_rows"])),
                      "Original first64 selected scope differs")
                    measured_stage_seconds += sum(h.seconds(item[k], "Actual endpoint/audit phase." + k) for k in
                         ("deployment_seconds", "reference_read_seconds", "rle_records_encoding_seconds", "identity_rle_parity_seconds", "input_tensor_audit_seconds", "output_write_seconds"))
                    q(item["cuda_peak_scope"] == GPU_SCOPE and item["cpu_peak_scope"] == CPU_SCOPE,
                      "Actual GPU/CPU memory scope differs")
                    before = memory(item["memory_before_endpoint"], "before_endpoint")
                    after = memory(item["memory_after_endpoint"], "after_endpoint")
                    q(after["cuda_peak_allocated_bytes"] >= before["cuda_peak_allocated_bytes"] and
                      after["cuda_peak_reserved_bytes"] >= before["cuda_peak_reserved_bytes"], "Reset endpoint CUDA peaks decreased")
                    peaks = [v for v in (cpu_previous, before["cpu_process_cumulative_peak_bytes"], after["cpu_process_cumulative_peak_bytes"]) if v is not None]
                    q(peaks == sorted(peaks), "OS lifetime CPU peaks decreased")
                    cpu_previous = after["cpu_process_cumulative_peak_bytes"]
                    memory_arms[arm].append(after)
                if cpu_previous is not None and final["cpu_process_cumulative_peak_bytes"] is not None:
                    q(final["cpu_process_cumulative_peak_bytes"] >= cpu_previous, "Final lifetime peak decreased")
                q(measured_stage_seconds <= rc["total_child_seconds"] + 1e-6, "Disjoint saved stage clocks exceed child lifetime")
                children.append(dict(path=str(child), receipt_sha256=watch(child / "CHILD_COMPLETE.json"),
                                     process_wall_seconds=observed["process_wall_seconds"], child_seconds=rc["total_child_seconds"],
                                     startup_supervision_shutdown_residual_seconds=observed["process_wall_seconds"]-rc["total_child_seconds"],
                                     pure_startup_seconds=None))
        # Source paths are production metadata; normalize only their Study-relative identity.
        declared = [{**v, "path": str(study / h.source_relative(v["path"], metadata["working_directory"]))} for v in summary["children"]]
        h.equal(declared, children, "Actual child process/receipt summary")
        q(summary["fresh_children"] == len(children) == 3 * repeats and summary["images_per_arm_repeat"] == count,
          "Child/panel denominators differ")
        memory_summary = {arm: dict(n=len(values), maximum_cuda_peak_allocated_bytes=max(v["cuda_peak_allocated_bytes"] for v in values),
                                   maximum_cuda_peak_reserved_bytes=max(v["cuda_peak_reserved_bytes"] for v in values),
                                   cpu_process_cumulative_peak_bytes_max=max((v["cpu_process_cumulative_peak_bytes"] for v in values if v["cpu_process_cumulative_peak_bytes"] is not None), default=None),
                                   cuda_scope=GPU_SCOPE, cpu_scope=CPU_SCOPE) for arm, values in memory_arms.items()}
        base = load_module(a.base_checker, "target_cost_v2_base_checker")
        old_argv = sys.argv
        try:
            sys.argv = [a.base_checker, "--helper", a.helper, "--study", str(study), "--input", str(source),
                        "--reference", str(reference), "--output", str(out / "BASE_CHECKER"), "--mode", a.mode]
            base.main()
        finally:
            sys.argv = old_argv
        for path, expected in watched.items():
            q(h.sha(path) == expected, "Source receipt changed during independent verification")
        result = read(out / "BASE_CHECKER" / "VERIFICATION.json")
        result.update(verifier_sha256=h.sha(__file__), base_checker_sha256=BASE_CHECKER_SHA,
                      source_provenance_and_fitted_profile_verified=True, memory_scopes_verified=True,
                      source_manifest_files=source_files, reference_manifest_files=reference_files,
                      memory_statistics=memory_summary, actual_fresh_child_process_receipts=children,
                      total_elapsed_seconds=time.perf_counter()-began,
                      verification_runtime=dict(python=sys.version, interpreter=sys.executable, numpy=np.__version__))
        result["limits"].append("GT/checkpoint/external vendor bytes not reread; their historical source hashes and retained snapshots bound, no training/forward/matching replay")
        h.dump(out / "VERIFICATION.json", result)
        h.dump(out / "SUMMARY.json", result)
    except Exception as exc:
        h.dump(out / "FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
