"""Independent CPU data reaggregation of retained target-cost evidence.

No benchmark module, Torch, GT, forward or original clock is executed. A pinned
pure-data helper supplies compressed-RLE decoding and generic hash/numeric
checks. Retained masks are losslessly reconstructed; native execution remains
the source Run's observed evidence, not a fresh inference claim.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

ARMS = ("baseline", "target_I", "target_H")
HELPER_SHA = "a2e5796fa3c9e3ac6e673957bfa73c84b3befcc49dac88147d6bcf526ec8f268"
BASE_SHA = "8b6ccf6d770cdacc10ed43f625b59c15fb37f8df6ec5554692cb4a44be09bcd6"
COST_PROTOCOL_SHA = "f3c22f25334316c429027df81e6250fe874dc44daf9a08f7ab8da365b71c11f1"
PARAMETERS = dict(loss="squared_error", learning_rate=.05, max_iter=100, max_leaf_nodes=7,
                  min_samples_leaf=80, l2_regularization=1., early_stopping=False, random_state=20260915)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("helper", "study", "input", "reference", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--mode", choices=("engineering", "formal"), required=True)
    a = p.parse_args()
    if digest(a.helper) != HELPER_SHA:
        raise ValueError("Pinned independent CPU helper changed")
    spec = importlib.util.spec_from_file_location("target_cost_cpu_helper", a.helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    q, read, watch = helper.require, helper.read, helper.watch
    study, source, reference, out = [Path(x).resolve() for x in (a.study, a.input, a.reference, a.output)]
    if (out / "VERIFICATION.json").exists():
        raise FileExistsError("Independent retry needs new Run")
    out.mkdir(parents=True, exist_ok=True)
    watched = {}
    began = time.perf_counter()
    try:
        metadata = read(source / "run.json")
        watch(source / "run.json", watched)
        q(metadata["status"] == "completed" and metadata["return_code"] == 0 and metadata["artifact_completeness"] == "complete", "Source runner not completed")
        transfer = read(source / "transfer.json")
        watch(source / "transfer.json", watched)
        q(transfer["status"] == "complete_scientific_consumption" and transfer["all_manifest_sha256_verified"] is True, "Full cost package not verified")
        # Every original cost file is required and watched, not only summary.
        manifest = source / "manifest.sha256"
        watch(manifest, watched)
        for line in manifest.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            h, relative = line.split(None, 1)
            path = (source / relative.strip().lstrip("*")).resolve()
            q(path.is_relative_to(source), "Manifest path escapes Run")
            watch(path, watched, h)
        inputs, summary, complete = [read(source / name) for name in ("COST_INPUTS.json", "SUMMARY.json", "COST_COMPLETE.json")]
        q(complete["passed"] and summary["passed"] and summary["status"] == "cost_complete", "Cost not complete")
        q(complete["summary_sha256"] == watch(source / "SUMMARY.json", watched), "Summary digest differs")
        q(complete["samples_sha256"] == watch(source / "COST_SAMPLES.json", watched), "Raw combined samples differ")
        binding = inputs["source_binding"]
        q(inputs["source_binding_sha256"] == helper.canonical_sha(binding) == summary["source_binding_sha256"], "Cost binding differs")
        q(binding["original_benchmark_sha256"] == BASE_SHA and binding["cost_protocol_sha256"] == COST_PROTOCOL_SHA, "Reviewed path/protocol differs")
        watch(study / "COST_PROTOCOL.md", watched, COST_PROTOCOL_SHA)
        q(binding["script_sha256"] in {v["revision"] for v in metadata["snapshots"]}, "Actual cost script not snapshotted")
        for snapshot in metadata["snapshots"]:
            watch(source / snapshot["snapshot"], watched, snapshot["revision"])
        reference_summary, reference_complete, reference_inputs = [read(reference / name) for name in ("SUMMARY.json", "COMPLETE.json", "INPUTS.json")]
        watch(reference / "SUMMARY.json", watched, binding["reference_summary_sha256"])
        watch(reference / "COMPLETE.json", watched, binding["reference_complete_sha256"])
        watch(reference / "run.json", watched, binding["reference_run_metadata_sha256"])
        watch(reference / "INPUTS.json", watched, reference_summary["inputs_sha256"])
        q(reference_summary["passed"] and reference_summary["image_count"] == 5000 and not reference_summary["engineering"], "Actual formal5k reference differs")
        q(reference_complete["summary_sha256"] == digest(reference / "SUMMARY.json"), "Formal reference completion differs")
        count, repeats = (4, 1) if a.mode == "engineering" else (32, 3)
        q(inputs["engineering"] == (a.mode == "engineering") == summary["engineering"], "Engineering/formal scope differs")
        ids = inputs["image_ids"]
        q(ids == reference_inputs["original_image_ids"][:count] and len(set(ids)) == count, "Fixed first panel identity/order differs")
        source_study = metadata["working_directory"]
        def map_source(path):
            return study / helper.source_relative(path, source_study)
        model_shas = {}
        for arm in ARMS[1:]:
            model = map_source(inputs["models"][arm])
            model_shas[arm] = watch(model, watched, binding["models_sha256"][arm])
            profile = binding["fit_profiles"][arm]
            watch(model.parent / "SUMMARY.json", watched, profile["summary_sha256"])
            watch(model.parent / "COMPLETE.json", watched, profile["complete_sha256"])
            fit = read(model.parent / "SUMMARY.json")
            q(fit["passed"] and not fit["engineering"] and fit["portable_sign_mismatches"] == 0 and fit["portable_max_abs_error"] < 1e-12, "Formal fitted gate proof differs")
            helper.equal(fit["full_hgb_parameters"], profile["full_hgb_parameters"], "Full HGB params")
            for key, value in PARAMETERS.items():
                q(profile["full_hgb_parameters"][key] == value, "Frozen HGB capacity differs")
        actual = read(source / "COST_SAMPLES.json")["samples"]
        q(len(actual) == count * repeats * 3 == summary["samples"], "Sample denominator differs")
        expected_order, children = [], []
        mask_rows, empty_images, empty_masks = 0, 0, 0
        all_images = {}
        for repeat in range(repeats):
            for arm in ARMS[repeat:] + ARMS[:repeat]:
                child = source / "measurements" / f"repeat{repeat}" / arm
                raw = list(helper.jsonlines(child / "COST_SAMPLES.jsonl"))
                rc = read(child / "CHILD_COMPLETE.json")
                process = read(child.parent / (arm + "_process") / "PROCESS_OBSERVED.json")
                q(process["return_code"] == 0 and rc["passed"] and rc["image_count"] == count and rc["repeat_index"] == repeat and rc["arm"] == arm, "Fresh child observation differs")
                q(rc["source_binding_sha256"] == inputs["source_binding_sha256"] and rc["samples_sha256"] == digest(child / "COST_SAMPLES.jsonl"), "Child source/sample binding differs")
                q(rc["runtime"]["torch"] == "2.5.1" and rc["runtime"]["gpu"] == "NVIDIA GeForce RTX 4060 Laptop GPU" and not rc["runtime"]["tf32"] and not rc["runtime"]["cudnn_tf32"] and rc["runtime"]["threads"] == 4, "Laptop actual runtime differs")
                wanted_assets = {} if arm == "baseline" else {"response": model_shas[arm]}
                q(rc["loaded_assets"] == wanted_assets and rc["model_sha256"] == model_shas.get(arm), "Child loads another arm asset")
                q([r["image_id"] for r in raw] == ids, "Per-child original image order differs")
                q([r["image_id"] for r in rc["warmup_samples"]] == ids[:4], "Fixed warmup differs")
                for item in raw:
                    expected_order.append((repeat, arm, item["image_id"]))
                    q(item["arm"] == arm and item["repeat_index"] == repeat and item["loaded_assets"] == wanted_assets, "Sample method identity differs")
                    for field in ("deployment_seconds", "reference_read_seconds", "rle_records_encoding_seconds", "identity_rle_parity_seconds", "input_tensor_audit_seconds", "output_write_seconds"):
                        helper.seconds(item[field], field)
                    work = item["method_work"]
                    n = work["supported_first64"]
                    q(0 <= n <= min(64, item["native_rows"]) and work["method_target_rows"] == n, "Action support scope differs")
                    expected = (0, 0, 0, 0) if arm == "baseline" else (n, n, 0, n)
                    q(tuple(work[k] for k in ("action_rows", "five_feature_rows", "eighteen_feature_rows", "gate_rows")) == expected, "Unneeded action/gate work occurs")
                    output = read(child / "outputs" / f"{item['image_id']:012d}.json")
                    refpath = reference / arm / "images" / f"{item['image_id']:012d}.json"
                    cached = read(refpath)
                    watch(refpath, watched, item["reference_sha256"])
                    q(output["image_id"] == cached["image_id"] == item["image_id"] and output["detections"] == cached["detections"], "Retained complete output/RLE parity differs")
                    q(len(output["detections"]) == item["native_rows"] and [r["detection_index"] for r in output["detections"]] == list(range(item["native_rows"])), "All-normal row count/order differs")
                    masks = [helper.decode_rle(d["segmentation"]) for d in output["detections"]]
                    mask_rows += len(masks)
                    empty_images += int(not masks)
                    empty_masks += sum(not bool(m.any()) for m in masks)
                    shape = tuple(cached["original_shape"])
                    q(all(m.shape == shape for m in masks), "Lossless original binary shape differs")
                    binary = np.stack(masks) if masks else np.empty((0, *shape), dtype=np.uint8)
                    q(hashlib.sha256(binary.tobytes()).hexdigest() == item["output_mask_binary_sha256"], "Recorded binary differs from lossless retained RLE")
                    q(item["native_input_tensor_exact"] and item["all_rows_identity_and_rle_exact"] and item["isolated_method_and_assets"], "Source audit claim failed")
                    all_images.setdefault(item["image_id"], set()).add(item["native_input_tensor_sha256"])
                children.append(dict(arm=arm, repeat=repeat, process_wall_seconds=process["process_wall_seconds"],
                                     child_seconds=rc["total_child_seconds"], init_seconds=rc["initialization_seconds"],
                                     rgb_read_seconds=rc["image_read_rgb_seconds"], warmup_seconds=rc["warmup_seconds"],
                                     cpu_lifetime_peak_bytes=rc["process_final_memory"]["cpu_process_cumulative_peak_bytes"],
                                     pure_startup_seconds=None))
        q([(v["repeat_index"], v["arm"], v["image_id"]) for v in actual] == expected_order, "Combined actual order differs")
        # Combined samples must equal the originals, not a reconstructed subset.
        concatenated = []
        for repeat in range(repeats):
            for arm in ARMS[repeat:] + ARMS[:repeat]:
                concatenated.extend(helper.jsonlines(source / "measurements" / f"repeat{repeat}" / arm / "COST_SAMPLES.jsonl"))
        q(actual == concatenated, "Combined samples differ from true child records")
        parity = {v["image_id"]: v for v in helper.jsonlines(reference / "BASELINE_PARITY_IMAGES.jsonl")}
        watch(reference / "BASELINE_PARITY_IMAGES.jsonl", watched)
        q(all(all_images[i] == {parity[i]["input_sha256"]} for i in ids), "Every source native input fingerprint differs")
        lookup = {(v["repeat_index"], v["image_id"], v["arm"]): v for v in actual}
        q(len(lookup) == len(actual), "Duplicate arm/repeat/image sample")
        statistics = {}
        for arm in ARMS:
            items = [v for v in actual if v["arm"] == arm]
            delta = [v["deployment_seconds"]-lookup[v["repeat_index"], v["image_id"], "baseline"]["deployment_seconds"] for v in items]
            statistics[arm] = dict(deployment_seconds=helper.description([v["deployment_seconds"] for v in items]),
                                  paired_signed_increment_seconds=helper.description(delta), negative_increment_count=sum(x < 0 for x in delta))
            helper.equal(summary["statistics"][arm], statistics[arm], "Independent signed cost statistics")
        q(summary["confidence_intervals"] is None and not summary["gt_used"] and not summary["ap_measured"], "Unperformed fields fabricated")
        resource = read(source / "RESOURCE_BEFORE.json")
        q(resource["active_target_heavy_processes"] == [], "Heavy same-study cost guard failed")
        for path, h in watched.items():
            q(digest(path) == h, "Input changed during independent data verification")
        result = dict(passed=True, mode=a.mode, source_run=source.name, source_summary_sha256=digest(source / "SUMMARY.json"),
                      source_binding_sha256=inputs["source_binding_sha256"], image_count=count, repeats=repeats,
                      fresh_children=len(children), samples=len(actual), losslessly_reconstructed_mask_rows=mask_rows,
                      empty_output_images=empty_images, empty_masks=empty_masks, statistics=statistics, children=children,
                      input_hashes_unchanged=len(watched), verifier_sha256=digest(__file__), helper_sha256=HELPER_SHA,
                      elapsed_seconds=time.perf_counter()-began, source_wall_clocks_remeasured=False,
                      fresh_forward=False, gt_parsed=False, gpu_used=False,
                      limits=["Saved clocks reaggregated; no new latency measurement", "RLE reconstructed to recorded binary hash; native execution receipts not rerun", "Memory scope from actual fresh child OS/CUDA counters, not marginal RAM minimum", "Single resource snapshot is not continuous observation", "Fixed panel/device/endpoint only; no AP or timing CI inferred"])
        helper.dump(out / "VERIFICATION.json", result)
        helper.dump(out / "SUMMARY.json", result)
    except Exception as exc:
        helper.dump(out / "FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
