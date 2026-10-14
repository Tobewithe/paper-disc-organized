"""Registered, bounded S1 execution on native caches; never runs the base model."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "experiments/mask_readout_mechanism_v2_20261012"
BASE_ID = "RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170"
BASE = ROOT / "experiments/official_gt_failure_rebaseline_20261013/runs" / BASE_ID
GT = ROOT / "assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json"
WEIGHTS = ROOT / "assets/models/coco_clean_20260911/yolo26m-seg.pt"
PROTOCOL = STUDY / "S1_EXECUTION_20261013.md"
sys.path.insert(0, str(ROOT / "shared/evaluation"))
from posttrain_audit import read_json, read_jsonl, require, sha256, write_new, make_panel, summarize_oracle
from finite_coefficient_solver import DEFAULTS, solve_coefficients, positive_control
from s1_geometry import iter_prediction_array, geometry, project_gt, predicted_roi, full_rle_iou


def now():
    return datetime.now(timezone.utc).isoformat()


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, value):
        self.original.write(value)
        self.log.write(value)
        return len(value)

    def flush(self):
        self.original.flush()
        self.log.flush()


def save_state(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    os.replace(temporary, path)


def deadline(start, seconds):
    if time.monotonic() - start > seconds:
        raise TimeoutError("frozen S1 wall-clock budget exhausted")


def known_basis_control(cache, decode_native):
    """Known +/- unit coefficients: logits are exactly +/- one cached P channel."""
    import torch
    from ultralytics.utils import ops
    count = len(cache["c0"])
    coefficients = torch.zeros_like(cache["c0"])
    signs = torch.where(torch.arange(count) % 2 == 0, 1., -1.)
    coefficients[:, 0] = signs
    actual = decode_native(cache, coefficients)
    channel = cache["proto"][0].to(cache["device"]).float()
    logits = signs.to(channel)[:, None, None] * channel[None]
    h, w = cache["native_decode_hw"]
    step = max(1, 32_000_000 // (h * w))
    expected = torch.cat([ops.scale_masks(logits[i:i + step][None], [h, w])[0].gt(0).byte()
                          for i in range(0, count, step)])
    expected = ops.crop_mask(expected, cache["boxes_input"].to(cache["device"]))
    expected = ops.scale_masks(expected[None], cache["original_hw"], ratio_pad=cache["ratio_pad"])[0].byte()
    passed = bool(torch.equal(actual, expected))
    require(passed and int(actual.sum()) > 0, "known basis coefficient decode control failed/empty")
    return {"passed": passed, "outputs": count, "foreground_pixels": int(actual.sum()),
            "definition": "alternating +e0/-e0; reference selects cached P channel without matmul"}


def evidence_manifest(run):
    records = []
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.name not in {"ARTIFACT_MANIFEST.json", "run.json"}:
            records.append({"relative_path": path.relative_to(run).as_posix(),
                            "sha256": sha256(path), "size_bytes": path.stat().st_size,
                            "storage_scope": "execution_host", "availability": "present"})
    require(sum(item["size_bytes"] for item in records) <= 1024**3, "artifact budget exceeded")
    write_new(run / "ARTIFACT_MANIFEST.json", {"artifacts": records})


def execute(scope, smoke_run=None, retry_of=None):
    run_id = "RUN_S1_" + ("SMOKE_" if scope == "smoke" else "ORACLE_") + uuid.uuid4().hex
    lock = STUDY / ".S1.lock"
    info = {"run_id": run_id, "study_id": "STUDY_MASK_READOUT_MECHANISM_V2_20261012",
            "stage_id": "S1_ENGINEERING" if scope == "smoke" else "S1",
            "host": socket.gethostname(), "pid": os.getpid(), "started_at": now()}
    with lock.open("x", encoding="utf8") as stream:
        json.dump(info, stream)
    run = STUDY / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    execution_log = (run / "execution.log").open("x", encoding="utf8", buffering=1)
    sys.stdout, sys.stderr = Tee(original_stdout, execution_log), Tee(original_stderr, execution_log)
    meta = {**info, "scope": scope, "baseline_run_id": BASE_ID,
            "execution_status": "running", "plan_status": "freezing", "scientific_status": "unadjudicated",
            "artifact_status": "partial", "transfer_status": "local_only", "exit_code": None,
            "executor": {"environment": "laptop_windows", "interpreter": sys.executable,
                         "directory": str(ROOT), "host": socket.gethostname()},
            "command": [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
            "parent_run_id": smoke_run.name if smoke_run else BASE_ID,
            "retry_of": retry_of.name if retry_of else None,
            "optimization_started": False}
    save_state(run / "run.json", meta)
    print("RUN_DIRECTORY", run, flush=True)
    start = time.monotonic()
    wall_budget = 600 if scope == "smoke" else 3600
    try:
        if retry_of:
            previous = read_json(retry_of / "run.json")
            require(previous["execution_status"] == "error" and previous["scope"] == scope,
                    "retry must identify a failed execution of the same scope")
        source_receipt = read_json(STUDY / "S1_EXECUTION_SOURCE.json")
        sources = source_receipt["files"]
        for rel, digest in sources.items():
            require(sha256(ROOT / rel) == digest, "execution source changed: " + rel)
        meta["code_commit"] = source_receipt["code_commit"]
        meta["dirty_patch_sha256"] = source_receipt["dirty_patch_sha256"]
        require(sha256(STUDY / "S1_SOURCE_DIRTY.patch") == meta["dirty_patch_sha256"], "dirty patch changed")
        shutil.copyfile(STUDY / "S1_SOURCE_DIRTY.patch", run / "dirty.patch")
        write_new(run / "SOURCE_MANIFEST.json", source_receipt)
        for index, rel in enumerate(sources):
            dest = run / "source_snapshot" / (f"{index:02d}_" + Path(rel).name)
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / rel, dest)
        shutil.copyfile(PROTOCOL, run / "PROTOCOL.md")
        meta.update(protocol_path=str(run / "PROTOCOL.md"), protocol_sha256=sha256(run / "PROTOCOL.md"))

        import psutil
        import torch
        from pycocotools.coco import COCO
        from pycocotools import mask as masks
        from native_output_bridge import (apply_runtime_policy, native_environment, decode_native,
                                          encode_rle, RUNTIME_KEYS)
        torch.set_num_threads(1)
        require(psutil.virtual_memory().available >= 4 * 1024**3, "less than 4 GiB available memory")
        require(shutil.disk_usage(run).free >= 2 * 1024**3, "less than 2 GiB free disk")
        require(torch.cuda.is_available(), "original CUDA decoder device unavailable")
        manifest_path = BASE / "trace/TRACE_MANIFEST.json"
        manifest = read_json(manifest_path)
        # Audit all recorded input geometry before selecting/optimizing any target.
        for record in manifest["images"]:
            geometry(record)
        require(read_json(BASE / "SUMMARY.json")["native_gt_ledger_validity"] == "passed", "invalid S0 ledger")
        require(read_json(BASE / "ZERO_DELTA.json")["native_zero_delta_passed"] is True, "invalid S0 replay")
        for name in ("capture_zero_delta_rle_equal", "base_state_unchanged"):
            require(manifest[name] is True, "S0 trace control failed: " + name)
        require(sha256(WEIGHTS) == manifest["model_sha256"], "weight identity changed")
        require(sha256(GT) == manifest["annotations_sha256"], "annotation identity changed")
        require(sha256(ROOT / "shared/evaluation/native_output_bridge.py") == manifest["bridge_code_sha256"],
                "native bridge differs from S0")
        apply_runtime_policy({k: manifest["environment"][k] for k in RUNTIME_KEYS})
        environment = native_environment()
        for field in (*RUNTIME_KEYS, "ultralytics_version", "torch_version", "cuda_version", "source_sha256", "packages"):
            require(environment[field] == manifest["environment"][field], "runtime differs from S0: " + field)
        write_new(run / "ENVIRONMENT.json", {**environment, "solver_device": "cpu", "solver_threads": 1,
                  "gpu": torch.cuda.get_device_name(0), "available_memory_bytes": psutil.virtual_memory().available})
        ledger_path = BASE / "ledger/PER_GT_FAILURE.jsonl"
        complete = make_panel(read_jsonl(ledger_path), sha256(ledger_path))
        require(complete["eligible_T"] == 6229 and complete["source_prediction_sha"] == manifest["predictions_sha256"],
                "S0 population/source changed")
        offset, length = (256, 4) if scope == "smoke" else (0, 256)
        panel = {**complete, "targets": complete["targets"][offset:offset + length],
                 "selection": "sha256_image_id_colon_annotation_id", "selection_offset": offset,
                 "limit": length, "selected_T": length}
        require(len(panel["targets"]) == length, "incomplete frozen panel")
        write_new(run / "TARGET_PANEL.json", panel)
        selected_images = {row["image_id"] for row in panel["targets"]}
        image_records = [item for item in manifest["images"] if item["image_id"] in selected_images]
        require(len(image_records) == len(selected_images), "missing source cache entries")
        predictions_path = Path(manifest["predictions_path"])
        require(sha256(predictions_path) == manifest["predictions_sha256"], "native predictions changed")
        input_record = {"baseline_run_id": BASE_ID, "manifest_sha256": sha256(manifest_path),
                        "predictions_sha256": manifest["predictions_sha256"], "ledger_sha256": sha256(ledger_path),
                        "annotations_sha256": sha256(GT), "model_sha256": sha256(WEIGHTS),
                        "cache_root": str(manifest_path.parent), "images": image_records}
        for record in image_records:
            path = (manifest_path.parent / record["cache"]).resolve()
            require(path.is_relative_to(manifest_path.parent.resolve()), "cache escaped root")
            require(sha256(path) == record["cache_sha256"], "cache changed")
        write_new(run / "INPUT_MANIFEST.json", input_record)
        oracle_config = {"solver": DEFAULTS, "solver_device": "cpu", "solver_threads": 1,
                         "gt_source": "original_COCO_annToRLE",
                         "gt_projection": "recorded_per_axis_resize_nearest_then_integer_zero_pad_then_proto_nearest",
                         "loss_roi": "predicted_box_half_open_integer_support_on_proto_grid",
                         "decode": "S0_full_image_coefficient_matrix_float32_CUDA_native",
                         "score": "original_full_GT_RLE_IoU", "runtime_policy": {k: environment[k] for k in RUNTIME_KEYS}}
        write_new(run / "ORACLE_CONFIG.json", oracle_config)
        contract = {"schema_version": 1, "run_id": run_id, "stage_id": info["stage_id"],
                    "baseline_run_id": BASE_ID, "candidate_policy": "fixed_output_detection",
                    "scope": "native_config_scoped", "panel_sha256": sha256(run / "TARGET_PANEL.json"),
                    "input_manifest_sha256": sha256(run / "INPUT_MANIFEST.json"),
                    "oracle_config_sha256": sha256(run / "ORACLE_CONFIG.json"),
                    "selected_T": length, "wall_budget_seconds": wall_budget, "max_output_bytes": 1024**3,
                    "main_metric": "same_detection_mask_iou_ge_0.75_fraction", "unit": "fraction_0_to_1",
                    "min_formal_samples": 200, "opportunity_threshold": .20,
                    "stop": "numerical_error_blocks; insufficient_found_with_unconverged_blocks; valid_below_threshold_stops_c_only"}
        write_new(run / "EVAL_CONTRACT.json", contract)
        meta.update(plan_status="frozen", input_manifest_sha256=contract["input_manifest_sha256"],
                    contract_sha256=sha256(run / "EVAL_CONTRACT.json"), phase="controls")
        save_state(run / "run.json", meta)
        if scope == "full":
            require(smoke_run is not None, "formal S1 requires real engineering smoke")
            require(read_json(smoke_run / "DECISION.json")["status"] == "passed", "S1 smoke did not pass")
            require(read_json(smoke_run / "SOURCE_MANIFEST.json") == source_receipt, "source changed after smoke")
            require(sha256(smoke_run / "ORACLE_CONFIG.json") == contract["oracle_config_sha256"], "config changed after smoke")
            for prior in (STUDY / "runs").glob("RUN_S1_ORACLE_*/run.json"):
                if prior.parent != run:
                    require(not read_json(prior).get("optimization_started"), "formal S1 budget already used: " + str(prior))
        else:
            with (run / "geometry_tests.log").open("x", encoding="utf8") as stream:
                checked = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("test_s1_geometry.py"))],
                                         stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
            require(checked.returncode == 0, "S1 geometry tests failed")
        positive = positive_control(DEFAULTS)
        write_new(run / "POSITIVE_CONTROL.json", positive)
        require(positive["passed"], "same-config solver positive control failed")
        selected_predictions = {}
        expected_indices = {index for record in image_records for index in record["prediction_indices"]}
        for index, prediction in enumerate(iter_prediction_array(predictions_path)):
            if index in expected_indices:
                selected_predictions[index] = prediction
        require(set(selected_predictions) == expected_indices, "selected native predictions missing")
        coco = COCO(str(GT))
        results, image_checks = optimize_panel(run, meta, panel, manifest_path, image_records,
                                             selected_predictions, coco, contract, start, wall_budget)
        controls = {"schema_version": 1, "positive_control_passed": positive["passed"],
                    "same_solver_config": True, "native_zero_delta_passed": all(r["zero_delta"] for r in image_checks),
                    "known_coefficient_mask_control_passed": all(r["basis_control"]["passed"] for r in image_checks),
                    "solver_code_sha256": sha256(Path(__file__).with_name("finite_coefficient_solver.py")),
                    "oracle_config_sha256": contract["oracle_config_sha256"], "images": image_checks}
        write_new(run / "CONTROLS.json", controls)
        summary = summarize_oracle(panel, results, controls)
        summary.update(run_id=run_id, baseline_run_id=BASE_ID, attempted=len(results), completed=len(results),
                       elapsed_seconds=time.monotonic() - start, protocol_sha256=meta["protocol_sha256"])
        summary["by_baseline_gap_to_075"] = {
            label: {"denominator": sum(lo <= r["baseline_mask_iou"] < hi for r in results),
                    "found": sum(lo <= r["baseline_mask_iou"] < hi and r.get("oracle_mask_iou") is not None
                                 and r["oracle_mask_iou"] >= .75 for r in results)}
            for label, lo, hi in (("iou_below_050", 0, .5), ("iou_050_to_065", .5, .65), ("iou_065_to_075", .65, .75))}
        for rel, digest in sources.items():
            require(sha256(ROOT / rel) == digest, "source mutated while running: " + rel)
        require(sha256(ledger_path) == input_record["ledger_sha256"], "ledger mutated while running")
        write_new(run / "SUMMARY.json", summary)
        gate = summary["gate_candidate_status"]
        if scope == "smoke":
            status = "passed" if not summary["solver_status_counts"].get("numerical_error", 0) else "blocked"
            next_action = "Execute frozen 256-GT formal S1" if status == "passed" else "Inspect engineering failure"
            scientific = "not_adjudicated_engineering_only"
        else:
            status = "passed" if gate == "opportunity_threshold_met" else "failed" if gate == "opportunity_threshold_not_met" else "blocked"
            next_action = "Prepare and execute authorized S2 fit2000 B/C pilot" if status == "passed" else "STOP_C_ONLY_OUTPUT_RESCUE" if status == "failed" else "Inspect numerical validity/budget evidence; do not train"
            scientific = status
        write_new(run / "DECISION.json", {"gate_id": "S1_ENGINEERING" if scope == "smoke" else "G1_FIXED_OUTPUT_ORACLE",
                  "status": status, "gate_candidate_status": gate, "scientific_status": scientific,
                  "claim_scope": "engineering_only" if scope == "smoke" else "finite_found_same_detection_repair_lower_bound",
                  "evidence_refs": {p: sha256(run / p) for p in ("SUMMARY.json", "CONTROLS.json", "EVAL_CONTRACT.json", "ORACLE_RESULTS.jsonl")},
                  "next_action": next_action})
        meta.update(execution_status="completed", scientific_status=scientific, artifact_status="complete", exit_code=0)
        print("S1_RESULT", json.dumps({"run_id": run_id, "status": status, "found": summary["found_same_detection_mask75"],
                                      "selected": len(results), "solver": summary["solver_status_counts"]}), flush=True)
    except BaseException as exc:
        (run / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf8")
        meta.update(execution_status="error", scientific_status="blocked_engineering", exit_code=1, error=str(exc))
        if not (run / "DECISION.json").exists():
            write_new(run / "DECISION.json", {"status": "blocked", "reason": str(exc), "next_action": "Inspect retained error; no training"})
        traceback.print_exc()
    finally:
        meta.update(ended_at=now(), elapsed_seconds=time.monotonic() - start)
        save_state(run / "run.json", meta)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = original_stdout, original_stderr
        execution_log.close()
        evidence_manifest(run)
        lock.unlink()
    return meta["exit_code"]


def optimize_panel(run, meta, panel, manifest_path, image_records, predictions, coco, contract, start, wall_budget):
    """Defined separately so all input freezing above completes before any target fit."""
    import torch
    from pycocotools import mask as masks
    from native_output_bridge import decode_native, encode_rle
    by_image = defaultdict(list)
    for target in panel["targets"]:
        by_image[target["image_id"]].append(target)
    solver_sha = sha256(Path(__file__).with_name("finite_coefficient_solver.py"))
    results, checks = [], []
    meta.update(optimization_started=True, phase="finite_oracle", completed_targets=0)
    save_state(run / "run.json", meta)
    with (run / "ORACLE_RESULTS.jsonl").open("x", encoding="utf8") as output:
        for record in image_records:
            deadline(start, wall_budget)
            path = manifest_path.parent / record["cache"]
            cache = torch.load(path, map_location="cpu", weights_only=True)
            require(cache["image_id"] == record["image_id"] and cache["prediction_indices"] == record["prediction_indices"],
                    "cache output identity changed")
            require(list(cache["input_hw"]) == record["input_hw"] and list(cache["original_hw"]) == record["original_hw"],
                    "cache input geometry changed")
            ranks = {index: rank for rank, index in enumerate(cache["prediction_indices"])}
            native = decode_native(cache)
            for rank, index in enumerate(cache["prediction_indices"]):
                require(predictions[index]["image_id"] == cache["image_id"], "prediction/cache image changed")
                require(encode_rle(native[rank]) == predictions[index]["segmentation"], "native zero-delta RLE mismatch")
            del native
            basis = known_basis_control(cache, decode_native)
            coefficient_matrix = cache["c0"].clone()
            pending = []
            for target in by_image[cache["image_id"]]:
                deadline(start, wall_budget)
                prefix, suffix = target["output_detection_id"].rsplit(":", 1)
                require(prefix == panel["source_prediction_sha"], "target prediction SHA mismatch")
                index = int(suffix)
                require(index in ranks, "target is not an output of this image")
                rank = ranks[index]
                annotation = coco.anns[target["annotation_id"]]
                require(annotation["image_id"] == cache["image_id"] and annotation["category_id"] == target["category_id"],
                        "target/annotation identity mismatch")
                gt_rle = coco.annToRLE(annotation)
                baseline_iou = full_rle_iou(predictions[index]["segmentation"], gt_rle)
                require(abs(baseline_iou - target["baseline_mask_iou"]) <= 1e-12, "baseline full-GT IoU differs from ledger")
                row = {**target, "source_prediction_sha": panel["source_prediction_sha"], "baseline_replay_equal": True,
                       "solver_code_sha256": solver_sha, "oracle_config_sha256": contract["oracle_config_sha256"],
                       "cache_sha256": record["cache_sha256"], "rank": rank, "geometry": geometry(cache)}
                try:
                    roi = predicted_roi(cache, rank)
                    require(bool(roi.any()), "empty fixed predicted ROI")
                    projected = project_gt(masks.decode(gt_rle), cache)
                    design = cache["proto"].permute(1, 2, 0)[roi]
                    solved = solve_coefficients(design, projected[roi], cache["c0"][rank], DEFAULTS)
                    coefficients = solved.pop("coefficients")
                    delta = solved.pop("delta")
                    coefficient_matrix[rank] = coefficients.to(coefficient_matrix)
                    row.update(solved, coefficients_fp32=coefficient_matrix[rank].tolist(),
                               delta_float64=delta.tolist(), roi_pixels=int(roi.sum()), gt_roi_pixels=int(projected[roi].sum()))
                except (ValueError, FloatingPointError) as exc:
                    row.update(solver_status="numerical_error", numerical_error=str(exc), oracle_mask_iou=None,
                               delta_norm=None, gradient_inf_norm=None)
                pending.append((row, gt_rle))
            modified = decode_native(cache, coefficient_matrix)
            changed_ranks = {row["rank"] for row, _ in pending}
            for rank, index in enumerate(cache["prediction_indices"]):
                if rank not in changed_ranks:
                    require(encode_rle(modified[rank]) == predictions[index]["segmentation"],
                            "unmodified detection mask changed")
            for row, gt_rle in pending:
                if row["solver_status"] != "numerical_error":
                    rle = encode_rle(modified[row["rank"]])
                    row.update(oracle_mask_iou=full_rle_iou(rle, gt_rle), oracle_rle=rle)
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                results.append(row)
            del modified
            require(sha256(path) == record["cache_sha256"], "cache mutated while solving")
            checks.append({"image_id": cache["image_id"], "zero_delta": True, "outputs": len(cache["c0"]), "basis_control": basis})
            meta["completed_targets"] = len(results)
            save_state(run / "run.json", meta)
            if len(results) % 8 == 0 or len(results) == len(panel["targets"]):
                print(now(), "TARGETS", len(results), "/", len(panel["targets"]), "elapsed", round(time.monotonic() - start, 1), flush=True)
            del cache
    return results, checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("smoke", "full"), required=True)
    parser.add_argument("--smoke-run", type=Path)
    parser.add_argument("--retry-of", type=Path)
    args = parser.parse_args()
    raise SystemExit(execute(args.scope, args.smoke_run, args.retry_of))
