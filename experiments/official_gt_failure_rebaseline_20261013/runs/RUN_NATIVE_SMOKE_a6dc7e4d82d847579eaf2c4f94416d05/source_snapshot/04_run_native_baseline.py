"""One registered native smoke/full execution, followed by saved-output verification.

No parameter search, training, automatic retry, or second full forward. A full run
requires a successful smoke with the identical profile and source manifest.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "experiments/official_gt_failure_rebaseline_20261013"
sys.path.insert(0, str(ROOT / "shared/evaluation"))
from posttrain_audit import read_json, require, sha256, write_new

PROFILE = ROOT / "paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json"
DATA = ROOT / "assets/datasets/coco_native_s0_20261013"
WEIGHTS = ROOT / "assets/models/coco_clean_20260911/yolo26m-seg.pt"
CODE_FILES = [
    "shared/evaluation/native_output_bridge.py", "shared/evaluation/native_source_pins.json",
    "shared/evaluation/posttrain_audit.py",
    "experiments/official_gt_failure_rebaseline_20261013/scripts/build_native_ledger.py",
    "experiments/official_gt_failure_rebaseline_20261013/scripts/run_native_baseline.py",
    "experiments/official_gt_failure_rebaseline_20261013/scripts/prepare_native_inputs.py",
    "shared/evaluation/tests/test_posttrain_audit.py", "shared/evaluation/tests/test_posttrain_completion.py",
    "shared/evaluation/tests/test_native_baseline_receipt.py",
    "experiments/official_gt_failure_rebaseline_20261013/scripts/run_engineering_checks.py",
    "experiments/official_gt_failure_rebaseline_20261013/scripts/check_native_configuration.py",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def save_state(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    os.replace(temp, path)


def call(command, log, meta, phase, run):
    print(now(), phase, "START", flush=True)
    meta["phase"] = phase
    meta["commands"].append({"phase": phase, "command": command, "log": log.name, "started_at": now()})
    save_state(run / "run.json", meta)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
           "YOLO_AUTOINSTALL": "false", "YOLO_OFFLINE": "true", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    with log.open("x", encoding="utf8") as f:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    meta["commands"][-1].update(exit_code=result.returncode, ended_at=now())
    save_state(run / "run.json", meta)
    require(result.returncode == 0, phase + " failed; inspect " + str(log))
    print(now(), phase, "PASS", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", choices=["smoke", "full"], required=True)
    p.add_argument("--smoke-run", type=Path)
    p.add_argument("--engineering-run", type=Path, required=True)
    a = p.parse_args()
    profile = read_json(PROFILE)
    sources = {f: sha256(ROOT / f) for f in CODE_FILES}
    engineering = read_json(a.engineering_run / "run.json")
    require(engineering["exit_code"] == 0 and len(engineering["commands"]) == 4,
            "strict engineering checks must pass first")
    if a.scope == "full":
        require(a.smoke_run is not None, "full requires a smoke Run")
        smoke = read_json(a.smoke_run / "DECISION.json")
        require(smoke["gate_id"] == "G1" and smoke["status"] == "passed", "smoke gate not passed")
        require(read_json(a.smoke_run / "SOURCE_MANIFEST.json") == sources, "code changed after smoke")
        require(sha256(a.smoke_run / "baseline_profile.json") == sha256(PROFILE), "profile changed after smoke")
        # Count attempted full forwards, including failed attempts. Never silently rerun the full set.
        for old in (STUDY / "runs").glob("*/run.json"):
            item = read_json(old)
            require(not (item.get("scope") == "full" and any(c.get("phase") == "native_trace"
                         for c in item.get("commands", []))), "full forward budget already used: " + str(old))
    run_id = "RUN_NATIVE_" + a.scope.upper() + "_" + uuid.uuid4().hex
    lock = STUDY / ".S0.lock"
    lock_record = {"run_id": run_id, "study_id": read_json(STUDY / "study.json")["study_id"],
                   "stage_id": "S0", "host": socket.gethostname(), "pid": os.getpid(), "started_at": now()}
    with lock.open("x", encoding="utf8") as f:
        json.dump(lock_record, f)
    run = STUDY / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    print("RUN_DIRECTORY", run, flush=True)
    source_receipt = read_json(ROOT / "EXECUTION_SOURCE.json") if (ROOT / "EXECUTION_SOURCE.json").exists() else None
    code_commit = source_receipt["code_commit"] if source_receipt else subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    meta = {**lock_record, "scope": a.scope, "executor": {"environment": profile.get("executor", {}).get("environment", "local_windows"),
            "interpreter": sys.executable, "host": socket.gethostname()}, "execution_status": "running",
            "plan_status": "frozen", "scientific_status": "unadjudicated", "artifact_status": "partial",
            "transfer_status": "local_only", "commands": [], "phase": "preflight", "exit_code": None,
            "parent_run_id": a.smoke_run.name if a.smoke_run else a.engineering_run.name,
            "code_commit": code_commit}
    save_state(run / "run.json", meta)
    try:
        for source in [PROFILE, STUDY / "PROTOCOL.md"]:
            shutil.copyfile(source, run / source.name)
        meta["protocol_path"] = str(STUDY / "PROTOCOL.md")
        meta["protocol_sha256"] = sha256(run / "PROTOCOL.md")
        patch = (ROOT / "SOURCE_DIRTY.patch").read_bytes() if source_receipt else subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=ROOT)
        (run / "dirty.patch").write_bytes(patch)
        meta["dirty_patch_sha256"] = sha256(run / "dirty.patch")
        write_new(run / "SOURCE_MANIFEST.json", sources)
        snapshot_map = {}
        for index, rel in enumerate(CODE_FILES):
            # Keep Windows paths bounded even under a long Study/Run prefix.
            dest = run / "source_snapshot" / (f"{index:02d}_" + Path(rel).name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, dest)
            snapshot_map[rel] = dest.relative_to(run).as_posix()
        write_new(run / "SOURCE_SNAPSHOT_MAP.json", snapshot_map)
        import ultralytics
        installed = Path(ultralytics.__file__).parent
        vendor = ROOT / "shared/vendor/ultralytics_8_4_100/ultralytics"
        package_sources = {}
        for source in sorted(vendor.rglob("*")):
            if source.is_file() and source.suffix in {".py", ".yaml", ".yml"}:
                rel = source.relative_to(vendor)
                expected = source.read_text(encoding="utf8")
                require((installed / rel).read_text(encoding="utf8") == expected, "installed source drift: " + str(rel))
                package_sources[rel.as_posix()] = hashlib.sha256(expected.encode("utf8")).hexdigest()
        require(len(package_sources) >= 342, "incomplete vendor source comparison")
        write_new(run / "ULTRALYTICS_SOURCE_MANIFEST.json", package_sources)
        shutil.copyfile(ROOT / "assets/dependencies/baseline_20261013/DOWNLOAD_RECEIPTS.json", run / "DOWNLOAD_RECEIPTS.json")
        require(shutil.disk_usage(run).free > 65 * 1024**3, "insufficient disk for bounded cache and outputs")
        import psutil
        require(psutil.virtual_memory().available >= 6 * 1024**3, "less than 6 GiB available RAM; release competing work first")
        require(sha256(WEIGHTS) == profile["model_sha256"], "model changed")
        gt = DATA / "annotations/instances_val2017.json"
        require(sha256(gt) == profile["annotations_sha256"], "GT changed")
        preparation = read_json(DATA / "PREPARATION.json")
        require(sha256(DATA / "IMAGE_MANIFEST.jsonl") == preparation["manifest_sha256"], "manifest changed")
        expected_images = {}
        for line in (DATA / "IMAGE_MANIFEST.jsonl").read_text(encoding="utf8").splitlines():
            record = json.loads(line)
            expected_images[record["image_id"]] = record["image_sha256"]
            require(sha256(DATA / "images/val2017" / record["file_name"]) == record["image_sha256"], "image changed")
            label = DATA / "labels/val2017" / Path(record["file_name"]).with_suffix(".txt").name
            require((sha256(label) if label.exists() else None) == record["label_sha256"], "label changed")
        ids_path = run / "image_ids.json"
        shutil.copyfile(DATA / a.scope / "image_ids.json", ids_path)
        shutil.copyfile(DATA / "PREPARATION.json", run / "DATA_PREPARATION.json")
        meta["input_manifest_sha256"] = preparation["manifest_sha256"]
        args = {**profile["native_args"], "data": str((DATA / a.scope / "coco.yaml").resolve())}
        write_new(run / "REQUESTED_ARGS.json", args)
        write_new(run / "PREFLIGHT.json", {
            "profile_sha256": sha256(PROFILE), "model_sha256": sha256(WEIGHTS),
            "annotations_sha256": sha256(gt), "image_ids_sha256": sha256(ids_path),
            "all_5000_image_and_label_hashes_verified": True,
            "engineering_run": str(a.engineering_run.resolve()),
            "engineering_run_sha256": sha256(a.engineering_run / "run.json"),
            "disk_free_bytes": shutil.disk_usage(run).free,
            "packages": {x: importlib.metadata.version(x) for x in profile["packages"]},
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"], text=True).strip()})
        call([sys.executable, "-B", str(ROOT / "shared/evaluation/native_output_bridge.py"), "trace",
              "--weights", str(WEIGHTS), "--annotations", str(gt), "--args", str(run / "REQUESTED_ARGS.json"),
              "--image-ids", str(ids_path), "--scope", a.scope, "--max-cache-bytes", str(profile["budget"]["cache_bytes"]),
              "--profile", str(run / "baseline_profile.json"), "--out", str(run / "trace")],
             run / "native_trace.log", meta, "native_trace", run)
        trace = read_json(run / "trace/TRACE_MANIFEST.json")
        require(all(r["image_sha256"] == expected_images[r["image_id"]] for r in trace["images"]),
                "native dataset verification changed an image after preflight")
        predictions = Path(trace["predictions_path"])
        call([sys.executable, "-B", str(ROOT / "shared/evaluation/native_output_bridge.py"), "replay",
              "--manifest", str(run / "trace/TRACE_MANIFEST.json"), "--predictions", str(predictions),
              "--out", str(run / "predictions_zero_delta.json"), "--report", str(run / "ZERO_DELTA.json")],
             run / "zero_delta.log", meta, "zero_delta", run)
        contract = {
            "run_id": run_id, "profile_id": profile["profile_id"], "code_commit": meta["code_commit"],
            "profile_sha256": sha256(run / "baseline_profile.json"),
            "ultralytics_version": profile["ultralytics_version"], "model_sha256": profile["model_sha256"],
            "annotations_sha256": sha256(gt), "predictions_sha256": sha256(predictions),
            "image_ids_sha256": sha256(ids_path), "input_geometry_manifest_sha256": sha256(run / "trace/TRACE_MANIFEST.json"),
            "end2end": True, "agnostic_nms": False, "conf": .001, "max_det": 300,
            "candidate_policy": "inference_output", "gt_assisted_inference": False,
            "decoder": "v8.4.100 process_mask_native -> scale_masks(binary, ratio_pad).byte",
            "native_args": trace["native_args"], "environment": trace["environment"],
            "detection_area_policy": "bbox_area_from_complete_native_JSON_for_both_tasks",
            "geometry_shape_counts": dict(Counter(str(r["input_hw"]) for r in trace["images"])),
            "acceptance": profile["acceptance"]}
        write_new(run / "EVAL_CONTRACT.json", contract)
        call([sys.executable, "-B", str(STUDY / "scripts/build_native_ledger.py"), "--annotations", str(gt),
              "--predictions", str(predictions), "--image-ids", str(ids_path), "--contract", str(run / "EVAL_CONTRACT.json"),
              "--scope", a.scope, "--native-summary", str(run / "trace/native_eval_summary.json"), "--out", str(run / "ledger")],
             run / "ledger.log", meta, "independent_coco_ledger", run)
        summary = read_json(run / "ledger/SUMMARY.json")
        require(summary["coco_backend_parity"] == "passed", "COCO backend mismatch")
        summary["native_gt_ledger_validity"] = "passed" if a.scope == "full" else "not_applicable_smoke"
        summary["native_smoke_validity"] = "passed" if a.scope == "smoke" else "inherited_passed"
        summary["run_id"] = run_id
        summary["native_trace_and_zero_delta_verified"] = True
        summary["source_evidence"] = {"trace": "trace/TRACE_MANIFEST.json", "replay": "ZERO_DELTA.json",
                                      "native_metrics": "trace/native_eval_summary.json", "offline": "ledger/SUMMARY.json"}
        write_new(run / "SUMMARY.json", summary)
        write_new(run / "DECISION.json", {
            "gate_id": "G1" if a.scope == "smoke" else "G2_G3", "status": "passed",
            "published_baseline_parity": summary["published_baseline_parity"],
            "native_gt_ledger_validity": summary["native_gt_ledger_validity"],
            "claim_scope": "engineering_smoke" if a.scope == "smoke" else (
                "published_AP_display_parity_and_native_ledger" if summary["published_baseline_parity"] == "passed"
                else "native_config_scoped_published_parity_failed"),
            "reason": "Strict source/identity checks, exact cache replay and independent 24-metric backend agreement passed.",
            "evidence_refs": {f: sha256(run / f) for f in ("SUMMARY.json", "ZERO_DELTA.json", "EVAL_CONTRACT.json")},
            "next_action": "one frozen full validation" if a.scope == "smoke" else "offline historical GT reconciliation and report; no additional forward or training"})
        meta.update(execution_status="completed", exit_code=0, artifact_status="complete",
                    scientific_status="engineering_passed" if a.scope == "smoke" else summary["published_baseline_parity"])
    except BaseException as e:
        meta.update(execution_status="failed", exit_code=1, error=str(e))
        (run / "failure.log").write_text(traceback.format_exc(), encoding="utf8")
        print(traceback.format_exc(), flush=True)
    finally:
        meta["ended_at"] = now()
        save_state(run / "run.json", meta)
        manifest = []
        for path in run.rglob("*"):
            if path.is_file():
                manifest.append({"relative_path": path.relative_to(run).as_posix(), "sha256": sha256(path),
                                 "size_bytes": path.stat().st_size, "storage_scope": "local", "availability": "present"})
        write_new(run / "ARTIFACT_MANIFEST.json", manifest)
        require(read_json(lock)["run_id"] == run_id, "stage lock ownership changed")
        lock.unlink()
    print("FINAL_STATUS", json.dumps({"run_id": run_id, "execution_status": meta["execution_status"],
                                      "exit_code": meta["exit_code"]}), flush=True)
    return meta["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
