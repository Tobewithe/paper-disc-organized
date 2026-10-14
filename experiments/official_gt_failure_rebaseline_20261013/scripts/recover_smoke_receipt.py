"""Recover the completed smoke's missing COCO receipt without another model forward.

This narrow recovery is for the result-hook dispatch failure. Original failed Run
and predictions remain immutable. The recorded bridge is used for cache replay.
"""
import argparse
import ast
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared/evaluation"))
from posttrain_audit import read_json, require, sha256, write_new
from native_output_bridge import make_trace_validator, apply_runtime_policy, RUNTIME_KEYS
from run_native_baseline import CODE_FILES, DATA, PROFILE, STUDY, call, save_state, now


def method_ast(path, name):
    matches = [n for n in ast.walk(ast.parse(path.read_text(encoding="utf8")))
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    require(len(matches) == 1, "ambiguous source method: " + name)
    return ast.dump(matches[0], include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--engineering-run", type=Path, required=True)
    a = parser.parse_args()
    parent = a.parent.resolve()
    old_meta = read_json(parent / "run.json")
    require(old_meta["scope"] == "smoke" and old_meta["execution_status"] == "failed", "not a failed smoke")
    require("AttributeError: 'TraceValidator' object has no attribute 'native_coco_receipts'" in
            (parent / "native_trace.log").read_text(encoding="utf8"), "not the specific receipt failure")
    require(read_json(a.engineering_run / "run.json")["exit_code"] == 0, "repaired native entry tests must pass")
    trace_path = parent / "trace/TRACE_MANIFEST.json"
    trace = read_json(trace_path)
    require(trace["scope"] == "smoke" and len(trace["image_ids"]) <= 16 and trace["all_output_rows_traced"]
            and trace["capture_zero_delta_rle_equal"] and trace["base_state_unchanged"], "incomplete forward evidence")
    require(sha256(parent / "baseline_profile.json") == sha256(PROFILE), "profile changed since forward")
    snapshot_map = read_json(parent / "SOURCE_SNAPSHOT_MAP.json")
    original_bridge = parent / snapshot_map["shared/evaluation/native_output_bridge.py"]
    require(sha256(original_bridge) == trace["bridge_code_sha256"], "original bridge snapshot changed")
    unchanged = ("decode_native", "encode_rle", "raw_location", "init_metrics", "postprocess",
                 "update_metrics", "__call__", "trace", "replay")
    for name in unchanged:
        require(method_ast(original_bridge, name) == method_ast(ROOT / "shared/evaluation/native_output_bridge.py", name),
                "recovery cannot reuse a smoke after forward/replay code changes: " + name)
    run = STUDY / "runs" / ("RUN_SMOKE_RECEIPT_RECOVERY_" + uuid.uuid4().hex)
    run.mkdir(parents=True)
    meta = {"run_id": run.name, "study_id": old_meta["study_id"], "stage_id": "S0_G1",
            "scope": "smoke", "execution_kind": "saved_output_receipt_recovery_no_model_forward",
            "execution_status": "running", "started_at": now(), "parent_run_id": parent.name,
            "executor": {"host": socket.gethostname(), "interpreter": sys.executable, "environment": "laptop_windows"},
            "protocol_path": old_meta["protocol_path"], "protocol_sha256": old_meta["protocol_sha256"],
            "code_commit": old_meta["code_commit"], "commands": [], "source_script_sha256": sha256(__file__),
            "scientific_status": "unadjudicated", "transfer_status": "local_only"}
    save_state(run / "run.json", meta)
    print("RECOVERY_RUN", run, flush=True)
    try:
        write_new(run / "SOURCE_MANIFEST.json", {f: sha256(ROOT / f) for f in CODE_FILES})
        shutil.copyfile(PROFILE, run / "baseline_profile.json")
        shutil.copyfile(parent / "image_ids.json", run / "image_ids.json")
        write_new(run / "FORWARD_REUSE_PROOF.json", {
            "parent_run": str(parent), "trace_manifest_sha256": sha256(trace_path),
            "original_bridge_sha256": sha256(original_bridge), "unchanged_method_ASTs": list(unchanged),
            "changed_behavior": "capture evaluator via actual Segment.eval_json entry; no prediction changes",
            "engineering_run": str(a.engineering_run), "engineering_run_sha256": sha256(a.engineering_run / "run.json")})
        predictions = Path(trace["predictions_path"])
        require(sha256(predictions) == trace["predictions_sha256"], "predictions changed")
        gt = DATA / "annotations/instances_val2017.json"
        require(sha256(gt) == trace["annotations_sha256"], "GT changed")
        apply_runtime_policy({k: trace["environment"][k] for k in RUNTIME_KEYS})
        cls = make_trace_validator(run, trace["image_ids"], "smoke", 1, sha256(gt))
        v = object.__new__(cls)
        v.args = SimpleNamespace(**trace["native_args"])
        v.save_dir, v.data = predictions.parent, {"path": DATA}
        v.is_coco, v.is_lvis, v.jdict = True, False, read_json(predictions)
        v.dataloader = SimpleNamespace(dataset=SimpleNamespace(im_files=[f"{i}.jpg" for i in trace["image_ids"]]))
        # Execute the real inherited Segment.eval_json path on the saved native JSON, with repaired observation.
        v.eval_json({})
        native = {"backend": "COCOeval_faster", "unit": "AP_points",
                  "backend_version": importlib.metadata.version("faster-coco-eval"),
                  "annotations_sha256": sha256(gt), "predictions_sha256": sha256(predictions),
                  "image_ids_sha256": sha256(run / "image_ids.json"),
                  **{task: receipt["metrics"] for task, receipt in v.native_coco_receipts.items()},
                  "evaluator_receipts": v.native_coco_receipts,
                  "capture": "offline native Segment.eval_json of original smoke JSON after receipt-hook repair"}
        write_new(run / "native_eval_summary.json", native)
        replay_code = run / "recorded_bridge"
        replay_code.mkdir()
        for rel in ("shared/evaluation/native_output_bridge.py", "shared/evaluation/native_source_pins.json",
                    "shared/evaluation/posttrain_audit.py"):
            shutil.copyfile(parent / snapshot_map[rel], replay_code / Path(rel).name)
        call([sys.executable, "-B", str(replay_code / "native_output_bridge.py"), "replay",
              "--manifest", str(trace_path), "--predictions", str(predictions),
              "--out", str(run / "predictions_zero_delta.json"), "--report", str(run / "ZERO_DELTA.json")],
             run / "zero_delta.log", meta, "recorded_bridge_zero_delta", run)
        contract = {"run_id": run.name, "parent_run_id": parent.name, "code_commit": meta["code_commit"],
                    "ultralytics_version": "8.4.100", "model_sha256": trace["model_sha256"],
                    "annotations_sha256": sha256(gt), "predictions_sha256": sha256(predictions),
                    "image_ids_sha256": sha256(run / "image_ids.json"),
                    "input_geometry_manifest_sha256": sha256(trace_path), "end2end": True,
                    "agnostic_nms": False, "conf": .001, "max_det": 300,
                    "candidate_policy": "inference_output", "gt_assisted_inference": False,
                    "decoder": "v8.4.100 process_mask_native -> scale_masks(binary, ratio_pad).byte",
                    "native_args": trace["native_args"], "environment": trace["environment"]}
        write_new(run / "EVAL_CONTRACT.json", contract)
        call([sys.executable, "-B", str(STUDY / "scripts/build_native_ledger.py"), "--annotations", str(gt),
              "--predictions", str(predictions), "--image-ids", str(run / "image_ids.json"),
              "--contract", str(run / "EVAL_CONTRACT.json"), "--scope", "smoke",
              "--native-summary", str(run / "native_eval_summary.json"), "--out", str(run / "ledger")],
             run / "ledger.log", meta, "independent_coco_ledger", run)
        summary = read_json(run / "ledger/SUMMARY.json")
        require(summary["coco_backend_parity"] == "passed", "backend parity failed")
        summary.update(native_gt_ledger_validity="not_applicable_smoke", native_smoke_validity="passed",
                       run_id=run.name, parent_run_id=parent.name, model_forward_runs_in_this_recovery=0)
        write_new(run / "SUMMARY.json", summary)
        write_new(run / "DECISION.json", {"gate_id": "G1", "status": "passed",
                  "claim_scope": "smoke_engineering_only_with_recorded_forward_and_repaired_receipt",
                  "reason": "The 16-image native forward, exact recorded-cache replay and 24-metric dual-backend check passed.",
                  "evidence_refs": {f: sha256(run / f) for f in ("FORWARD_REUSE_PROOF.json", "SUMMARY.json", "ZERO_DELTA.json")},
                  "next_action": "one full validation using the same profile and repaired observation hook"})
        meta.update(execution_status="completed", exit_code=0, scientific_status="engineering_passed", artifact_status="complete")
    except BaseException as e:
        import traceback
        (run / "failure.log").write_text(traceback.format_exc(), encoding="utf8")
        meta.update(execution_status="failed", exit_code=1, error=str(e), artifact_status="partial")
        print(traceback.format_exc(), flush=True)
    finally:
        meta["ended_at"] = now()
        save_state(run / "run.json", meta)
        write_new(run / "ARTIFACT_MANIFEST.json", [
            {"relative_path": p.relative_to(run).as_posix(), "sha256": sha256(p), "size_bytes": p.stat().st_size}
            for p in run.rglob("*") if p.is_file()])
    print("FINAL_STATUS", meta["execution_status"], run.name, flush=True)
    return meta["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
