"""Finite continuation of literal approved inference02, then serial CPU stages.

This is one recorded dependency-following execution, not an automation/scheduler.
Failed source/stages stop; partial Runs remain; retries require fresh identifiers.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from segrefiner_scoring_bridge import SEG_RUN_ID, ARMS, check, read, runner_gate, sha, write


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def execute(plan, plan_path, stage, name, extra=()):
    out = Path(plan["study"]) / "runs" / name
    check(not out.exists(), "A scientific stage retry needs a new Run: " + name)
    script = Path(__file__).with_name("score_segrefiner_comparison.py")
    helper = Path(__file__).with_name("segrefiner_scoring_bridge.py")
    complete = "SCORE_COMPLETE.json" if stage in ("score", "engineering") else "WRAPPER_COMPLETE.json" if stage == "readout" else "VERIFY_COMPLETE.json"
    command = [sys.executable, plan["runner"], "--study", "STUDY_SEGREFINER_OFFICIAL_CONTENT_BASELINE_20261009",
               "--output", str(out), "--run-id", name, "--cwd", plan["workspace"], "--input", str(plan_path),
               "--snapshot", str(script), "--snapshot", str(helper), "--scope",
               json.dumps({"environment": "local_desktop_CPU", "stage": stage, "three_actual_sources": True, "GPU_or_model_execution": False}),
               "--metrics", str(out / "SUMMARY.json")]
    for name in ("SUMMARY.json", complete, "SOURCE_LOCK.json"):
        command += ["--expect", str(out / name)]
    command += ["--", sys.executable, str(script), "--plan", str(plan_path), "--stage", stage, "--out-dir", str(out)] + list(extra)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    done = subprocess.run(command, env=env, check=False)
    check(done.returncode == 0, "Scientific stage failed; chain stops: " + out.name)
    class Lock:
        def add(self, path):
            return sha(path)
    runner_gate(out, ["SUMMARY.json", complete, "SOURCE_LOCK.json"], Lock())
    return out


def close_source(inference, out):
    """Consumer-observed full file seal after the actual GPU runner has exited."""
    summary = read(inference / "SUMMARY.json")
    receipt = read(inference / "INFERENCE_COMPLETE.json")
    arm = read(inference / ARMS[2] / "COMPLETE.json")
    check(summary["passed"] is True and summary["status"] == "prediction_complete" and summary["image_count"] == 5000
          and receipt["summary_sha256"] == sha(inference / "SUMMARY.json") and summary["engineering"] is False,
          "Full inference terminal/summary scope differs")
    check(summary["model_integrity"]["passed"] is True
          and summary["model_integrity"]["initial_state_sha256"] == summary["model_integrity"]["final_state_sha256"],
          "Actual released model state changed during source inference")
    check(arm["official_model_provenance_sha256"] == sha(inference / "MODEL_PROVENANCE.json")
          and arm["predictions_sha256"] == summary["predictions_sha256"] == sha(inference / ARMS[2] / "predictions.json"),
          "Actual official model/prediction provenance differs")
    check(len(list((inference / ARMS[2] / "images").glob("*.json"))) == 5000
          and len(list((inference / "rng").glob("*.npz"))) == 5000, "Full-frame image/RNG output set incomplete")
    sealed = {}
    for p in sorted(inference.rglob("*")):
        if p.is_file() and p.name != ".run.claim":
            sealed[p.relative_to(inference).as_posix()] = {"sha256": sha(p), "size_bytes": p.stat().st_size}
    write(out / "SEGREFINER_SOURCE_CLOSED_SEAL.json", {"source_kind": "consumer_observed_full_seal_after_runner_exit",
          "source_run": str(inference), "source_run_id": SEG_RUN_ID, "sealed_at": utc(), "files": sealed,
          "producer_run_json_modified": False, "inference_was_actually_completed_before_seal": True})
    return sealed


def verify_closed_source(inference, sealed):
    for relative, expected in sealed.items():
        p = inference / relative
        check(p.is_file() and p.stat().st_size == expected["size_bytes"] and sha(p) == expected["sha256"],
              "Complete source seal changed: " + relative)


def main(args):
    plan = read(args.plan)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    check(not (out / "PIPELINE_INPUTS.json").exists(), "Fresh finite continuation Run required")
    inference = Path(plan["seg_run"]).resolve()
    check(inference.name == SEG_RUN_ID, "Only literal approved inference02 may be followed")
    initial = read(inference / "run.json")
    check(initial["source_kind"] == "runner_observed" and initial["run_id"] == SEG_RUN_ID, "Source is not an actual registered inference02")
    identity = {key: initial[key] for key in ("run_id", "created_at", "command", "working_directory", "source_kind", "pid")}
    source_hashes = {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("segrefiner_scoring_bridge.py"), Path(__file__).with_name("score_segrefiner_comparison.py"))}
    write(out / "PIPELINE_INPUTS.json", {"literal_inference": str(inference), "source_identity": identity,
        "source_hashes": source_hashes, "plan_sha256": sha(args.plan), "actual_interpreter": sys.executable,
        "wait_max_seconds": args.wait_max_seconds, "started_at": utc(), "automatic_retry": False,
        "execution_order": ["actual_GPU_inference02_terminal", "bounded_identity_engineering", "ordinary_three_arm_CPU_score", "pixels_and_Boundary02", "saved_array_and_pixel_verification"]})
    stages = {}
    began = time.monotonic()
    try:
        while True:
            current = read(inference / "run.json")
            check(all(current[key] == value for key, value in identity.items()), "Literal inference identity changed while waiting")
            if current["status"] == "completed":
                check(current["return_code"] == 0 and current["artifact_completeness"] == "complete", "GPU source completed without complete successful artifacts")
                break
            check(current["status"] in ("preparing", "running"), "GPU source failed/cancelled; finite chain stops")
            check(time.monotonic() - began <= args.wait_max_seconds, "Finite wait deadline elapsed; source remains incomplete")
            atomic(out / "PIPELINE_PROGRESS.json", {"phase": "waiting_for_actual_GPU_source_terminal", "source_run_id": SEG_RUN_ID,
                  "source_status": current["status"], "source_last_heartbeat": current.get("last_heartbeat"), "quality_metrics": None, "updated_at": utc()})
            time.sleep(15)
        for name, expected in source_hashes.items():
            check(sha(Path(__file__).with_name(name)) == expected, "Frozen continuation source changed while waiting")
        check(sha(args.plan) == read(out / "PIPELINE_INPUTS.json")["plan_sha256"], "Frozen scoring plan changed while waiting")
        class Lock:
            def add(self, path):
                return sha(path)
        runner_gate(inference, ["SUMMARY.json", "INFERENCE_COMPLETE.json"], Lock(), fresh_inputs=True)
        sealed = close_source(inference, out)
        # The completed 32-image engineering source tests only identity and
        # source plumbing; quality values remain unknown and never select scope.
        engineering_plan = dict(plan, seg_run=plan["engineering_seg_run"])
        engineering_path = out / "ENGINEERING_PLAN.json"
        write(engineering_path, engineering_plan)
        stage = execute(engineering_plan, engineering_path, "engineering", "RUN_SEGREFINER_SCORING_ENGINEERING32_CPU_S0_01")
        stages["engineering"] = str(stage)
        verify_closed_source(inference, sealed)
        atomic(out / "PIPELINE_PROGRESS.json", {"phase": "CPU_scoring", "stages": stages, "updated_at": utc()})
        scored = execute(plan, args.plan, "score", "RUN_SEGREFINER_THREE_ARM_SCORING_CPU_S0_01")
        stages["score"] = str(scored)
        verify_closed_source(inference, sealed)
        atomic(out / "PIPELINE_PROGRESS.json", {"phase": "CPU_pixels_Boundary", "stages": stages, "updated_at": utc()})
        readout = execute(plan, args.plan, "readout", "RUN_SEGREFINER_THREE_ARM_MULTIVIEW_CPU_S0_01", ["--scoring-run", str(scored)])
        stages["readout"] = str(readout)
        verify_closed_source(inference, sealed)
        atomic(out / "PIPELINE_PROGRESS.json", {"phase": "CPU_verification", "stages": stages, "updated_at": utc()})
        verified = execute(plan, args.plan, "verify", "RUN_SEGREFINER_THREE_ARM_VERIFY_CPU_S0_01",
                           ["--scoring-run", str(scored), "--readout-run", str(readout)])
        stages["verify"] = str(verified)
        verify_closed_source(inference, sealed)
        write(out / "SUMMARY.json", {"status": "finite_scoring_pipeline_complete", "passed": True, "stages": stages,
              "source_inference_run": SEG_RUN_ID, "source_model_inference_replayed": False, "completed_at": utc(),
              "report_owner": "root", "GPU_or_model_execution": False})
        write(out / "PIPELINE_COMPLETE.json", {"status": "completed", "summary_sha256": sha(out / "SUMMARY.json"), "stages": stages})
    except Exception as error:
        write(out / "PIPELINE_FAILURE.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(),
              "stages": stages, "failed_at": utc(), "source_inference_status": read(inference / "run.json").get("status"),
              "quality_metrics_not_fabricated": True})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wait-max-seconds", type=int, default=172800)
    main(p.parse_args())
