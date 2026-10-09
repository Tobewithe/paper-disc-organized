"""One foreground-SSH finite continuation: all source/cost seals, then CPU.

Waiting reads only small runner metadata. Full hashing, GT and cache parsing
start only after both cost Runs AND both collection processes have exited 0.
No detached-lifecycle claim, scheduler, quality retry, or model parameter edit.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

FROZEN = {"evaluate_target_comparison.py": "54e278c9450aad1111a4864415255055b5522e8bfca3fc48af53ca5e933f316d",
          "bootstrap_target_ap.py": "05978268aae334298212bd27559e41e249d62bbac94dde9f982b4ff368cbd4d9",
          "verify_target_comparison.py": "a69927ce7db1e1b61b13fd30976cb861ffdcc7dc6d5ca8234c55d6b6c08f58a3",
          "target_scoring_common.py": "a4441164d589e7c929196ddfb3612cb9bf0c2c04d1c296514d1d8a75de8ac835"}
PROTOCOL_SHA = "cc0d3169e32fc1d42dbd8f1027861ef29fa75f18bb145addd73e6c77d6a94057"
COST_SHA = "f3c22f25334316c429027df81e6250fe874dc44daf9a08f7ab8da365b71c11f1"


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def atomic(path, value):
    p = Path(path)
    temp = p.with_name(p.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(p)


def now():
    return datetime.now(timezone.utc).isoformat()


def observed(root, required):
    record = read(root / "run.json")
    check(record["source_kind"] == "runner_observed" and record["status"] == "completed"
          and record["return_code"] == 0 and record["artifact_completeness"] == "complete", "Actual successful runner required: " + root.name)
    origin = Path(record["locations"][0]["path"])
    artifacts = {}
    for row in record["artifacts"]:
        try:
            relative = Path(row["path"]).relative_to(origin).as_posix()
        except ValueError:
            continue
        artifacts[relative] = row
    for name in required:
        check(name in artifacts and artifacts[name]["exists"] is True and sha(root / name) == artifacts[name]["sha256"],
              "Actual terminal artifact changed/missing: " + root.name + "/" + name)
    return record


def manifest_gate(root):
    records = {}
    for line in (root / "manifest.sha256").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        relative = relative.strip().lstrip("*")
        p = (root / relative).resolve(strict=True)
        p.relative_to(root.resolve())
        check(relative not in records and sha(p) == expected, "Original source manifest hash differs: " + root.name + "/" + relative)
        records[relative] = expected
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
              and p.name not in ("manifest.sha256", "transfer.json") and not p.name.endswith(".tmp")}
    check(set(records) == actual, "Whole source file inventory differs from sealed manifest: " + root.name)
    return {"path": str(root), "manifest_sha256": sha(root / "manifest.sha256"), "files": records}


def source_lock_gate(root):
    lock = read(root / "SOURCE_LOCK.json")
    for name, row in lock["files"].items():
        check(sha(row["path"]) == row["sha256"], "Frozen producer source/input changed: " + name)
        if row.get("snapshot"):
            check(sha(root / row["snapshot"]) == row["sha256"], "Producer snapshot identity differs")
    vendor = Path(lock["vendor_root"])
    actual = {p.relative_to(vendor).as_posix(): sha(p) for p in sorted(vendor.rglob("*.py"))}
    check(actual == lock["vendor_python_files"], "Frozen producer vendor source tree changed")
    return sha(root / "SOURCE_LOCK.json")


def close_inputs(plan, out):
    study = Path(plan["study"])
    check(sha(study / "PROTOCOL.md") == PROTOCOL_SHA and sha(study / "COST_PROTOCOL.md") == COST_SHA, "Frozen science/cost protocol differs")
    for name, expected in FROZEN.items():
        check(sha(study / "scripts" / name) == expected, "Frozen scoring executable changed: " + name)
    roots = {key: study / "runs" / value for key, value in plan["upstream_runs"].items()}
    seals = {}
    for key in ("extract", "fit_I", "fit_H", "inference"):
        root = roots[key]
        observed(root, ["SUMMARY.json", "COMPLETE.json"])
        summary, complete = read(root / "SUMMARY.json"), read(root / "COMPLETE.json")
        check(summary["passed"] is True and summary["engineering"] is False and summary["source_unchanged"] is True
              and complete["summary_sha256"] == sha(root / "SUMMARY.json"), "Producer science terminal binding differs: " + key)
        check(summary["source_lock_sha256"] == source_lock_gate(root), "Producer explicit source/input lock differs")
        seals[key] = manifest_gate(root)
    fits = [read(roots[key] / "SUMMARY.json") for key in ("fit_I", "fit_H")]
    check(read(roots["extract"] / "SUMMARY.json")["images"] == 20000, "Actual full20k extraction required")
    for key in ("extraction_run", "extraction_summary_sha256", "extraction_complete_sha256", "full_hgb_parameters", "sample_weight",
                "weight_bytes_sha256", "training_feature_bytes_sha256", "supervised_binding_sha256", "training_rows", "training_images"):
        check(fits[0][key] == fits[1][key], "Targets changed common fit budget/input: " + key)
    inference = read(roots["inference"] / "SUMMARY.json")
    check(inference["image_count"] == 5000 and inference["gt_parsed"] is False and inference["gt_used_in_inference"] is False,
          "Actual complete GT-free native5000 source required")
    for key, engineering, samples, children in (("cost_engineering", True, 12, 3), ("cost_panel", False, 288, 9)):
        root = roots[key]
        observed(root, ["SUMMARY.json", "COST_COMPLETE.json", "COST_SAMPLES.json"])
        summary, complete, inputs = [read(root / name) for name in ("SUMMARY.json", "COST_COMPLETE.json", "COST_INPUTS.json")]
        check(summary["passed"] is True and summary["status"] == "cost_complete" and summary["engineering"] is engineering
              and summary["samples"] == samples and summary["fresh_children"] == children and summary["gt_used"] is False
              and summary["source_unchanged"] is True and complete["summary_sha256"] == sha(root / "SUMMARY.json")
              and complete["samples_sha256"] == sha(root / "COST_SAMPLES.json"), "Cost actual terminal/count/profile differs")
        binding = inputs["source_binding"]
        canonical_binding = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        check(binding["reference_summary_sha256"] == sha(roots["inference"] / "SUMMARY.json")
              and binding["reference_run_metadata_sha256"] == sha(roots["inference"] / "run.json")
              and binding["script_sha256"] == plan["cost_script_sha256"] and binding["cost_protocol_sha256"] == COST_SHA,
              "Cost profile is not the frozen actual formal inference")
        check(canonical_binding == inputs["source_binding_sha256"] == summary["source_binding_sha256"], "Actual cost source binding hash differs")
        seals[key] = manifest_gate(root)
    for key, bundle in (("production_collection", "PRODUCTION_BUNDLE.zip"), ("cost_collection", "EXTRA_RUNS_BUNDLE.zip")):
        root = roots[key]
        observed(root, ["SUMMARY.json", "COMPLETE.json", "BUNDLE_MANIFEST.json", bundle])
        summary, complete, inventory = [read(root / n) for n in ("SUMMARY.json", "COMPLETE.json", "BUNDLE_MANIFEST.json")]
        check(summary["passed"] is True and summary["status"] == "sealed" and summary["source_unchanged"] is True
              and complete["summary_sha256"] == sha(root / "SUMMARY.json") and complete["bundle_sha256"] == summary["bundle_sha256"] == sha(root / bundle),
              "Completed original-byte collection terminal differs")
        for row in inventory["files"]:
            check(sha(row["source"]) == row["sha256"] and Path(row["source"]).stat().st_size == row["bytes"], "Sealed source inventory changed")
        seals[key] = {"path": str(root), "bundle_sha256": summary["bundle_sha256"], "manifest_sha256": sha(root / "BUNDLE_MANIFEST.json")}
    view = Path(plan["reference_scoring"])
    origin = read(view / "ORIGINAL_SCORE_COMPLETE.json")
    check(origin["artifact_sha256"]["baseline/COCO_METRICS.json"] == sha(view / "baseline" / "COCO_METRICS.json"), "Accepted baseline reference output origin differs")
    check(origin["summary_sha256"] == sha(view / "ORIGINAL_SCORE_SUMMARY.json"), "Accepted baseline reference summary differs")
    record = read(view / "ORIGINAL_SCORE_RUN.json")
    check(record["source_kind"] == "runner_observed" and record["status"] == "completed" and record["return_code"] == 0,
          "Baseline reference view cannot replace a real original scorer")
    write(out / "UPSTREAM_SCIENCE_COST_CLOSED_SEALS.json", {"sealed_at": now(), "source_kind": "consumer_verified_original_source_manifests",
          "sources": seals, "cost_timing_and_collection_finished_before_full_hashes": True, "producer_or_cost_run_modified": False})


def execute(plan, stage, name, argv):
    study = Path(plan["study"])
    out = study / "runs" / name
    check(not out.exists(), "Fresh CPU scientific Run required: " + name)
    script_name = {"score": "evaluate_target_comparison.py", "bootstrap": "bootstrap_target_ap.py", "verify": "verify_target_comparison.py"}[stage]
    command = [sys.executable, plan["runner"], "--study", "STUDY_NATIVE_RESPONSE_TARGET_COMPARISON_20261009", "--output", str(out),
               "--run-id", name, "--cwd", str(study), "--input", plan["plan_path"], "--snapshot", str(study / "scripts" / script_name),
               "--snapshot", str(study / "scripts/target_scoring_common.py"), "--scope",
               json.dumps({"environment": "local_laptop_CPU_foreground_SSH", "stage": stage, "GT_only_scoring": True,
                           "cost_completed_before_this_CPU_stage": True, "no_auto_quality_retry": True})]
    expected = ["COMPLETE.json", "SOURCE_LOCK.json", "VERIFICATION.json" if stage == "verify" else "SUMMARY.json"]
    for filename in expected:
        command += ["--expect", str(out / filename)]
    command += ["--metrics", str(out / ("VERIFICATION.json" if stage == "verify" else "SUMMARY.json")),
                "--", sys.executable, str(study / "scripts" / script_name)] + list(argv) + ["--out-dir", str(out)]
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = "4"
    result = subprocess.run(command, env=env, check=False)
    check(result.returncode == 0, "CPU scientific stage failed; preserve and stop: " + name)
    observed(out, expected)
    return out


def main(args):
    plan = read(args.plan)
    plan["plan_path"] = str(args.plan.resolve())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    check(not (out / "PIPELINE_INPUTS.json").exists(), "Finite continuation retry requires a new Run")
    write(out / "PIPELINE_INPUTS.json", {"source_run_ids": plan["upstream_runs"], "plan_sha256": sha(args.plan), "actual_python": sys.executable,
          "launch_lifetime": "foreground SSH session and inherited Windows Job; not proven detached", "wait_max_seconds": args.wait_max_seconds,
          "source_hashes": FROZEN, "created_at": now(), "automatic_retry": False, "scheduler_or_notification_service": False})
    began, identities, stages = time.monotonic(), {}, {}
    try:
        while True:
            snapshots, all_done = {}, True
            for key, name in plan["upstream_runs"].items():
                path = Path(plan["study"]) / "runs" / name / "run.json"
                if not path.is_file():
                    snapshots[key] = {"run_id": name, "status": "not_created_yet"}
                    all_done = False
                    continue
                record = read(path)  # Only small mutable execution metadata here.
                check(record["source_kind"] == "runner_observed" and record["run_id"] == name, "Literal upstream runner identity differs")
                identity = {k: record[k] for k in ("run_id", "created_at", "command", "working_directory", "source_kind")}
                if key not in identities:
                    identities[key] = identity
                check(identities[key] == identity, "Upstream Run was replaced while waiting")
                check(record["status"] in ("preparing", "running", "completed"), "Actual upstream failed/cancelled: " + name)
                if record["status"] == "completed":
                    check(record["return_code"] == 0 and record["artifact_completeness"] == "complete", "Upstream terminal is incomplete/failed: " + name)
                else:
                    all_done = False
                snapshots[key] = {"run_id": name, "status": record["status"], "last_heartbeat": record.get("last_heartbeat")}
            atomic(out / "PIPELINE_PROGRESS.json", {"phase": "waiting_for_all_actual_producer_cost_and_collection_terminals",
                  "upstream": snapshots, "GT_parsed": False, "full_data_hashes_started": False, "quality_metrics": None, "updated_at": now()})
            if all_done:
                break
            check(time.monotonic() - began <= args.wait_max_seconds, "Finite upstream wait deadline elapsed")
            time.sleep(15)
        atomic(out / "PIPELINE_PROGRESS.json", {"phase": "verifying_complete_closed_source_and_cost_manifests", "updated_at": now()})
        close_inputs(plan, out)
        root = Path(plan["study"])
        inference = root / "runs" / plan["upstream_runs"]["inference"]
        score = execute(plan, "score", "RUN_NATIVE_TARGET_FORMAL_SCORING_CPU_S0_02", ["--run", str(inference), "--annotations", plan["annotations"],
            "--reference-baseline", plan["reference_baseline"], "--source-paired", plan["fixed_source"], "--protocol", str(root / "PROTOCOL.md"),
            "--reference-scoring", plan["reference_scoring"], "--boundary-vendor", plan["boundary_vendor"],
            "--boundary-provenance", plan["boundary_provenance"], "--boundary-provenance-sha256", plan["boundary_provenance_sha256"]])
        stages["score"] = str(score)
        atomic(out / "PIPELINE_PROGRESS.json", {"phase": "1000_paired_joint_image_AP_bootstrap", "stages": stages, "updated_at": now()})
        boot = execute(plan, "bootstrap", "RUN_NATIVE_TARGET_FORMAL_AP_BOOTSTRAP_CPU_S0_02",
                       ["--scoring-run", str(score), "--engineering-run", plan["AP_engineering_run"]])
        stages["bootstrap"] = str(boot)
        atomic(out / "PIPELINE_PROGRESS.json", {"phase": "saved_ledgers_intervals_and_four_AP_replays_verification", "stages": stages, "updated_at": now()})
        verified = execute(plan, "verify", "RUN_NATIVE_TARGET_FORMAL_SCORING_VERIFY_CPU_S0_02", ["--scoring-run", str(score), "--bootstrap-run", str(boot)])
        stages["verify"] = str(verified)
        write(out / "SUMMARY.json", {"status": "finite_target_scientific_pipeline_complete", "passed": True, "stages": stages,
              "completed_at": now(), "GPU_or_model_execution": False, "quality_retry_or_parameter_search": False,
              "verification_scope": "1000 fixed-cluster ratio/RNG/CI reconstruction and four AP replays reuse producer paired_aps; independent algorithm evidence is fresh official copy engineering",
              "independent_B_four_draws_three_separate_reaccumulates": "subsequent B-owned verification; not performed or claimed by this pipeline"})
        write(out / "PIPELINE_COMPLETE.json", {"status": "completed", "summary_sha256": sha(out / "SUMMARY.json"), "stages": stages})
    except Exception as error:
        write(out / "PIPELINE_FAILURE.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(),
              "stages": stages, "failed_at": now(), "partial_scientific_outputs_preserved": True, "automatic_retry": False})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wait-max-seconds", type=int, default=86400)
    main(p.parse_args())
