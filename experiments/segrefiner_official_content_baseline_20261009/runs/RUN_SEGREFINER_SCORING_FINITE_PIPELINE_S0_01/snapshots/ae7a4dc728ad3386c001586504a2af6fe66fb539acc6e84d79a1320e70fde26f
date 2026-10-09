"""Launch one recorder with structured argv, preserving literal JSON arguments."""
import json
from pathlib import Path
import subprocess
import sys

workspace = Path("C:/Dpan/codexproject/paper-disc-organized")
study = workspace / "experiments/segrefiner_official_content_baseline_20261009"
out = study / "runs/RUN_SEGREFINER_SCORING_FINITE_PIPELINE_S0_01"
plan = study / "assets/scoring_plan_batch8_fixed5k_20261009.json"
script = study / "scripts/run_segrefiner_scoring_pipeline.py"
runner = workspace / "shared/tools/research_runner/runner.py"
argv = [sys.executable, str(runner), "--study", "STUDY_SEGREFINER_OFFICIAL_CONTENT_BASELINE_20261009",
        "--output", str(out), "--run-id", out.name, "--cwd", str(workspace), "--input", str(plan),
        "--snapshot", str(script), "--snapshot", str(study / "scripts/score_segrefiner_comparison.py"),
        "--snapshot", str(study / "scripts/segrefiner_scoring_bridge.py"), "--snapshot", str(Path(__file__)),
        "--snapshot", str(runner), "--expect", str(out / "SUMMARY.json"), "--expect", str(out / "PIPELINE_COMPLETE.json"),
        "--metrics", str(out / "SUMMARY.json"), "--scope", json.dumps({"type": "one_finite_dependency_continuation",
            "wait_for_literal_actual_inference02": True, "scientific_retry": False, "scheduler_or_automation": False,
            "CPU_stages_only_after_actual_GPU_source_exit": True, "max_wait_seconds": 172800}),
        "--", sys.executable, str(script), "--plan", str(plan), "--output", str(out), "--wait-max-seconds", "172800"]
with (out / "launcher.stdout.log").open("wb") as stdout, (out / "launcher.stderr.log").open("wb") as stderr:
    done = subprocess.run(argv, cwd=workspace, stdout=stdout, stderr=stderr, check=False)
raise SystemExit(done.returncode)
