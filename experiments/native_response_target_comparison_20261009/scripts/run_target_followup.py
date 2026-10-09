"""Finite foreground wait -> producer seal -> isolated cost -> cost seal.

No scientific retry, WMI, scheduler, training or new parameter. One literal
source chain must actually complete before any cost child can be started.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from target_common import dump, load_config, now, sha


def terminal(root):
    metadata = json.loads((root / "run.json").read_text())
    if metadata["status"] in ("failed", "cancelled"):
        raise RuntimeError("Literal source failed; preserve and stop: " + str(root))
    if metadata["status"] != "completed":
        return False
    if metadata["return_code"] != 0 or metadata["artifact_completeness"] != "complete":
        raise RuntimeError("Literal source completion gate failed: " + str(root))
    return True


def run_stage(c, root, name, script, arguments, expected):
    output = root / "runs" / name
    if output.exists():
        raise FileExistsError("Finite follow-up never repeats an existing Run: " + name)
    source = Path(__file__).with_name(script)
    command = [sys.executable, c["runner"], "--study", c["study_id"], "--output", str(output),
               "--run-id", name, "--cwd", str(root), "--snapshot", str(source),
               "--input", c["protocol"], "--metrics", str(output / "SUMMARY.json")]
    for filename in expected:
        command += ["--expect", str(output / filename)]
    command += ["--", sys.executable, str(source), "--output", str(output)] + arguments
    result = subprocess.run(command, check=False)
    if result.returncode or not terminal(output):
        raise RuntimeError("Follow-up stage failed: " + name)
    return output


def main():
    p = argparse.ArgumentParser()
    for name in ("config", "protocol-sha256", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--max-hours", type=float, default=12)
    a = p.parse_args()
    c = load_config(a.config, a.protocol_sha256)
    root, out = Path(c["root"]), Path(a.output)
    if (out / "FOLLOWUP_INPUTS.json").exists():
        raise FileExistsError("Finite controller already used")
    began = time.monotonic()
    deadline = began + min(a.max_hours, 12) * 3600
    pipeline = root / "runs" / "RUN_NATIVE_TARGET_PIPELINE_S0_01"
    preflight = root / "runs" / "RUN_NATIVE_TARGET_PREFLIGHT_S0_01"
    dump(out / "FOLLOWUP_INPUTS.json", dict(source_pipeline=str(pipeline), source_config_sha256=sha(a.config),
         protocol_sha256=a.protocol_sha256, deadline_hours=min(a.max_hours, 12), actual_mode="SSH foreground dependent",
         detached=False, scientific_retry=False, cost_engineering="RUN_NATIVE_TARGET_COST_ENGINEERING_S0_01",
         cost_panel="RUN_NATIVE_TARGET_COST_PANEL_S0_01", source_collection="RUN_NATIVE_TARGET_COLLECTION_S0_01",
         cost_collection="RUN_NATIVE_TARGET_COST_COLLECTION_S0_01"))
    try:
        while not terminal(pipeline):
            if time.monotonic() > deadline:
                raise TimeoutError("Finite source wait deadline reached; no cost started")
            dump(out / "PHASE.json", dict(phase="waiting_original_production_pipeline", source=str(pipeline), updated_at=now()))
            time.sleep(20)
        pc = json.loads((pipeline / "PIPELINE_COMPLETE.json").read_text())
        if not pc["passed"] or pc["summary_sha256"] != sha(pipeline / "SUMMARY.json"):
            raise ValueError("Actual finite production completion differs")
        formal = pc["stages"]["FORMAL"]
        for key in ("extract", "fit_I", "fit_H", "inference"):
            if not terminal(Path(formal[key])):
                raise ValueError("Formal source not actually terminal")
        dump(out / "PHASE.json", dict(phase="sealing_production", updated_at=now()))
        production = run_stage(c, root, "RUN_NATIVE_TARGET_COLLECTION_S0_01", "collect_target_runs.py",
             ["--pipeline", str(pipeline), "--preflight", str(preflight)],
             ["SUMMARY.json", "COMPLETE.json", "BUNDLE_MANIFEST.json", "PRODUCTION_BUNDLE.zip"])
        dump(out / "PHASE.json", dict(phase="serial_isolated_cost_engineering_then_panel", updated_at=now()))
        done = subprocess.run(["C:/Program Files/PowerShell/7/pwsh.exe", "-NoProfile", "-File",
                               str(Path(__file__).with_name("register_target_cost.ps1"))], check=False)
        if done.returncode:
            raise RuntimeError("Finite cost sequence failed; do not retry implicitly")
        costs = [root / "runs" / ("RUN_NATIVE_TARGET_COST_" + name + "_S0_01") for name in ("ENGINEERING", "PANEL")]
        for cost in costs:
            if not terminal(cost):
                raise ValueError("Cost runner completion missing")
            complete = json.loads((cost / "COST_COMPLETE.json").read_text())
            if not complete["passed"] or complete["summary_sha256"] != sha(cost / "SUMMARY.json"):
                raise ValueError("Cost completion digest differs")
        dump(out / "PHASE.json", dict(phase="sealing_cost", updated_at=now()))
        sealed = run_stage(c, root, "RUN_NATIVE_TARGET_COST_COLLECTION_S0_01", "seal_extra_runs.py",
                           sum((["--run", str(path)] for path in costs), []),
                           ["SUMMARY.json", "COMPLETE.json", "BUNDLE_MANIFEST.json", "EXTRA_RUNS_BUNDLE.zip"])
        summary = dict(passed=True, status="followup_complete", production_collection=str(production), cost_collection=str(sealed),
                       elapsed_seconds=time.monotonic()-began, completed_at=now(), detached=False,
                       scientific_retries=0, scoring_started=False, limits="Foreground SSH connection required; no AP or latency CI produced here")
        dump(out / "SUMMARY.json", summary)
        dump(out / "COMPLETE.json", dict(passed=True, summary_sha256=sha(out / "SUMMARY.json")))
    except Exception as exc:
        dump(out / "FOLLOWUP_FAILURE.json", dict(error=repr(exc), traceback=traceback.format_exc(), completed_at=now()))
        raise


if __name__ == "__main__":
    main()
