"""One finite local CPU scoring continuation using structured recorder argv."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--plan", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
plan = json.loads(a.plan.read_text(encoding="utf-8-sig"))
study = Path(plan["study"])
driver = study / "scripts/run_segrefiner_scoring_pipeline_v2.py"
argv = [sys.executable, plan["runner"], "--study", "STUDY_SEGREFINER_OFFICIAL_CONTENT_BASELINE_20261009",
        "--output", str(a.output), "--run-id", a.output.name, "--cwd", plan["workspace"], "--input", str(a.plan)]
for path in (driver, study / "scripts" / plan["scoring_script_name"], study / "scripts/segrefiner_scoring_bridge.py", Path(__file__), Path(plan["runner"])):
    argv += ["--snapshot", str(path)]
argv += ["--expect", str(a.output / "SUMMARY.json"), "--expect", str(a.output / "PIPELINE_COMPLETE.json"),
         "--metrics", str(a.output / "SUMMARY.json"), "--scope", json.dumps({"type": "one_finite_dependency_continuation_v2_GT_binding",
             "wait_for_literal_actual_inference02": True, "scientific_retry": False, "scheduler_or_automation": False,
             "CPU_stages_only_after_actual_GPU_source_exit": True, "max_wait_seconds": 172800}),
         "--", sys.executable, str(driver), "--plan", str(a.plan), "--output", str(a.output), "--wait-max-seconds", "172800"]
raise SystemExit(subprocess.run(argv, cwd=plan["workspace"], check=False).returncode)
