"""Foreground SSH launcher using structured Python argv; no WMI/detachment."""
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
driver = study / "scripts/run_target_scoring_pipeline_inputpath_v2.py"
argv = [sys.executable, plan["runner"], "--study", "STUDY_NATIVE_RESPONSE_TARGET_COMPARISON_20261009",
        "--output", str(a.output), "--run-id", a.output.name, "--cwd", str(study), "--input", str(a.plan),
        "--snapshot", str(driver), "--snapshot", str(Path(__file__))]
for name in ("evaluate_target_comparison.py", "bootstrap_target_ap.py", "verify_target_comparison.py", "target_scoring_common.py"):
    argv += ["--snapshot", str(study / "scripts" / name)]
argv += ["--expect", str(a.output / "SUMMARY.json"), "--expect", str(a.output / "PIPELINE_COMPLETE.json"),
         "--metrics", str(a.output / "SUMMARY.json"), "--scope", json.dumps({"type": "finite_target_CPU_scientific_followup",
             "foreground_SSH_lifetime": True, "SSH_independence_proven": False, "WMI_used": False,
             "all_producer_cost_collection_terminals_required_before_heavy_CPU": True, "automatic_retry": False}),
         "--", sys.executable, str(driver), "--plan", str(a.plan), "--output", str(a.output)]
raise SystemExit(subprocess.run(argv, cwd=study, check=False).returncode)
