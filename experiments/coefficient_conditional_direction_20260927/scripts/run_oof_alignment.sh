#!/usr/bin/env bash
set -uo pipefail

root=/root/coefficient_conditional_direction_20260927
source=/root/coefficient_expected_benefit_20260927
run="$root/runs/RUN_0a169e2ff40248218b0d69a2c2154a7f"
mkdir -p "$run"
python - "$run/run.json" <<'PY'
import datetime, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "run_id": "RUN_0a169e2ff40248218b0d69a2c2154a7f",
    "status": "running", "source_kind": "remote_secondary_analysis",
    "started_at": datetime.datetime.now().astimezone().isoformat(),
}))
PY
python -u "$root/scripts/analyze_oof_alignment.py" \
  --targets "$source/runs/RUN_9ad188b832fd9bc49c0aafc7a61de989" \
  --retrieval "$root/runs/RUN_ae21f899bf5042108fb81b9f67896538/ROWS.jsonl.gz" \
  --score-folds \
  "$source/runs/RUN_504a6a764f714baca3fc42d34310e5ab" \
  "$source/runs/RUN_10cd105b47514a51a73038c8a478920f" \
  "$source/runs/RUN_d7fc4c8b8ee64dd79fd24085f73903ed" \
  "$source/runs/RUN_d325294a7db0476bbc9c8d65d8a90f52" \
  "$source/runs/RUN_9264566bf4fd46568df4ddf3b2d29042" \
  --model-folds \
  "$source/runs/RUN_c894608d969246939c6dab973c2eaa59" \
  "$source/runs/RUN_601742b6b8a24fa6bf776727bbec7cb8" \
  "$source/runs/RUN_98e99aca052745eb9d3aff1d022b849a" \
  "$source/runs/RUN_776616ea3d98485f91bf3a5dd0d8e9d9" \
  "$source/runs/RUN_63d0bff81fdb4574a1f3b47e5f75234e" \
  --out "$run" --bootstrap 2000
code=$?
python - "$run/run.json" "$code" <<'PY'
import datetime, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
obj = json.loads(path.read_text())
obj["status"] = "completed" if int(sys.argv[2]) == 0 else "failed"
obj["exit_code"] = int(sys.argv[2])
obj["finished_at"] = datetime.datetime.now().astimezone().isoformat()
path.write_text(json.dumps(obj, indent=2))
PY
exit "$code"
