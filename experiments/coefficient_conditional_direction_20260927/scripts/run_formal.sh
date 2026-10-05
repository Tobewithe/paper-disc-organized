#!/usr/bin/env bash
set -uo pipefail

root=/root/coefficient_conditional_direction_20260927
source=/root/coefficient_expected_benefit_20260927
run="$root/runs/RUN_ae21f899bf5042108fb81b9f67896538"
mkdir -p "$run"
python - "$run/run.json" <<'PY'
import datetime, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "run_id": "RUN_ae21f899bf5042108fb81b9f67896538",
    "status": "running",
    "source_kind": "remote_analysis",
    "started_at": datetime.datetime.now().astimezone().isoformat(),
    "protocol": "/root/coefficient_conditional_direction_20260927/PROTOCOL.md",
}))
PY
python -u "$root/scripts/analyze_conditional_direction.py" \
  --manifest "$source/MANIFEST.json" \
  --targets "$source/runs/RUN_9ad188b832fd9bc49c0aafc7a61de989" \
  --annotations /root/autodl-tmp/coefficient_predictability_20260924/data/annotations/instances_train2017.json \
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
