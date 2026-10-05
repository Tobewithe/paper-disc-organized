#!/usr/bin/env bash
set -uo pipefail

run_id="$1"
per_group="$2"
root=/root/coefficient_conditional_direction_20260927
source=/root/coefficient_expected_benefit_20260927
run="$root/runs/$run_id"
mkdir -p "$run"
python - "$run/run.json" "$run_id" "$per_group" <<'PY'
import datetime, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "run_id": sys.argv[2], "status": "running",
    "source_kind": "remote_exploratory_analysis",
    "per_group": int(sys.argv[3]),
    "started_at": datetime.datetime.now().astimezone().isoformat(),
}))
PY
python -u "$root/scripts/analyze_gradient_terms.py" \
  --manifest "$source/MANIFEST.json" \
  --targets "$source/runs/RUN_9ad188b832fd9bc49c0aafc7a61de989" \
  --annotations /root/autodl-tmp/coefficient_predictability_20260924/data/annotations/instances_train2017.json \
  --reference "$root/runs/RUN_ae21f899bf5042108fb81b9f67896538/ROWS.jsonl.gz" \
  --cache /root/autodl-tmp/coefficient_data_scaling_20260924/runs/RUN_a24f6c759f714fb9bfcd5a59bb04e6d3/images \
  --official-source /root/autodl-tmp/coefficient_predictability_20260924 \
  --out "$run" --per-group "$per_group"
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
