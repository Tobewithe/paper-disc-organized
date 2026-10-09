#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/qcr_run
PY=/root/miniconda3/bin/python
export TMPDIR=$ROOT/tmp
SOURCE=$ROOT/runs/RUN_stage1_dev_original_metrics_seed0
while [[ ! -f "$SOURCE/COMPLETE.json" ]]; do
  if [[ -f "$SOURCE/FAILURE.json" ]]; then exit 2; fi
  sleep 10
done
OUT=$ROOT/runs/RUN_stage1_dev_metrics_repair_seed0
mkdir "$OUT"
"$PY" - "$SOURCE/SUMMARY.json" "$OUT/SOURCE_METADATA.json" <<'PY'
from pathlib import Path
import json,sys
m=json.loads(Path(sys.argv[1]).read_text())['metadata']
m['statistical_repair']='Map unchanged native pyramid_level 0/1/2 to P3/P4/P5. No model execution or raw-row change.'
m['source_evaluation_summary']=sys.argv[1]
Path(sys.argv[2]).write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
PY
"$PY" "$ROOT/runner.py" --study STUDY_6f0c44af49e846f0a14f44a4b43d38e4 --run-id RUN_stage1_dev_metrics_repair_seed0 \
  --output "$OUT" --cwd "$ROOT" --snapshot "$ROOT/qcr_metrics.py" \
  --input "$SOURCE/PER_CANDIDATE.jsonl" --input "$OUT/SOURCE_METADATA.json" --input "$ROOT/IMAGE_SPLIT.json" \
  --expect "$OUT/SUMMARY.json" --expect "$OUT/METRICS.md" \
  --scope '{"purpose":"pyramid-index statistical correction","model_instantiated":false,"source_evaluation":"RUN_stage1_dev_original_metrics_seed0"}' \
  -- "$PY" "$ROOT/qcr_metrics.py" --rows "$SOURCE/PER_CANDIDATE.jsonl" --out "$OUT" --images "$ROOT/IMAGE_SPLIT.json" \
  --split dev --bootstrap 5000 --seed 20261006 --metadata "$OUT/SOURCE_METADATA.json"
