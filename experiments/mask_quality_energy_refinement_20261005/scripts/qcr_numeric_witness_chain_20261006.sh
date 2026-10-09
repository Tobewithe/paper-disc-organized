#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/qcr_run
PY=/root/miniconda3/bin/python
export TMPDIR=$ROOT/tmp
PREV=$ROOT/runs/RUN_stage1_dev_fp32_witness_seed0
while [[ ! -f "$PREV/COMPLETE.json" ]]; do
  for label in RUN_stage1_final_complete_seed0_retry1 RUN_stage1_dev_original_metrics_seed0 RUN_stage1_dev_fp32_witness_seed0; do
    if [[ -f "$ROOT/runs/$label/FAILURE.json" ]]; then
      echo "Prior evaluation failed; preserving it and not starting the numeric witness" >&2
      exit 2
    fi
  done
  sleep 30
done
OUT=$ROOT/runs/RUN_stage1_dev_numeric_witness_seed0
if [[ -e "$OUT/run.json" ]]; then echo 'Existing witness retained; no duplicate' >&2; exit 2; fi
"$PY" "$ROOT/runner.py" --study STUDY_6f0c44af49e846f0a14f44a4b43d38e4 --run-id RUN_stage1_dev_numeric_witness_seed0 \
  --output "$OUT" --cwd "$ROOT" --snapshot "$ROOT/qcr_fp32_cache_witness.py" \
  --snapshot "$ROOT/qcr_streaming_data.py" --snapshot "$ROOT/online_runtime.py" \
  --snapshot "$ROOT/prepare_qcr_cache.py" --snapshot "$ROOT/legacy_prepare_cache.py" \
  --snapshot "$ROOT/EVALUATION_CONFIG_20261006.json" --snapshot "$ROOT/IMAGE_SPLIT.json" \
  --input "$ROOT/runs/RUN_stage1_dev_original_metrics_seed0/PER_CANDIDATE.jsonl" \
  --input "$PREV/PER_CANDIDATE.jsonl" --expect "$OUT/COMPLETE.json" --expect "$OUT/NUMERIC_WITNESS.json" \
  --scope '{"purpose":"ten-image FP32/cache numerical witness","split":"dev","new_heads_instantiated":false,"training":false}' \
  -- "$PY" "$ROOT/qcr_fp32_cache_witness.py" --config "$ROOT/EVALUATION_CONFIG_20261006.json" --out "$OUT" \
  --cached-rows "$ROOT/runs/RUN_stage1_dev_original_metrics_seed0/PER_CANDIDATE.jsonl" --replayed-rows "$PREV/PER_CANDIDATE.jsonl"
