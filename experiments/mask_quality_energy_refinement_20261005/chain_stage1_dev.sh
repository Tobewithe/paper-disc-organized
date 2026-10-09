#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/qcr_run
PY=/root/miniconda3/bin/python
RUNNER=$ROOT/runner.py
QRUN=$ROOT/runs/RUN_stage1_quality_seed0
export TMPDIR=$ROOT/tmp

while [[ ! -f "$QRUN/run.json" ]] || grep -q '"status": "running"' "$QRUN/run.json"; do
  sleep 30
done

if [[ ! -f "$QRUN/final.pt" ]]; then
  echo "quality training did not produce final.pt; skip dev evaluations" >&2
  exit 2
fi

CFG=/root/mask_quality_energy_refinement_20261005/RUN_CONFIG.json
PROTO=/root/mask_quality_energy_refinement_20261005/PROTOCOL.md
SCRIPT=$ROOT/qcr_train_fixed.py
RUNTIME=$ROOT/online_runtime.py
MANIFEST=$ROOT/assets/CANDIDATE_MANIFEST.json

$PY "$RUNNER" --study STUDY_6f0c44af49e846f0a14f44a4b43d38e4 \
  --output $ROOT/runs/RUN_stage1_direct_dev_seed0 \
  --cwd $ROOT --snapshot "$SCRIPT" --snapshot "$RUNTIME" --snapshot "$CFG" --snapshot "$PROTO" \
  --input $ROOT/runs/RUN_stage1_direct_seed0_retry5/final.pt \
  --expect $ROOT/runs/RUN_stage1_direct_dev_seed0/COMPLETE.json \
  --scope '{"dataset":"COCO train2017 2k DEV","protocol":"QCR Stage I eval","seed":0,"mode":"direct","split":"dev"}' \
  -- $PY "$SCRIPT" --config "$CFG" --manifest "$MANIFEST" --stage eval --mode direct \
  --checkpoint $ROOT/runs/RUN_stage1_direct_seed0_retry5/final.pt \
  --out $ROOT/runs/RUN_stage1_direct_dev_seed0 --split dev

$PY "$RUNNER" --study STUDY_6f0c44af49e846f0a14f44a4b43d38e4 \
  --output $ROOT/runs/RUN_stage1_quality_dev_seed0 \
  --cwd $ROOT --snapshot "$SCRIPT" --snapshot "$RUNTIME" --snapshot "$CFG" --snapshot "$PROTO" \
  --input $QRUN/final.pt \
  --expect $ROOT/runs/RUN_stage1_quality_dev_seed0/COMPLETE.json \
  --scope '{"dataset":"COCO train2017 2k DEV","protocol":"QCR Stage I eval","seed":0,"mode":"quality","split":"dev"}' \
  -- $PY "$SCRIPT" --config "$CFG" --manifest "$MANIFEST" --stage eval --mode quality \
  --checkpoint $QRUN/final.pt --out $ROOT/runs/RUN_stage1_quality_dev_seed0 --split dev
