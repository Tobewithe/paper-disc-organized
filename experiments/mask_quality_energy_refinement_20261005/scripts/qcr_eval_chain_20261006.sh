#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/qcr_run
PY=/root/miniconda3/bin/python
CFG=/root/mask_quality_energy_refinement_20261005/RUN_CONFIG.json
export TMPDIR=$ROOT/tmp

run_eval() {
  local label=$1
  local split=$2
  shift 2
  local out=$ROOT/runs/$label
  if [[ -e "$out/run.json" ]]; then
    echo "Existing Run retained: $label; refusing duplicate execution" >&2
    return 2
  fi
  "$PY" "$ROOT/runner.py" --study STUDY_6f0c44af49e846f0a14f44a4b43d38e4 --run-id "$label" \
    --output "$out" --cwd "$ROOT" \
    --snapshot "$ROOT/qcr_evaluate_complete.py" --snapshot "$ROOT/qcr_metrics.py" \
    --snapshot "$ROOT/qcr_streaming_data.py" --snapshot "$ROOT/qcr_train_fixed.py" \
    --snapshot "$ROOT/online_runtime.py" --snapshot "$ROOT/prepare_qcr_cache.py" \
    --snapshot "$ROOT/legacy_prepare_cache.py" --snapshot "$CFG" \
    --snapshot "$ROOT/IMAGE_SPLIT.json" --snapshot "$ROOT/FINAL_SPLIT.json" \
    --input "$ROOT/runs/RUN_stage1_direct_seed0_retry5/final.pt" \
    --input "$ROOT/runs/RUN_stage1_quality_seed0/final.pt" \
    --expect "$out/COMPLETE.json" --expect "$out/SUMMARY.json" \
    --scope "{\"protocol\":\"QCR Stage-I complete diagnostic evaluation\",\"split\":\"$split\",\"nominal_seed_label\":0,\"actual_training_seed\":20261005,\"training\":false}" \
    -- "$PY" "$ROOT/qcr_evaluate_complete.py" --config "$CFG" \
    --direct-checkpoint "$ROOT/runs/RUN_stage1_direct_seed0_retry5/final.pt" \
    --quality-checkpoint "$ROOT/runs/RUN_stage1_quality_seed0/final.pt" \
    --split "$split" --out "$out" "$@"
}

run_eval RUN_stage1_dev_complete_seed0 dev
run_eval RUN_stage1_final_complete_seed0 final
run_eval RUN_stage1_dev_fp32_witness_seed0 dev --stream-dev --limit-images 10
echo 'All complete-evaluation Runs finished. No Stage-II training is scheduled.'
