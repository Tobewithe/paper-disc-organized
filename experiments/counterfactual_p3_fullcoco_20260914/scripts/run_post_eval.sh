#!/usr/bin/env bash
set -euo pipefail
EXP=/root/autodl-tmp/counterfactual_p3_fullcoco_20260914
while [[ ! -f "$EXP/TRAINING_PAIR_COMPLETE" ]]; do
  if ! screen -ls 2>/dev/null | grep -q '[.]fullcoco_pair'; then
    echo '{"status":"training_failed_before_post_eval"}' > "$EXP/POST_EVAL_STATUS.json"
    exit 1
  fi
  sleep 60
done
echo '{"status":"official_validation_running"}' > "$EXP/POST_EVAL_STATUS.json"
python "$EXP/code/validate_full.py" \
  --data /root/autodl-tmp/datasets/coco2017/coco_full.yaml \
  --baseline "$EXP/runs/baseline_s0_full/weights/best.pt" \
  --method "$EXP/runs/cfp3r_s0_full/weights/best.pt" \
  --project "$EXP/official_eval" > "$EXP/logs/official_validation.log" 2>&1
python /root/autodl-tmp/counterfactual_p3_20260914/evaluate_saved_predictions.py \
  --annotation /root/autodl-tmp/datasets/coco2017/annotations/instances_val2017.json \
  --runs "$EXP/official_eval" --out "$EXP/official_cocoeval" \
  > "$EXP/logs/official_cocoeval.log" 2>&1
echo '{"status":"targeted_rich_running"}' > "$EXP/POST_EVAL_STATUS.json"
python /root/autodl-tmp/counterfactual_p3_20260914/eval_targeted_rich.py \
  --annotation /root/autodl-tmp/datasets/coco2017/annotations/instances_val2017.json \
  --images /root/autodl-tmp/datasets/coco2017/images/val2017 \
  --selection /root/autodl-tmp/counterfactual_p3_20260914/selection_remote.csv \
  --baseline "$EXP/runs/baseline_s0_full/weights/best.pt" \
  --method "$EXP/runs/cfp3r_s0_full/weights/best.pt" \
  --out "$EXP/targeted_rich" --seed 0 > "$EXP/logs/targeted_rich.log" 2>&1
echo '{"status":"all_complete"}' > "$EXP/POST_EVAL_STATUS.json"
touch "$EXP/ALL_COMPLETE"
