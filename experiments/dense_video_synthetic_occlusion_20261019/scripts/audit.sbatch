#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
ROOT=/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion
STUDY=$ROOT/experiments/dense_video_synthetic_occlusion_20261019
PILOT_ID=${1:?actual completed pilot Run ID required}
cd "$ROOT"
/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python "$STUDY/scripts/verify_dataset.py" --dataset "$STUDY/runs/$PILOT_ID/dataset" --study "$STUDY" --parent-run "$PILOT_ID"
