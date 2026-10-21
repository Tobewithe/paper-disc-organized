#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /srv/gpu-lab/users/a25023104/projects/dense-video-occlusion-v2
export PYTHONPATH="$PWD/deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
/opt/yolo/ultralytics-8.4.100/.venv/bin/python -m shared.dense_video_occlusion.runner "$@"
