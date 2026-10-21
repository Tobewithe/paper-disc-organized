#!/bin/bash
set -euo pipefail
cd /srv/gpu-lab/users/a25023104/projects/dense-video-occlusion-v2
export PYTHONPATH="$PWD/deps:$PWD"
export OPENBLAS_NUM_THREADS=1
/opt/yolo/ultralytics-8.4.100/.venv/bin/python -m shared.dense_video_occlusion.adjudicate
