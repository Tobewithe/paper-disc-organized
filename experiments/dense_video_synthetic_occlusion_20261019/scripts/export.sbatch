#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python /srv/gpu-lab/users/a25023104/projects/dense-video-occlusion/experiments/dense_video_synthetic_occlusion_20261019/scripts/export_labels.py
