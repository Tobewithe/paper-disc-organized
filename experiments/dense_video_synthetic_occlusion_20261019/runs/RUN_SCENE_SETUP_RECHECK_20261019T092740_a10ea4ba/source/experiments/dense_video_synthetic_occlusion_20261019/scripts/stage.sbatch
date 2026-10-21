#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
cd /srv/gpu-lab/users/a25023104/projects/dense-video-occlusion
/usr/bin/python3 experiments/dense_video_synthetic_occlusion_20261019/scripts/execute.py "${1:?stage required}"
