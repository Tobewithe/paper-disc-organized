#!/bin/bash
#SBATCH --job-name=occlusion-resource-check
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
cd /srv/gpu-lab/users/a25023104/projects/dense-video-occlusion
/opt/yolo/ultralytics-8.4.100/.venv/bin/python experiments/dense_video_synthetic_occlusion_20261019/scripts/server_preflight.py
