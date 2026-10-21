#!/bin/bash
#SBATCH --job-name=pc-joint-preflight
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/pc-joint-preflight-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export YOLO_CONFIG_DIR=/srv/gpu-lab/users/a25023104/.cache/ultralytics-joint-20261018
cd /srv/gpu-lab/users/a25023104/projects/paper-disc-organized
.venv/bin/python experiments/mask_branch_joint_adaptation_20261018/scripts/server_preflight.py --config experiments/mask_branch_joint_adaptation_20261018/CONFIG.json
