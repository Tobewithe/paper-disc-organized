#!/bin/bash
#SBATCH --job-name=pc-wheel-recover
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/pc-wheel-recover-%j.log
set -euo pipefail
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR=/srv/gpu-lab/users/a25023104/.cache/uv
export UV_PYTHON_INSTALL_DIR=/srv/gpu-lab/users/a25023104/.cache/uv-python
cd /srv/gpu-lab/users/a25023104/projects/paper-disc-organized
python3 experiments/mask_branch_joint_adaptation_20261018/scripts/data_environment_wheel_recovery.py
