#!/bin/bash
#SBATCH --job-name=YOLO-blender-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/yoloseg-blender-%j.out
#SBATCH --error=/srv/gpu-lab/users/a25023104/logs/yoloseg-blender-%j.err
set -euo pipefail
ROOT=/srv/gpu-lab/users/a25023104/projects/yoloseg-blender-stress-20261020
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/srv/gpu-lab/users/a25023104/projects/lcf-server-20261020/_site
export YOLO_CONFIG_DIR="$ROOT/yolo-settings"
cd "$ROOT"
srun --kill-on-bad-exit=1 /srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python scripts/runner.py
