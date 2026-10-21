#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=144G
set -euo pipefail
ROOT=/srv/gpu-lab/users/a25023104/projects/lcf-server-20261020
PY=/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ROOT"
srun --kill-on-bad-exit=1 --gpu-bind=single:1 "$PY" experiments/mask_local_connectivity_server_20261020/scripts/server_runner.py "${1:?stage required}"
