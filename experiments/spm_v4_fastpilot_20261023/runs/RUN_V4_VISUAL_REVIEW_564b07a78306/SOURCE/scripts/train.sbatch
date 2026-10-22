#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --job-name=spm-v4-train
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-v4-train-%j.log
set -euo pipefail
TASK_ROOT=/srv/gpu-lab/users/a25023104/projects/spm-v4-fastpilot-20261023
TASK_PY=/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$TASK_ROOT"
exec "$TASK_PY" scripts/run_pipeline.py all --root "$TASK_ROOT"
