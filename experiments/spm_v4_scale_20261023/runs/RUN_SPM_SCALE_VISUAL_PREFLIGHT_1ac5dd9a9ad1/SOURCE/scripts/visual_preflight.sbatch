#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=spm-visual-preflight
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-visual-preflight-%j.log
set -euo pipefail
TASK_ROOT=/srv/gpu-lab/users/a25023104/projects/spm-v4-scale-20261023
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
cd "$TASK_ROOT"
exec /srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python scripts/visual_preflight.py --root "$TASK_ROOT"
