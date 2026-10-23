#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --job-name=spm-feature-prefetch
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-feature-prefetch-%A_%a.log
set -euo pipefail
TASK_ROOT=/srv/gpu-lab/users/a25023104/projects/spm-v4-scale-20261023
STAGES=(warm synthetic)
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$TASK_ROOT"
exec /srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python scripts/prefetch_features.py "${STAGES[${SLURM_ARRAY_TASK_ID:?}]}" --root "$TASK_ROOT"
