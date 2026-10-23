#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=spm-task-smoke
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-task-smoke-%j.log
set -euo pipefail
TASK_ROOT=/srv/gpu-lab/users/a25023104/projects/spm-v4-scale-20261023
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$TASK_ROOT"
exec /srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python scripts/engineering_task.py --root "$TASK_ROOT"
