#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=spm-task-method
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-task-method-%A_%a.log
set -euo pipefail
TASK_ROOT=/srv/gpu-lab/users/a25023104/projects/spm-v4-scale-20261023
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ "${1:-}" == "smoke" ]]; then
  STAGE=smoke
else
  ARMS=(continued adapter random_memory task_spm)
  STAGE="${ARMS[${SLURM_ARRAY_TASK_ID:?}]}"
fi
cd "$TASK_ROOT"
exec /srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python scripts/run_task.py "$STAGE" --root "$TASK_ROOT"
