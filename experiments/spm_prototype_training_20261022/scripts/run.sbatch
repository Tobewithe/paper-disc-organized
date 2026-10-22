#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-prototype-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
TASK_ROOT=/srv/gpu-lab/users/a25023104/projects/spm-prototype-training-20261022
TASK_PY=/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python
cd "$TASK_ROOT"
exec "$TASK_PY" scripts/prototype_train.py "${SPM_STAGE:?stage required}" --root "$TASK_ROOT"
