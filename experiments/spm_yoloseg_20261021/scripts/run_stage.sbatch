#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
TASK_ROOT=/srv/gpu-lab/users/a25023104/projects/spm-yoloseg-20261022
TASK_PY=/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python
cd "$TASK_ROOT"
exec "$TASK_PY" scripts/contact_probe.py "${SPM_STAGE:?stage is required}" --root "$TASK_ROOT"
