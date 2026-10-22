#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --job-name=spm-v4-export
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-v4-export-%j.log
set -euo pipefail
TASK_ROOT=/srv/gpu-lab/users/a25023104/projects/spm-v4-fastpilot-20261023
TASK_PY=/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$TASK_ROOT"
"$TASK_PY" scripts/export_lite.py --root "$TASK_ROOT"
"$TASK_PY" scripts/package_results.py --root "$TASK_ROOT"
"$TASK_PY" scripts/package_results.py --root "$TASK_ROOT" --weights
