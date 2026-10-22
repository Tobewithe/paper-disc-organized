#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=spm-farm-render
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-farm-render-%j.log
set -euo pipefail
TASK_ROOT=/srv/gpu-lab/users/a25023104/projects/spm-farm-scenes-20261023
TASK_PY=/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python
export LD_LIBRARY_PATH=/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion/software/runtime-libs/usr/lib64:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
cd "$TASK_ROOT"
exec "$TASK_PY" scripts/run_render.py "${SPM_FARM_STAGE:?stage required}" --root "$TASK_ROOT"
