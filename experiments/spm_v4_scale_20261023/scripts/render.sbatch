#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=spm-scale-render
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-scale-render-%A_%a.log
set -euo pipefail
TASK_ROOT=/srv/gpu-lab/users/a25023104/projects/spm-v4-scale-20261023
TASK_PY=/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python
export LD_LIBRARY_PATH=/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion/software/runtime-libs/usr/lib64:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
if [[ "${1:-}" == "smoke" ]]; then
  CHILD_ROOT="$TASK_ROOT/render_plans/smoke"
  STAGE=smoke
else
  CHILD_ROOT="$TASK_ROOT/render_plans/shard_${SLURM_ARRAY_TASK_ID:?}"
  STAGE=formal
fi
cd "$CHILD_ROOT"
exec "$TASK_PY" scripts/run_render.py "$STAGE" --root "$CHILD_ROOT"
