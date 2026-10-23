#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=spm-instance-acceptance
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-instance-acceptance-%j.log
set -euo pipefail
SPM_ROOT=/srv/gpu-lab/users/a25023104/projects/spm-instance-readout-opportunity-20261023
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$SPM_ROOT"
exec /srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python scripts/acceptance.py --root "$SPM_ROOT"
