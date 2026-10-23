#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=spm-instance-opportunity
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-instance-opportunity-%j.log
set -euo pipefail
SPM_ROOT=/srv/gpu-lab/users/a25023104/projects/spm-instance-readout-opportunity-20261023
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$SPM_ROOT"
exec /srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python scripts/run_opportunity.py --root "$SPM_ROOT"
