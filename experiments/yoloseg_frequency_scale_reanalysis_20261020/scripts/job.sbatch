#!/bin/bash
#SBATCH --job-name=FREQ-S2-scale
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/freq-s2-%j.out
#SBATCH --error=/srv/gpu-lab/users/a25023104/logs/freq-s2-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
cd /srv/gpu-lab/users/a25023104/projects/paper-disc-organized/experiments/yoloseg_frequency_scale_reanalysis_20261020
PY=/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python
"$PY" -u scripts/analyze.py smoke
"$PY" -u scripts/analyze.py full
