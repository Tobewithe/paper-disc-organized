#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=spm-object-asset
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-object-asset-%j.out
#SBATCH --error=/srv/gpu-lab/users/a25023104/logs/spm-object-asset-%j.err
set -euo pipefail
export LD_LIBRARY_PATH=/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion/software/runtime-libs/usr/lib64:${LD_LIBRARY_PATH:-}
cd /srv/gpu-lab/users/a25023104/projects/spm-object-prototype-20261023
PY=/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python
"$PY" scripts/prepare_data.py --root "$PWD" --stage asset --device cpu
