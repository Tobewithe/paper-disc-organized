#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=spm-object-check
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-object-check-%j.out
#SBATCH --error=/srv/gpu-lab/users/a25023104/logs/spm-object-check-%j.err
set -euo pipefail
cd /srv/gpu-lab/users/a25023104/projects/spm-object-prototype-20261023
PY=/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python
"$PY" scripts/check_package.py --root "$PWD"
