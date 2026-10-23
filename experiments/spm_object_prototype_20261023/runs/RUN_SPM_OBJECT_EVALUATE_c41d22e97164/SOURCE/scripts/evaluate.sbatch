#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=spm-object-eval
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-object-eval-%j.out
#SBATCH --error=/srv/gpu-lab/users/a25023104/logs/spm-object-eval-%j.err
set -euo pipefail
cd /srv/gpu-lab/users/a25023104/projects/spm-object-prototype-20261023
PY=/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python
"$PY" scripts/evaluate_objects.py --root "$PWD"
