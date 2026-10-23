#!/bin/bash
#SBATCH --job-name=spm-object-real-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:14:00
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-object-real-smoke-%j.log
set -euo pipefail
cd /srv/gpu-lab/users/a25023104/projects/spm-object-real-transfer-20261023
/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python -u scripts/run_transfer.py --root . --stage engineering
