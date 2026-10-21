#!/bin/bash
#SBATCH --job-name=pc-data-range
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/pc-data-range-%j.log
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /srv/gpu-lab/users/a25023104/projects/paper-disc-organized
python3 experiments/mask_branch_joint_adaptation_20261018/scripts/data_environment_recover_train.py
