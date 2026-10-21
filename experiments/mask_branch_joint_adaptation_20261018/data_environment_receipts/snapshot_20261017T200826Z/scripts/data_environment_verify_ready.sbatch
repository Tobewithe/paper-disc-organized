#!/bin/bash
#SBATCH --job-name=pc-input-verify
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/pc-input-verify-%j.log
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /srv/gpu-lab/users/a25023104/projects/paper-disc-organized
python3 experiments/mask_branch_joint_adaptation_20261018/scripts/data_environment_verify_ready.py
