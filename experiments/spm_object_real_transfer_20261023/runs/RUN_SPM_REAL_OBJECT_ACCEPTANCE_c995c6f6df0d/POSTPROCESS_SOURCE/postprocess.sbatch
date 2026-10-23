#!/bin/bash
#SBATCH --job-name=spm-object-real-accept
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-object-real-accept-%j.log
set -euo pipefail
cd /srv/gpu-lab/users/a25023104/projects/spm-object-real-transfer-20261023
/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python -u presentation_tools/accept_results.py --root .
/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/.venv/bin/python -u presentation_tools/publish_figures.py --root .
