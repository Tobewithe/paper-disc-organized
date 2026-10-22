#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:03:00
#SBATCH --job-name=spm-rig-inspect
#SBATCH --output=/srv/gpu-lab/users/a25023104/logs/spm-farm-inspect-%j.log
set -euo pipefail
TASK_ROOT=/srv/gpu-lab/users/a25023104/projects/spm-farm-scenes-20261023
BLENDER=/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion/software/blender-4.2.1-linux-x64/blender
export LD_LIBRARY_PATH=/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion/software/runtime-libs/usr/lib64:${LD_LIBRARY_PATH:-}
cd "$TASK_ROOT"
exec "$BLENDER" -b --factory-startup -t 1 --python scripts/inspect_asset.py -- "$TASK_ROOT/ASSET_INSPECTION.json"
