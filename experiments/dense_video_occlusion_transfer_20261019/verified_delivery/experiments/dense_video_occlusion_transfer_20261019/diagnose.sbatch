#!/bin/bash
set -euo pipefail
export LD_LIBRARY_PATH=/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion/software/runtime-libs/usr/lib64
/srv/gpu-lab/users/a25023104/projects/dense-video-occlusion/software/blender-4.2.1-linux-x64/blender --background --factory-startup --python-exit-code 1 --python /srv/gpu-lab/users/a25023104/projects/dense-video-occlusion-v2/shared/dense_video_occlusion/diagnose_geometry.py
