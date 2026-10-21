#!/bin/bash
# One execution after this campaign ends; not a timer or polling service.
set -euo pipefail
ROOT=/srv/gpu-lab/users/a25023104/projects/paper-disc-organized
exec "$ROOT/.venv/bin/python" "$ROOT/experiments/mask_branch_joint_replay_20261018/closure/close_existing_results.py"
