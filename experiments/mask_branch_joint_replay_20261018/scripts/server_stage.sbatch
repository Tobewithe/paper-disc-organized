#!/bin/bash
# This file runs one finite, explicitly submitted stage. It is not a timer.
# Resources, log path, dependency and wall limit are supplied by sbatch at submission.
set -euo pipefail
ROOT=/srv/gpu-lab/users/a25023104/projects/paper-disc-organized
STUDY=$ROOT/experiments/mask_branch_joint_replay_20261018
PY=$ROOT/.venv/bin/python
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export YOLO_CONFIG_DIR=/srv/gpu-lab/users/a25023104/.cache/ultralytics-joint-20261018
cd "$ROOT"
stage=${1:?stage required}
"$PY" "$STUDY/scripts/check_frozen.py" --config "$STUDY/CONFIG.json" --require-ready
case "$stage" in
  gate)
    "$PY" "$STUDY/scripts/joint_runner.py" --config "$STUDY/CONFIG.json" prepare
    "$PY" "$STUDY/scripts/joint_runner.py" --config "$STUDY/CONFIG.json" native
    "$PY" "$STUDY/scripts/joint_runner.py" --config "$STUDY/CONFIG.json" smoke
    ;;
  G_budget|C_native|PC_native)
    "$PY" "$STUDY/scripts/joint_runner.py" --config "$STUDY/CONFIG.json" train --arm "$stage"
    ;;
  evaluation)
    "$PY" "$STUDY/scripts/joint_runner.py" --config "$STUDY/CONFIG.json" eval --arm G_budget
    "$PY" "$STUDY/scripts/joint_runner.py" --config "$STUDY/CONFIG.json" eval --arm C_native
    "$PY" "$STUDY/scripts/joint_runner.py" --config "$STUDY/CONFIG.json" eval --arm PC_native
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    "$PY" "$STUDY/scripts/joint_statistics.py" --config "$STUDY/CONFIG.json" --workers 4
    ;;
  *) echo "Unknown stage: $stage" >&2; exit 64 ;;
esac
