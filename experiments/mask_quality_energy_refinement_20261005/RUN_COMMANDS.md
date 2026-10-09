# Commands reserved for the CUDA server

All commands below run under the server's conda `pytorch` environment and are wrapped by the copied `runner.py` from the workbench. They are not run on the desktop.

```bash
ROOT=/root/mask_quality_energy_refinement_20261005
PY=/opt/conda/envs/pytorch/bin/python
CFG=$ROOT/RUN_CONFIG.json

$PY $ROOT/scripts/prepare_qcr_cache.py --config $CFG --out $ROOT/runs/RUN_prepare_seed0
$PY $ROOT/scripts/select_candidates.py --config $CFG --out $ROOT/assets/CANDIDATE_MANIFEST.json
$PY $ROOT/scripts/build_oracle_selected.py --config $CFG --manifest $ROOT/assets/CANDIDATE_MANIFEST.json --out $ROOT/assets/ORACLE_failure.pt --batch 4 --max-iter 60

$PY $ROOT/scripts/qcr_train.py --config $CFG --manifest $ROOT/assets/CANDIDATE_MANIFEST.json --stage train --mode direct --out $ROOT/runs/RUN_stage1_direct_seed0
$PY $ROOT/scripts/qcr_train.py --config $CFG --manifest $ROOT/assets/CANDIDATE_MANIFEST.json --stage train --mode quality --out $ROOT/runs/RUN_stage1_quality_seed0

$PY $ROOT/scripts/qcr_train.py --config $CFG --manifest $ROOT/assets/CANDIDATE_MANIFEST.json --stage eval --mode direct --checkpoint $ROOT/runs/RUN_stage1_direct_seed0/final.pt --out $ROOT/runs/RUN_stage1_direct_dev_seed0 --split dev
$PY $ROOT/scripts/qcr_train.py --config $CFG --manifest $ROOT/assets/CANDIDATE_MANIFEST.json --stage eval --mode quality --checkpoint $ROOT/runs/RUN_stage1_quality_seed0/final.pt --out $ROOT/runs/RUN_stage1_quality_dev_seed0 --split dev
```

Before the first command, verify the server has the COCO2017 ZIPs/annotations, the vendored Ultralytics 8.4.100 tree and the official checkpoint SHA. If any prerequisite fails, keep the failed preparation Run and stop; do not silently substitute another model, cache or machine.
