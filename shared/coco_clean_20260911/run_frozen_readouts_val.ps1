$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'C:/Dpan/codexproject/paper-disc'
$env:PYTHONPATH = 'C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/local_readout_runtime_20260912/vendor'
$env:YOLO_CONFIG_DIR = 'C:/Dpan/codexproject/paper-disc/experiments/coco_clean_20260911/local_readout_runtime_20260912/settings'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:PYTHONUTF8 = '1'
$taskPython = 'C:/Dpan/envsfiles/CondaData/envs/pytorch/python.exe'
$taskRun = 'experiments/coco_clean_20260911/diagnostics/frozen_readouts_fullval_20260912'
& $taskPython -u experiments/coco_clean_20260911/eval_frozen_readouts_val.py --out $taskRun --images experiments/coco_clean_20260911/local_readout_runtime_20260912/data/images/val2017 --annotation datasets/coco/annotations/instances_val2017.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $taskPython -u experiments/coco_clean_20260911/summarize_frozen_readouts_val.py --run $taskRun
exit $LASTEXITCODE
