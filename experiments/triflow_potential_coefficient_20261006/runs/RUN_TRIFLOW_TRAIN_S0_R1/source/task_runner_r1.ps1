$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:YOLO_OFFLINE = 'true'
$env:YOLO_AUTOINSTALL = 'false'
$triFlowRoot = 'D:/coco_wire/experiments/triflow_potential_coefficient_20261006'
$triFlowPipeline = Join-Path $triFlowRoot 'runs/RUN_TRIFLOW_PIPELINE_S0_R1'
Set-Location -LiteralPath $triFlowRoot
& 'C:/Users/28358/anaconda3/envs/pytorch/python.exe' -u (Join-Path $triFlowRoot 'scripts/run_stage.py') --root $triFlowRoot --run-id RUN_TRIFLOW_PIPELINE_S0_R1 --purpose fixed_formal_training_and_evaluation_pipeline --script run_pipeline.py -- --root $triFlowRoot --run-id RUN_TRIFLOW_PIPELINE_S0_R1 2>&1 | Out-File -LiteralPath (Join-Path $triFlowPipeline 'scheduler_stdout.log') -Encoding utf8
exit $LASTEXITCODE
