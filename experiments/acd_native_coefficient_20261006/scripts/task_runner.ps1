$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'D:/coco_wire/vendor_8.4.100'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:YOLO_OFFLINE = 'true'
$env:YOLO_AUTOINSTALL = 'false'
$experimentRoot = 'D:/coco_wire/experiments/acd_native_coefficient_20261006'
$pipelineLog = Join-Path $experimentRoot 'runs/RUN_ACD_PIPELINE_S0_RETRY1/stdout.log'
Set-Location -LiteralPath $experimentRoot
& 'C:/Users/28358/anaconda3/envs/pytorch/python.exe' -u (Join-Path $experimentRoot 'scripts/run_pipeline.py') --root $experimentRoot --pipeline-run RUN_ACD_PIPELINE_S0_RETRY1 --baseline-smoke-run RUN_BASELINE_SMOKE_S0_RETRY1 2>&1 | Out-File -LiteralPath $pipelineLog -Encoding utf8
exit $LASTEXITCODE
