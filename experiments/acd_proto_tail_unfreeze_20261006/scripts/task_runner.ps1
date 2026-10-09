$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'D:/coco_wire/vendor_8.4.100'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:YOLO_OFFLINE = 'true'
$env:YOLO_AUTOINSTALL = 'false'
$acdProtoExperimentRoot = 'D:/coco_wire/experiments/acd_proto_tail_unfreeze_20261006'
$acdProtoPipelineRoot = Join-Path $acdProtoExperimentRoot 'runs/RUN_PROTO_TAIL_PIPELINE_S0'
Set-Location -LiteralPath $acdProtoExperimentRoot
& 'C:/Users/28358/anaconda3/envs/pytorch/python.exe' -u (Join-Path $acdProtoPipelineRoot 'source/run_pipeline.py') --root $acdProtoExperimentRoot 2>&1 | Out-File -LiteralPath (Join-Path $acdProtoPipelineRoot 'stdout.log') -Encoding utf8
exit $LASTEXITCODE
