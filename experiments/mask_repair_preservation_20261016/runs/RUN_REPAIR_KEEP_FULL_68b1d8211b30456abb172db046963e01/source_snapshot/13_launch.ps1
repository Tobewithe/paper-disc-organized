param(
    [Parameter(Mandatory=$true)][ValidateSet('smoke','full')][string]$Scope,
    [string]$ParentRun
)
$ErrorActionPreference = 'Stop'
$taskRoot = 'D:\coco_wire\baseline_native_20261013'
$studyRoot = Join-Path $taskRoot 'experiments\mask_repair_preservation_20261016'
Set-Location -LiteralPath $taskRoot
$env:PYTHONPATH = Join-Path $taskRoot '_site'
$env:YOLO_OFFLINE = 'true'
$env:YOLO_AUTOINSTALL = 'false'
$env:YOLO_CONFIG_DIR = Join-Path $taskRoot '_ultralytics_config'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
Remove-Item Env:MKL_NUM_THREADS -ErrorAction SilentlyContinue
$taskArgs = @('-B', (Join-Path $studyRoot 'scripts\run_mechanism.py'), '--scope', $Scope)
if ($ParentRun) { $taskArgs += @('--parent-run', $ParentRun) }
& 'C:\Users\28358\anaconda3\envs\pytorch\python.exe' @taskArgs
exit $LASTEXITCODE
