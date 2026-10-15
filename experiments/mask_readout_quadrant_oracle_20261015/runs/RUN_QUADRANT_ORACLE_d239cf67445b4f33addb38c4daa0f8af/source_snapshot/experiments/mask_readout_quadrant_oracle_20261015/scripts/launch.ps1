param([ValidateSet('tests','smoke','full')][string]$Scope='tests',[string]$SmokeRun='')
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$e3Root='D:\coco_wire\baseline_native_20261013'
$e3Study=Join-Path $e3Root 'experiments\mask_readout_quadrant_oracle_20261015'
$e3Python='C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$env:PYTHONPATH=Join-Path $e3Root '_site'
$env:YOLO_OFFLINE='true'
$env:YOLO_AUTOINSTALL='false'
$env:YOLO_CONFIG_DIR=Join-Path $e3Root '_ultralytics_config'
$env:OMP_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
Set-Location -LiteralPath $e3Root
if ($Scope -eq 'tests') { & $e3Python -B (Join-Path $e3Study 'scripts\test_quadrant.py');exit $LASTEXITCODE }
if ($Scope -eq 'full') {
    if (-not $SmokeRun) { throw 'Completed smoke required' }
    & $e3Python -B (Join-Path $e3Study 'scripts\run_oracle.py') --scope full --smoke-run $SmokeRun
} else { & $e3Python -B (Join-Path $e3Study 'scripts\run_oracle.py') --scope smoke }
exit $LASTEXITCODE
