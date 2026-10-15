param([ValidateSet('preflight','tests','smoke','train','eval')][string]$Scope='preflight',[string]$PriorRun='')
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$e2Root='D:\coco_wire\baseline_native_20261013'
$e2Study=Join-Path $e2Root 'experiments\mask_readout_sphere_recovery_20261015'
$e2Python='C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$env:PYTHONPATH=Join-Path $e2Root '_site'
$env:YOLO_OFFLINE='true'
$env:YOLO_AUTOINSTALL='false'
$env:YOLO_CONFIG_DIR=Join-Path $e2Root '_ultralytics_config'
$env:OMP_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
Set-Location -LiteralPath $e2Root
if ($Scope -eq 'tests') { & $e2Python -B (Join-Path $e2Study 'scripts\test_spherical.py');exit $LASTEXITCODE }
if ($Scope -eq 'preflight') { & $e2Python -B (Join-Path $e2Study 'scripts\run_training.py') --preflight;exit $LASTEXITCODE }
if ($Scope -eq 'train') {
    if (-not $PriorRun) { throw 'Completed smoke required' }
    & $e2Python -B (Join-Path $e2Study 'scripts\run_training.py') --scope full --smoke-run $PriorRun
} elseif ($Scope -eq 'eval') {
    if (-not $PriorRun) { throw 'Completed training required' }
    & $e2Python -B (Join-Path $e2Study 'scripts\run_evaluation.py') --training-run $PriorRun
} else {
    & $e2Python -B (Join-Path $e2Study 'scripts\run_training.py') --scope smoke
}
exit $LASTEXITCODE
