param([ValidateSet('preflight','tests','smoke','full')][string]$Scope='preflight',[string]$SmokeRun='')
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$repulsionRoot='D:\coco_wire\baseline_native_20261013'
$repulsionStudy=Join-Path $repulsionRoot 'experiments\mask_readout_subspace_repulsion_20261015'
$repulsionPython='C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$env:PYTHONPATH=Join-Path $repulsionRoot '_site'
$env:YOLO_OFFLINE='true'
$env:YOLO_AUTOINSTALL='false'
$env:YOLO_CONFIG_DIR=Join-Path $repulsionRoot '_ultralytics_config'
$env:OMP_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
Set-Location -LiteralPath $repulsionRoot
if ($Scope -eq 'tests') { & $repulsionPython -B (Join-Path $repulsionStudy 'scripts\test_repulsion.py');exit $LASTEXITCODE }
if ($Scope -eq 'preflight') { & $repulsionPython -B (Join-Path $repulsionStudy 'scripts\run_repulsion.py') --preflight;exit $LASTEXITCODE }
if ($Scope -eq 'full') {
    if (-not $SmokeRun) { throw 'Completed smoke required' }
    & $repulsionPython -B (Join-Path $repulsionStudy 'scripts\run_repulsion.py') --scope full --smoke-run $SmokeRun
} else {
    & $repulsionPython -B (Join-Path $repulsionStudy 'scripts\run_repulsion.py') --scope smoke
}
exit $LASTEXITCODE
