param([ValidateSet('preflight','tests','smoke','full')][string]$Scope='preflight',[string]$SmokeRun='')
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$mixedRoot='D:\coco_wire\baseline_native_20261013'
$mixedStudy=Join-Path $mixedRoot 'experiments\mask_readout_large_mask_gate_20261015'
$mixedPython='C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$env:PYTHONPATH=Join-Path $mixedRoot '_site'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:YOLO_OFFLINE='true'
$env:YOLO_AUTOINSTALL='false'
$env:YOLO_CONFIG_DIR=Join-Path $mixedRoot '_ultralytics_config'
Set-Location -LiteralPath $mixedRoot
if ($Scope -eq 'tests') { & $mixedPython -B (Join-Path $mixedStudy 'scripts\test_area_gate.py'); exit $LASTEXITCODE }
if ($Scope -eq 'preflight') { & $mixedPython -B (Join-Path $mixedStudy 'scripts\run_mixed.py') --preflight; exit $LASTEXITCODE }
if ($Scope -eq 'full') {
    if (-not $SmokeRun) { throw 'Successful smoke required' }
    & $mixedPython -B (Join-Path $mixedStudy 'scripts\run_mixed.py') --scope full --smoke-run $SmokeRun
} else {
    & $mixedPython -B (Join-Path $mixedStudy 'scripts\run_mixed.py') --scope smoke
}
exit $LASTEXITCODE
