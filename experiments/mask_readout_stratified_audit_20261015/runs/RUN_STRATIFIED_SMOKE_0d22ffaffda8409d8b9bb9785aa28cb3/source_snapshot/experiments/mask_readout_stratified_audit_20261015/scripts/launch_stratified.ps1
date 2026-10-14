param([ValidateSet('preflight','tests','smoke','full')][string]$Scope='preflight',[string]$SmokeRun='')
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$strataRoot='D:\coco_wire\baseline_native_20261013'
$strataStudy=Join-Path $strataRoot 'experiments\mask_readout_stratified_audit_20261015'
$strataPython='C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$env:PYTHONPATH=Join-Path $strataRoot '_site'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:YOLO_OFFLINE='true'
$env:YOLO_AUTOINSTALL='false'
Set-Location -LiteralPath $strataRoot
if ($Scope -eq 'tests') { & $strataPython -B (Join-Path $strataStudy 'scripts\test_stratified.py'); exit $LASTEXITCODE }
if ($Scope -eq 'preflight') { & $strataPython -B (Join-Path $strataStudy 'scripts\run_stratified.py') --preflight; exit $LASTEXITCODE }
if ($Scope -eq 'full') {
    if (-not $SmokeRun) { throw 'Full analysis requires a completed smoke Run.' }
    & $strataPython -B (Join-Path $strataStudy 'scripts\run_stratified.py') --scope $Scope --smoke-run $SmokeRun
} else {
    & $strataPython -B (Join-Path $strataStudy 'scripts\run_stratified.py') --scope $Scope
}
exit $LASTEXITCODE
