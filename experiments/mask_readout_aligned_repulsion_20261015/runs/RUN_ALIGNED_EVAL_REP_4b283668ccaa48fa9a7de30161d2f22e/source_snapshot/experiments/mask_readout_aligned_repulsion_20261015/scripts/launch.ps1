param([ValidateSet('preflight','tests','smoke','train','eval')][string]$Scope='preflight',[string]$PriorRun='',[ValidateSet('R','REP')][string]$Arm='REP')
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$e4Root='D:\coco_wire\baseline_native_20261013'
$e4Study=Join-Path $e4Root 'experiments\mask_readout_aligned_repulsion_20261015'
$e4Python='C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$env:PYTHONPATH=Join-Path $e4Root '_site'
$env:YOLO_OFFLINE='true'
$env:YOLO_AUTOINSTALL='false'
$env:YOLO_CONFIG_DIR=Join-Path $e4Root '_ultralytics_config'
$env:OMP_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
Set-Location -LiteralPath $e4Root
if ($Scope -eq 'tests') { & $e4Python -B (Join-Path $e4Study 'scripts\test_aligned.py');exit $LASTEXITCODE }
if ($Scope -eq 'preflight') { & $e4Python -B (Join-Path $e4Study 'scripts\run_training.py') --preflight;exit $LASTEXITCODE }
if ($Scope -eq 'train') {
    if (-not $PriorRun) { throw 'Completed smoke required' }
    & $e4Python -B (Join-Path $e4Study 'scripts\run_training.py') --scope full --smoke-run $PriorRun --arm $Arm
} elseif ($Scope -eq 'eval') {
    if (-not $PriorRun) { throw 'Completed training required' }
    & $e4Python -B (Join-Path $e4Study 'scripts\run_evaluation.py') --training-run $PriorRun
} else {
    & $e4Python -B (Join-Path $e4Study 'scripts\run_training.py') --scope smoke --arm $Arm
}
exit $LASTEXITCODE
