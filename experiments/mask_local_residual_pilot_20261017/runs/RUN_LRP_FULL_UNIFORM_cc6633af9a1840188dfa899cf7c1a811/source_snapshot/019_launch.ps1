param(
    [Parameter(Mandatory=$true)][ValidateSet('test','preflight','smoke','verify')][string]$Stage,
    [ValidateSet('REAL','PERM')][string]$Arm,
    [string]$PreflightRun,
    [string]$VerifyRun
)
$ErrorActionPreference = 'Stop'
$taskRoot = 'D:\coco_wire\baseline_native_20261013'
$taskPython = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$taskStudy = Join-Path $taskRoot 'experiments\mask_direction_learnability_20261016'
Set-Location -LiteralPath $taskRoot
$env:PYTHONPATH = Join-Path $taskRoot '_site'
$env:YOLO_OFFLINE = 'true'
$env:YOLO_AUTOINSTALL = 'false'
$env:YOLO_CONFIG_DIR = Join-Path $taskRoot '_ultralytics_config'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
Remove-Item Env:MKL_NUM_THREADS -ErrorAction SilentlyContinue
if ($Stage -eq 'test') {
    & $taskPython -B (Join-Path $taskStudy 'scripts\test_cld.py')
} elseif ($Stage -eq 'preflight') {
    & $taskPython -B (Join-Path $taskStudy 'scripts\preflight.py')
} elseif ($Stage -eq 'smoke') {
    if (-not $Arm -or -not $PreflightRun) { throw 'Arm and PreflightRun are required' }
    & $taskPython -B (Join-Path $taskStudy 'scripts\train_cld.py') --scope smoke --arm $Arm --preflight (Join-Path $taskStudy ('runs\' + $PreflightRun))
} elseif ($Stage -eq 'verify') {
    if (-not $VerifyRun) { throw 'VerifyRun is required' }
    & $taskPython -B (Join-Path $taskStudy 'scripts\verify_cld.py') (Join-Path $taskStudy ('runs\' + $VerifyRun))
}
exit $LASTEXITCODE
