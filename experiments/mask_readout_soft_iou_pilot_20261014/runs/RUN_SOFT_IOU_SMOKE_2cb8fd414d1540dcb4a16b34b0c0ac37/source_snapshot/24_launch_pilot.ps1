param(
    [ValidateSet('preflight','tests','smoke','full','eval')][string]$Scope = 'preflight',
    [string]$SmokeRun = '',
    [string]$TrainingRun = ''
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$pilotRoot = 'D:\coco_wire\baseline_native_20261013'
$pilotStudy = Join-Path $pilotRoot 'experiments\mask_readout_soft_iou_pilot_20261014'
$pilotPython = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$env:PYTHONPATH = Join-Path $pilotRoot '_site'
$env:YOLO_OFFLINE = 'true'
$env:YOLO_AUTOINSTALL = 'false'
$env:YOLO_CONFIG_DIR = Join-Path $pilotRoot '_ultralytics_config'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $pilotRoot
if ($Scope -eq 'preflight') {
    & $pilotPython -B (Join-Path $pilotStudy 'scripts\run_training.py') --preflight
    exit $LASTEXITCODE
}
if ($Scope -eq 'tests') {
    & $pilotPython -B (Join-Path $pilotStudy 'scripts\test_soft_iou.py')
    exit $LASTEXITCODE
}
if (Test-Path -LiteralPath (Join-Path $pilotStudy '.S2_SOFT_IOU.lock')) {
    throw 'Study has an active lock; inspect existing process and Run.'
}
$pilotOthers = @(Get-CimInstance Win32_Process -Filter "name='python.exe'" | Select-Object ProcessId,CommandLine)
if ($pilotOthers.Count -gt 0) { throw 'Existing Python process requires ownership/status inspection before launch.' }
$pilotStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$pilotLogBase = Join-Path $pilotStudy ('launch_' + $Scope + '_' + $pilotStamp)
if ($Scope -eq 'eval') {
    if (-not $TrainingRun) { throw 'Evaluation requires completed formal training Run.' }
    $pilotArguments = @('-B', (Join-Path $pilotStudy 'scripts\run_evaluation.py'), '--training-run', $TrainingRun)
} else {
    $pilotArguments = @('-B', (Join-Path $pilotStudy 'scripts\run_training.py'), '--scope', $Scope)
    if ($Scope -eq 'full') {
        if (-not $SmokeRun) { throw 'Formal training requires completed smoke Run.' }
        $pilotArguments += @('--smoke-run', $SmokeRun)
    }
}
$pilotProcess = Start-Process -FilePath $pilotPython -ArgumentList $pilotArguments -WorkingDirectory $pilotRoot -WindowStyle Hidden -RedirectStandardOutput ($pilotLogBase+'.stdout.log') -RedirectStandardError ($pilotLogBase+'.stderr.log') -PassThru
$pilotReceipt = [ordered]@{pid=$pilotProcess.Id; scope=$Scope; launched_at=(Get-Date -Format o); stdout=($pilotLogBase+'.stdout.log'); stderr=($pilotLogBase+'.stderr.log'); interpreter=$pilotPython; arguments=$pilotArguments}
$pilotReceipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath ($pilotLogBase+'.json') -Encoding UTF8
$pilotReceipt | ConvertTo-Json -Depth 6 -Compress
