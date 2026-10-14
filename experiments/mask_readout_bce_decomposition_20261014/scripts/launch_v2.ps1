param(
    [ValidateSet('smoke','full','preflight','tests')][string]$Scope = 'preflight',
    [string]$SmokeRun = ''
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$diagnosticRoot = 'D:\coco_wire\baseline_native_20261013'
$diagnosticStudy = Join-Path $diagnosticRoot 'experiments\mask_readout_bce_decomposition_20261014'
$diagnosticPython = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$env:PYTHONPATH = Join-Path $diagnosticRoot '_site'
$env:YOLO_OFFLINE = 'true'
$env:YOLO_AUTOINSTALL = 'false'
$env:YOLO_CONFIG_DIR = Join-Path $diagnosticRoot '_ultralytics_config'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $diagnosticRoot
if ($Scope -eq 'preflight') {
    & $diagnosticPython -B (Join-Path $diagnosticStudy 'scripts\run_bce_decomposition_v2.py') --preflight
    exit $LASTEXITCODE
}
if ($Scope -eq 'tests') {
    & $diagnosticPython -B (Join-Path $diagnosticStudy 'scripts\test_decomposition_v2.py')
    exit $LASTEXITCODE
}
if (Test-Path -LiteralPath (Join-Path $diagnosticStudy '.BCE_DECOMPOSITION.lock')) {
    throw 'Diagnostic stage already has a lock; inspect the actual process before retrying.'
}
$diagnosticStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$diagnosticLogBase = Join-Path $diagnosticStudy ('launch_' + $Scope + '_' + $diagnosticStamp)
$diagnosticArguments = @('-B', (Join-Path $diagnosticStudy 'scripts\run_bce_decomposition_v2.py'), '--scope', $Scope)
if ($Scope -eq 'full') {
    if (-not $SmokeRun) { throw 'Full diagnostic requires an actual completed smoke Run.' }
    $diagnosticArguments += @('--smoke-run', $SmokeRun)
}
$diagnosticProcess = Start-Process -FilePath $diagnosticPython -ArgumentList $diagnosticArguments -WorkingDirectory $diagnosticRoot -WindowStyle Hidden -RedirectStandardOutput ($diagnosticLogBase + '.stdout.log') -RedirectStandardError ($diagnosticLogBase + '.stderr.log') -PassThru
$diagnosticReceipt = [ordered]@{pid=$diagnosticProcess.Id; scope=$Scope; launched_at=(Get-Date -Format o); stdout=($diagnosticLogBase+'.stdout.log'); stderr=($diagnosticLogBase+'.stderr.log'); interpreter=$diagnosticPython; arguments=$diagnosticArguments}
$diagnosticReceipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath ($diagnosticLogBase+'.json') -Encoding UTF8
$diagnosticReceipt | ConvertTo-Json -Depth 6 -Compress
