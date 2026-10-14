param(
    [ValidateSet('smoke','full','preflight','tests')][string]$Scope = 'preflight',
    [string]$SmokeRun = ''
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$auditRoot = 'D:\coco_wire\baseline_native_20261013'
$auditStudy = Join-Path $auditRoot 'experiments\mask_readout_benefit_audit_20261014'
$auditPython = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$env:PYTHONPATH = Join-Path $auditRoot '_site'
$env:YOLO_OFFLINE = 'true'
$env:YOLO_AUTOINSTALL = 'false'
$env:YOLO_CONFIG_DIR = Join-Path $auditRoot '_ultralytics_config'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $auditRoot
if ($Scope -eq 'tests') {
    & $auditPython -B (Join-Path $auditStudy 'scripts\test_audit.py')
    exit $LASTEXITCODE
}
if ($Scope -eq 'preflight') {
    & $auditPython -B (Join-Path $auditStudy 'scripts\run_audit.py') --preflight
    exit $LASTEXITCODE
}
$auditStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$auditLogBase = Join-Path $auditStudy ('launch_' + $Scope + '_' + $auditStamp)
$auditArguments = @('-B', (Join-Path $auditStudy 'scripts\run_audit.py'), '--scope', $Scope)
if ($Scope -eq 'full') {
    if (-not $SmokeRun) { throw 'Full diagnostic requires a completed smoke Run.' }
    $auditArguments += @('--smoke-run', $SmokeRun)
}
$auditProcess = Start-Process -FilePath $auditPython -ArgumentList $auditArguments -WorkingDirectory $auditRoot -WindowStyle Hidden -RedirectStandardOutput ($auditLogBase + '.stdout.log') -RedirectStandardError ($auditLogBase + '.stderr.log') -PassThru
$auditReceipt = [ordered]@{pid=$auditProcess.Id; scope=$Scope; launched_at=(Get-Date -Format o); stdout=($auditLogBase+'.stdout.log'); stderr=($auditLogBase+'.stderr.log'); interpreter=$auditPython; arguments=$auditArguments}
$auditReceipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath ($auditLogBase+'.json') -Encoding UTF8
$auditReceipt | ConvertTo-Json -Depth 6 -Compress
