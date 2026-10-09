param([string]$Root, [string]$ControllerRun)
$ErrorActionPreference = 'Stop'
function FileSHA([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try { return [System.BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
    finally { $hasher.Dispose(); $stream.Dispose() }
}
$oldRun = Join-Path $Root 'runs\RUN_TRIFLOW_20K_TRAIN_S0_R1'
$controlRun = Join-Path $Root "runs\$ControllerRun"
$began = Get-Date
while ($true) {
    $meta = Get-Content -LiteralPath (Join-Path $oldRun 'run.json') -Raw | ConvertFrom-Json
    if ($meta.execution_status -ne 'running') { throw 'Original observer is not running; preserve and inspect' }
    $checkpoint = Get-Content -LiteralPath (Join-Path $oldRun 'CHECKPOINT_LATEST.json') -Raw | ConvertFrom-Json
    $age = ((Get-Date).ToUniversalTime() - ([DateTimeOffset]::Parse($checkpoint.created_at).UtcDateTime)).TotalSeconds
    if ($age -ge 0 -and $age -le 3) { break }
    if (((Get-Date)-$began).TotalSeconds -gt 330) { throw 'No new durable snapshot; original observer preserved' }
    Start-Sleep -Milliseconds 200
}
$processRecord = Get-Content -LiteralPath (Join-Path $oldRun 'PROCESS.json') -Raw | ConvertFrom-Json
$ownedSource = Join-Path $oldRun 'source\train_epoch_observer.py'
$ownedProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($processRecord.pid)"
if (-not $ownedProcess -or $ownedProcess.Name -ne 'python.exe' -or $ownedProcess.CommandLine -notlike "*$ownedSource*") { throw 'Exact owned observer PID/source identity differs' }
@{reason='repair_epoch_callback_metadata_schema_before_first_evaluation';requested_at=(Get-Date).ToString('o');next_training_run='RUN_TRIFLOW_20K_TRAIN_S0_R2';scientific_source4_unchanged=$true;checkpoint=$checkpoint} | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath (Join-Path $oldRun 'CALLBACK_SCHEMA_REVISION.json') -Encoding utf8
Stop-Process -Id $processRecord.pid -Force
$payload = Join-Path $oldRun 'checkpoint_latest.pt'
$durable = Get-Content -LiteralPath (Join-Path $oldRun 'CHECKPOINT_LATEST.json') -Raw | ConvertFrom-Json
if ((FileSHA $payload) -ne $durable.sha256) { throw 'Actual durable payload differs from receipt' }
$target = Join-Path $controlRun 'resume_snapshot.pt'
Copy-Item -LiteralPath $payload -Destination $target
if ((FileSHA $target) -ne $durable.sha256) { throw 'Actual copied continuation snapshot differs' }
$receipt=@{passed=$true;source_run='RUN_TRIFLOW_20K_TRAIN_S0_R1';snapshot_sha256=$durable.sha256;checkpoint_receipt=$durable;resume_snapshot=$target;observed_at=(Get-Date).ToString('o');source4_unchanged=$true;history_preserved=$true;scope='Callback metadata/filename correction only; optimizer/order/all RNG durable prefix retained'}
$receipt | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath (Join-Path $controlRun 'CONTINUATION_RECEIPT.json') -Encoding utf8
$receipt | ConvertTo-Json -Depth 3 -Compress
