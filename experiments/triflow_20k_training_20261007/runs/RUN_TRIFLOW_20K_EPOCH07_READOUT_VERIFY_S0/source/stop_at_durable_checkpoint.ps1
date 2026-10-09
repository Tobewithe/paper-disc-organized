param([string]$Root, [string]$ControllerRun, [string]$TrainingRun = 'RUN_TRIFLOW_20K_TRAIN_S0')
$ErrorActionPreference = 'Stop'
$oldRun = Join-Path $Root "runs\$TrainingRun"
$controlRun = Join-Path $Root "runs\$ControllerRun"
$started = Get-Date
# Wait for a freshly committed snapshot to avoid replaying minutes of work.
while ($true) {
    $meta = Get-Content -LiteralPath (Join-Path $oldRun 'run.json') -Raw | ConvertFrom-Json
    if ($meta.execution_status -ne 'running') { throw 'Owned training is no longer running; inspect before continuation' }
    $checkpoint = Get-Content -LiteralPath (Join-Path $oldRun 'CHECKPOINT_LATEST.json') -Raw | ConvertFrom-Json
    $age = ((Get-Date).ToUniversalTime() - ([DateTimeOffset]::Parse($checkpoint.created_at).UtcDateTime)).TotalSeconds
    if ($age -ge 0 -and $age -le 3) { break }
    if (((Get-Date) - $started).TotalSeconds -gt 330) { throw 'No fresh durable snapshot within330seconds; original training preserved' }
    Start-Sleep -Milliseconds 200
}
$processRecord = Get-Content -LiteralPath (Join-Path $oldRun 'PROCESS.json') -Raw | ConvertFrom-Json
$ownedSource = Join-Path $oldRun 'source\train_20k.py'
$ownedProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($processRecord.pid)"
if (-not $ownedProcess -or $ownedProcess.Name -ne 'python.exe' -or $ownedProcess.CommandLine -notlike "*$ownedSource*") { throw 'Exact owned process identity did not match' }
$reason = @{reason='user_requested_evaluation_at_every_epoch';requested_at=(Get-Date).ToString('o');next_training_run='RUN_TRIFLOW_20K_TRAIN_S0_R1';training_algorithm_changed=$false;planned_resume_snapshot=$checkpoint}
$reason | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $oldRun 'USER_EPOCH_CONTROL_CHANGE.json') -Encoding utf8
Stop-Process -Id $processRecord.pid -Force
$oldSnapshot = Join-Path $oldRun 'checkpoint_latest.pt'
$stoppedRecord = Get-Content -LiteralPath (Join-Path $oldRun 'CHECKPOINT_LATEST.json') -Raw | ConvertFrom-Json
$digest = (Get-FileHash -LiteralPath $oldSnapshot -Algorithm SHA256).Hash.ToLowerInvariant()
if ($digest -ne $stoppedRecord.sha256) { throw 'Latest durable payload and receipt differ; preserve and inspect' }
Copy-Item -LiteralPath $oldSnapshot -Destination (Join-Path $controlRun 'resume_snapshot.pt')
$copied = (Get-FileHash -LiteralPath (Join-Path $controlRun 'resume_snapshot.pt') -Algorithm SHA256).Hash.ToLowerInvariant()
if ($copied -ne $digest) { throw 'Copied durable payload differs' }
$receipt = @{passed=$true;source_run=$TrainingRun;source_pid=$processRecord.pid;stopped_at=(Get-Date).ToString('o');checkpoint_sha256=$digest;checkpoint_receipt=$stoppedRecord;resume_snapshot=(Join-Path $controlRun 'resume_snapshot.pt');scientific_training_budget=8;resume_semantics='Committed prefix retained; any post-snapshot updates may be replayed';history_preserved=$true}
$receipt | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath (Join-Path $controlRun 'CONTINUATION_RECEIPT.json') -Encoding utf8
$receipt | ConvertTo-Json -Depth 5 -Compress
