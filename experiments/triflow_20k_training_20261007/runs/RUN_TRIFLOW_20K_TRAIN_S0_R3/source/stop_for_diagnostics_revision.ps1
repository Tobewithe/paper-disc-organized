param([string]$Root, [string]$ControllerRun)
$ErrorActionPreference = 'Stop'
function FileSHA([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try { return [System.BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
    finally { $hasher.Dispose(); $stream.Dispose() }
}
$resolvedRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
if ($resolvedRoot -ne 'D:\coco_wire\experiments\triflow_20k_training_20261007') { throw 'Unexpected experiment root' }
if ($ControllerRun -ne 'RUN_TRIFLOW_20K_DIAGNOSTICS_REVISION_S0') { throw 'Unexpected controller Run' }
$oldRun = Join-Path $resolvedRoot 'runs\RUN_TRIFLOW_20K_TRAIN_S0_R2'
$controlRun = Join-Path $resolvedRoot "runs\$ControllerRun"
$evalRun = Join-Path $resolvedRoot 'runs\RUN_TRIFLOW_20K_EPOCH02_EVAL_S0'
$verifyRun = Join-Path $resolvedRoot 'runs\RUN_TRIFLOW_20K_EPOCH02_READOUT_VERIFY_S0'
$callbackPath = Join-Path $evalRun 'OBSERVER_EPOCH_COMPLETE.json'
$verificationPath = Join-Path $verifyRun 'READOUT_VERIFICATION.json'
$callbackWaitBegan = Get-Date
while (-not (Test-Path -LiteralPath $callbackPath) -or -not (Test-Path -LiteralPath $verificationPath)) {
    if (Test-Path -LiteralPath (Join-Path $evalRun 'OBSERVER_EPOCH_FAILURE.json')) { throw 'Epoch2 callback failed; preserve for inspection' }
    if (((Get-Date)-$callbackWaitBegan).TotalSeconds -gt 900) { throw 'Epoch2 callback has not finished; owned observer preserved' }
    Start-Sleep -Seconds 1
}
$callback = Get-Content -LiteralPath $callbackPath -Raw | ConvertFrom-Json
$verification = Get-Content -LiteralPath $verificationPath -Raw | ConvertFrom-Json
if (-not $callback.passed -or -not $verification.passed) { throw 'Epoch2 callback including verifier must pass before stopping' }
$began = Get-Date
while ($true) {
    $meta = Get-Content -LiteralPath (Join-Path $oldRun 'run.json') -Raw | ConvertFrom-Json
    if ($meta.execution_status -ne 'running') { throw 'Current owned observer is not running' }
    $checkpoint = Get-Content -LiteralPath (Join-Path $oldRun 'CHECKPOINT_LATEST.json') -Raw | ConvertFrom-Json
    if ($checkpoint.next_cursor.epoch -gt 3) { throw 'Next epoch callback may be active; preserve observer for inspection' }
    $age = ((Get-Date).ToUniversalTime() - ([DateTimeOffset]::Parse($checkpoint.created_at).UtcDateTime)).TotalSeconds
    if ($checkpoint.next_cursor.epoch -eq 3 -and $age -ge 0 -and $age -le 3) { break }
    if (((Get-Date)-$began).TotalSeconds -gt 330) { throw 'No fresh durable checkpoint; preserve current observer' }
    Start-Sleep -Milliseconds 200
}
$processRecord = Get-Content -LiteralPath (Join-Path $oldRun 'PROCESS.json') -Raw | ConvertFrom-Json
$ownedSource = (Join-Path $oldRun 'source\train_epoch_observer.py').Replace('/','\').ToLowerInvariant()
$ownedProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($processRecord.pid)"
if (-not $ownedProcess -or $ownedProcess.Name -ne 'python.exe' -or -not $ownedProcess.CommandLine.Replace('/','\').ToLowerInvariant().Contains($ownedSource)) { throw 'Exact owned observer PID/source identity differs' }
$request = @{reason='user_requested_minimal_diagnostics';requested_at=(Get-Date).ToString('o');next_training_run='RUN_TRIFLOW_20K_TRAIN_S0_R3';original_source4_bytes_preserved=$true;runtime_diagnostic_calculation_changed=$true;checkpoint=$checkpoint;epoch2_callback_completed=$true}
$request | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath (Join-Path $oldRun 'DIAGNOSTICS_REVISION_REQUEST.json') -Encoding utf8
Stop-Process -Id $processRecord.pid -Force
$payload = Join-Path $oldRun 'checkpoint_latest.pt'
$durable = Get-Content -LiteralPath (Join-Path $oldRun 'CHECKPOINT_LATEST.json') -Raw | ConvertFrom-Json
if ((FileSHA $payload) -ne $durable.sha256) { throw 'Actual durable payload differs from receipt' }
$target = Join-Path $controlRun 'resume_snapshot.pt'
if (Test-Path -LiteralPath $target) { throw 'Immutable resume target already exists' }
Copy-Item -LiteralPath $payload -Destination $target
if ((FileSHA $target) -ne $durable.sha256) { throw 'Copied continuation payload differs' }
$receipt = @{passed=$true;source_run='RUN_TRIFLOW_20K_TRAIN_S0_R2';next_training_run='RUN_TRIFLOW_20K_TRAIN_S0_R3';snapshot_sha256=$durable.sha256;checkpoint_receipt=$durable;resume_snapshot=$target;observed_at=(Get-Date).ToString('o');original_source4_bytes_preserved=$true;runtime_diagnostic_calculation_changed=$true;history_preserved=$true;epoch2_callback_completed=$true;scope='User-authorized diagnostic-only execution revision; persisted head/AdamW/all RNG/order/cursor retained, equivalence verification required before new launch'}
$receipt | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath (Join-Path $controlRun 'CONTINUATION_RECEIPT.json') -Encoding utf8
$receipt | ConvertTo-Json -Depth 4 -Compress
