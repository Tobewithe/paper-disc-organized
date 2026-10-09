$ErrorActionPreference = 'Stop'
function FileSHA([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try { return [System.BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
    finally { $hasher.Dispose(); $stream.Dispose() }
}
$epochRoot = 'D:\coco_wire\experiments\triflow_20k_training_20261007'
$epochTask = 'Codex_TriFlow_20K_EpochEval_R3_20261007_01a11060'
$epochRun = Join-Path $epochRoot 'runs\RUN_TRIFLOW_20K_PIPELINE_S0_R3'
if (Get-ScheduledTask -TaskName $epochTask -ErrorAction SilentlyContinue) { throw 'Owned R3 task already exists' }
$control = Get-Content -LiteralPath (Join-Path $epochRoot 'runs\RUN_TRIFLOW_20K_DIAGNOSTICS_REVISION_S0\CONTINUATION_RECEIPT.json') -Raw | ConvertFrom-Json
$parity = Get-Content -LiteralPath (Join-Path $epochRoot 'runs\RUN_TRIFLOW_20K_DIAGNOSTICS_PARITY_S0\DIAGNOSTICS_PARITY.json') -Raw | ConvertFrom-Json
$engineering = Get-Content -LiteralPath (Join-Path $epochRoot 'runs\RUN_TRIFLOW_20K_DIAGNOSTICS_OBSERVER_ENGINEERING_VERIFY_S0\DIAGNOSTICS_OBSERVER_ENGINEERING_VERIFICATION.json') -Raw | ConvertFrom-Json
$readout = Get-Content -LiteralPath (Join-Path $epochRoot 'runs\RUN_TRIFLOW_20K_DIAGNOSTICS_EVAL_PARITY_S0\DIAGNOSTICS_EVAL_PARITY.json') -Raw | ConvertFrom-Json
if (-not $control.passed -or -not $control.epoch2_callback_completed -or -not $parity.passed -or -not $parity.smoke_only) { throw 'Actual durable handoff and diagnostic equivalence evidence required' }
if (-not $engineering.passed -or (FileSHA (Join-Path $epochRoot 'scripts\train_epoch_observer_r3.py')) -ne $engineering.observer_source_sha256) { throw 'Actual R3 observer trajectory proof required' }
if (-not $readout.passed -or -not $readout.smoke_only) { throw 'Actual minimal/full original-pixel readout parity required' }
$snapshotPath = Join-Path $epochRoot 'runs\RUN_TRIFLOW_20K_DIAGNOSTICS_REVISION_S0\resume_snapshot.pt'
if ((FileSHA $snapshotPath) -ne $control.snapshot_sha256) { throw 'Durable continuation snapshot changed' }
$oldTaskName = 'Codex_TriFlow_20K_EpochEval_R2_20261007_01a11060'
$oldTask = Get-ScheduledTask -TaskName $oldTaskName -ErrorAction SilentlyContinue
if ($oldTask) {
    if ($oldTask.State -ne 'Ready') { throw 'Original owned task has not settled' }
    if (-not ($oldTask.Actions.Arguments -like '*task_epoch_observer_r2.ps1*')) { throw 'Old task action differs from owned execution' }
    Unregister-ScheduledTask -TaskName $oldTaskName -Confirm:$false
}
New-Item -ItemType Directory -Path $epochRun -Force | Out-Null
$epochUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$epochArgs = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File D:\coco_wire\experiments\triflow_20k_training_20261007\scripts\task_epoch_observer_r3.ps1'
$action = New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument $epochArgs -WorkingDirectory $epochRoot
$principal = New-ScheduledTaskPrincipal -UserId $epochUser -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -Hidden -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Days 3)
Register-ScheduledTask -TaskName $epochTask -Action $action -Principal $principal -Settings $settings -Description 'Fixed eight epochs continued with verified minimal diagnostics and full5000 evaluation at every actual epoch boundary' | Out-Null
Start-ScheduledTask -TaskName $epochTask
$receipt = @{task_name=$epochTask;requested_at=(Get-Date).ToString('o');command=$epochArgs;resume_snapshot_sha256=$control.snapshot_sha256;diagnostics_parity_receipt_sha256=(FileSHA (Join-Path $epochRoot 'runs\RUN_TRIFLOW_20K_DIAGNOSTICS_PARITY_S0\DIAGNOSTICS_PARITY.json'));training_images=20000;total_epochs=8;eval_images_each_epoch=5000;old_owned_task_removed=$true;execution_limit_days=3;runtime_diagnostic_calculation_changed=$true;scientific_training_contract_changed=$false}
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $epochRun 'SCHEDULER_LAUNCH.json') -Encoding utf8
$receipt | ConvertTo-Json -Compress
