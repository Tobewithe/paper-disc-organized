$ErrorActionPreference = 'Stop'
function FileSHA([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try { return [System.BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
    finally { $hasher.Dispose(); $stream.Dispose() }
}
$epochRoot = 'D:\coco_wire\experiments\triflow_20k_training_20261007'
$epochTask = 'Codex_TriFlow_20K_EpochEval_20261007_01a11060'
$epochRun = Join-Path $epochRoot 'runs\RUN_TRIFLOW_20K_PIPELINE_S0_R1'
if (Get-ScheduledTask -TaskName $epochTask -ErrorAction SilentlyContinue) { throw 'Owned epoch task already exists; preserve it' }
$control = Get-Content -LiteralPath (Join-Path $epochRoot 'runs\RUN_TRIFLOW_20K_EPOCH_CONTINUATION_S0_R1\CONTINUATION_RECEIPT.json') -Raw | ConvertFrom-Json
$engineering = Get-Content -LiteralPath (Join-Path $epochRoot 'runs\RUN_TRIFLOW_20K_OBSERVER_ENGINEERING_VERIFY_S0\OBSERVER_ENGINEERING_VERIFICATION.json') -Raw | ConvertFrom-Json
if (-not $control.passed -or -not $engineering.passed) { throw 'Actual durable continuation and transparent-observer engineering proof required' }
$snapshotPath = Join-Path $epochRoot 'runs\RUN_TRIFLOW_20K_EPOCH_CONTINUATION_S0_R1\resume_snapshot.pt'
if ((FileSHA $snapshotPath) -ne $control.checkpoint_sha256) { throw 'Durable continuation snapshot changed' }
$observerPath = Join-Path $epochRoot 'scripts\train_epoch_observer.py'
if ((FileSHA $observerPath) -ne $engineering.observer_source_sha256) { throw 'Observer differs from actual engineering proof' }
$oldTaskName = 'Codex_TriFlow_20K_20261007_01a11060'
$oldTask = Get-ScheduledTask -TaskName $oldTaskName -ErrorAction SilentlyContinue
if ($oldTask) {
    if ($oldTask.State -ne 'Ready') { throw 'Original task has not settled; preserve and inspect' }
    Unregister-ScheduledTask -TaskName $oldTaskName -Confirm:$false
}
New-Item -ItemType Directory -Path $epochRun -Force | Out-Null
$epochUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$epochArgs = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File D:\coco_wire\experiments\triflow_20k_training_20261007\scripts\task_epoch_observer.ps1'
$action = New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument $epochArgs -WorkingDirectory $epochRoot
$principal = New-ScheduledTaskPrincipal -UserId $epochUser -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -Hidden -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Days 3)
Register-ScheduledTask -TaskName $epochTask -Action $action -Principal $principal -Settings $settings -Description 'One continuation of fixed eight epochs; evaluate complete5000 images at every actual epoch boundary' | Out-Null
Start-ScheduledTask -TaskName $epochTask
$receipt = @{task_name=$epochTask;requested_at=(Get-Date).ToString('o');command=$epochArgs;resume_snapshot_sha256=$control.checkpoint_sha256;training_images=20000;total_epochs=8;eval_images_each_epoch=5000;old_task_removed=$true;execution_limit_days=3;training_algorithm_changed=$false}
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $epochRun 'SCHEDULER_LAUNCH.json') -Encoding utf8
$receipt | ConvertTo-Json -Compress
