$ErrorActionPreference = 'Stop'
$taskName = 'Codex_E0_Complete_20261012'
$studyRoot = 'D:\coco_wire\experiments\coefficient_shared_affine_alignment_audit_20261011'
$taskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$taskArgs = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File D:/coco_wire/experiments/coefficient_shared_affine_alignment_audit_20261011/scripts/complete_e0_cross_audit.ps1'
$taskAction = New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument $taskArgs -WorkingDirectory $studyRoot
$taskPrincipal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
$taskSettings = New-ScheduledTaskSettingsSet -Hidden -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 72)
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) { throw 'Task already exists; inspect its state before changing it.' }
Register-ScheduledTask -TaskName $taskName -Action $taskAction -Principal $taskPrincipal -Settings $taskSettings -Description 'Complete the interrupted E0 cross-objective audit: official640 solver (exact queued arguments) then evaluator over both maps. Detached; survives disconnect.' | Out-Null
Start-ScheduledTask -TaskName $taskName
$receipt = @{task_name=$taskName; principal=$taskUser; launch_mode='Windows task scheduler, single manual start, hidden, detached'; started_at_utc=[DateTime]::UtcNow.ToString('o'); runs=@('RUN_CROSS_OFFICIAL640_COMPLETE_20261012','RUN_CROSS_EVAL_COMPLETE_20261012'); historical_runs_touched=$false}
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $studyRoot 'runs/RUN_E0_STATUS_AUDIT_20261012/SCHEDULER_LAUNCH.json') -Encoding UTF8
$receipt | ConvertTo-Json -Compress
