$ErrorActionPreference = 'Stop'
$taskName = 'Codex_Rederive_20261010'
$experimentRoot = 'D:\coco_wire\experiments\coefficient_recoverability_rederive_20261010'
$taskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$taskArgs = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File D:/coco_wire/experiments/coefficient_recoverability_rederive_20261010/task_runner_rederive.ps1'
$taskAction = New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument $taskArgs -WorkingDirectory $experimentRoot
$taskPrincipal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
$taskSettings = New-ScheduledTaskSettingsSet -Hidden -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 6)
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) { throw 'Task already exists; inspect its state before changing it.' }
Register-ScheduledTask -TaskName $taskName -Action $taskAction -Principal $taskPrincipal -Settings $taskSettings -Description 'Single detached execution of the locked recoverability re-derivation; no recurring trigger' | Out-Null
Start-ScheduledTask -TaskName $taskName
$receipt = @{task_name=$taskName; principal=$taskUser; launch_mode='Windows task scheduler, single manual start, hidden window, detached from ssh'; started_at_utc=[DateTime]::UtcNow.ToString('o'); command=$taskArgs; run='RUN_REDERIVE_S0_R1'}
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $experimentRoot 'runs/RUN_REDERIVE_S0_R1/SCHEDULER_LAUNCH.json') -Encoding UTF8
$receipt | ConvertTo-Json -Compress
