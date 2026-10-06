$ErrorActionPreference = 'Stop'
$taskName = 'Codex_ACD_20261006_01a11060'
$experimentRoot = 'D:\coco_wire\experiments\acd_native_coefficient_20261006'
$taskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$taskArgs = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File D:/coco_wire/experiments/acd_native_coefficient_20261006/scripts/task_runner.ps1'
$taskAction = New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument $taskArgs -WorkingDirectory $experimentRoot
$taskPrincipal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
$taskSettings = New-ScheduledTaskSettingsSet -Hidden -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 12)
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) { throw 'This experiment task already exists; inspect its state before changing it.' }
Register-ScheduledTask -TaskName $taskName -Action $taskAction -Principal $taskPrincipal -Settings $taskSettings -Description 'Single execution of the locked ACD coefficient experiment, no recurring trigger' | Out-Null
Start-ScheduledTask -TaskName $taskName
$receipt = @{task_name=$taskName; principal=$taskUser; launch_mode='Windows task scheduler, single manual start, hidden window'; requested_at_utc=[DateTime]::UtcNow.ToString('o'); command=$taskArgs}
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $experimentRoot 'runs/RUN_ACD_PIPELINE_S0_RETRY1/SCHEDULER_LAUNCH.json') -Encoding UTF8
$receipt | ConvertTo-Json -Compress
