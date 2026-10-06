$ErrorActionPreference = 'Stop'
$taskName = 'Codex_ACD_20261006_01a11060'
$experimentRoot = 'D:\coco_wire\experiments\acd_native_coefficient_20261006'
$task = Get-ScheduledTask -TaskName $taskName
if ($task.State -eq 'Running') { throw 'An existing pipeline is running; do not change it' }
$taskArgs = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File D:\coco_wire\experiments\acd_native_coefficient_20261006\scripts\task_runner_v2.ps1'
$taskAction = New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument $taskArgs -WorkingDirectory $experimentRoot
Set-ScheduledTask -TaskName $taskName -Action $taskAction | Out-Null
Start-ScheduledTask -TaskName $taskName
$receipt = @{task_name=$taskName; launch_mode='Single hidden manual start after numerical-audit correction'; requested_at_utc=[DateTime]::UtcNow.ToString('o'); command=$taskArgs}
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $experimentRoot 'runs/RUN_ACD_PIPELINE_S0_RETRY2/SCHEDULER_LAUNCH.json') -Encoding UTF8
$receipt | ConvertTo-Json -Compress
