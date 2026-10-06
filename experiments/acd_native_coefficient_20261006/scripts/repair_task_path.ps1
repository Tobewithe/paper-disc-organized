$ErrorActionPreference = 'Stop'
$taskName = 'Codex_ACD_20261006_01a11060'
$experimentRoot = 'D:\coco_wire\experiments\acd_native_coefficient_20261006'
$task = Get-ScheduledTask -TaskName $taskName
if ($task.State -eq 'Running') { throw 'Already running; do not change the task' }
$taskInfo = Get-ScheduledTaskInfo -TaskName $taskName
@{task_name=$taskName; last_task_result=$taskInfo.LastTaskResult; last_run=$taskInfo.LastRunTime.ToString('o'); observation='Scheduler could not find bare powershell.exe; no queue execution occurred'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $experimentRoot 'runs/RUN_ACD_PIPELINE_S0_RETRY1/SCHEDULER_PATH_FAILURE.json') -Encoding UTF8
$taskArgs = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File D:\coco_wire\experiments\acd_native_coefficient_20261006\scripts\task_runner.ps1'
$taskAction = New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument $taskArgs -WorkingDirectory $experimentRoot
Set-ScheduledTask -TaskName $taskName -Action $taskAction | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
