$ErrorActionPreference = 'Stop'
$triFlowInterruptedRoot = 'D:\coco_wire\experiments\triflow_potential_coefficient_20261006'
$triFlowInterruptedTrain = Join-Path $triFlowInterruptedRoot 'runs\RUN_TRIFLOW_TRAIN_S0'
$triFlowInterruptedPipe = Join-Path $triFlowInterruptedRoot 'runs\RUN_TRIFLOW_PIPELINE_S0'
$triFlowInterruption = Get-Content -LiteralPath (Join-Path $triFlowInterruptedTrain 'INTERRUPTION.json') -Raw | ConvertFrom-Json
if (-not $triFlowInterruption.stopped) { throw 'No completed owned audit interruption receipt.' }
foreach ($triFlowInterruptedPath in @($triFlowInterruptedTrain,$triFlowInterruptedPipe)) {
    $triFlowInterruptedMeta = Get-Content -LiteralPath (Join-Path $triFlowInterruptedPath 'run.json') -Raw | ConvertFrom-Json
    if ($triFlowInterruptedMeta.execution_status -ne 'failed' -or $triFlowInterruptedMeta.exit_code -eq 0) { throw 'Expected interrupted failed Run not settled.' }
    if (Get-Process -Id ([int]$triFlowInterruptedMeta.pid) -ErrorAction SilentlyContinue) { throw 'Recorded owned process still present.' }
}
$triFlowInterruptedTask = 'Codex_TriFlow_20261006_01a11060'
$triFlowInterruptedJob = Get-ScheduledTask -TaskName $triFlowInterruptedTask -ErrorAction Stop
$triFlowInterruptedInfo = Get-ScheduledTaskInfo -TaskName $triFlowInterruptedTask -ErrorAction Stop
if ($triFlowInterruptedJob.State -eq 'Running' -or $triFlowInterruptedInfo.LastTaskResult -eq 0) { throw 'Expected failed owned scheduler not settled.' }
$triFlowInterruptedReceipt = [ordered]@{task_name=$triFlowInterruptedTask;state_before_cleanup=$triFlowInterruptedJob.State.ToString();last_task_result=$triFlowInterruptedInfo.LastTaskResult;observed_at=(Get-Date).ToString('o');reason=$triFlowInterruption.reason;one_shot_task_removed=$false;owned_processes_absent=$true}
$triFlowInterruptedReceipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $triFlowInterruptedPipe 'SCHEDULER_INTERRUPTED.json') -Encoding utf8
Unregister-ScheduledTask -TaskName $triFlowInterruptedTask -Confirm:$false -ErrorAction Stop
$triFlowInterruptedReceipt.one_shot_task_removed=$true
$triFlowInterruptedReceipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $triFlowInterruptedPipe 'SCHEDULER_INTERRUPTED.json') -Encoding utf8
$triFlowInterruptedReceipt | ConvertTo-Json -Compress
