$ErrorActionPreference = 'Stop'
$acdExperimentRoot = 'D:\coco_wire\experiments\acd_proto_tail_unfreeze_20261006'
$acdPipelineRoot = Join-Path $acdExperimentRoot 'runs\RUN_PROTO_TAIL_PIPELINE_S0'
$acdPipelineMeta = Get-Content -LiteralPath (Join-Path $acdPipelineRoot 'run.json') -Raw | ConvertFrom-Json
$acdPipelineComplete = Get-Content -LiteralPath (Join-Path $acdPipelineRoot 'COMPLETE.json') -Raw | ConvertFrom-Json
if ($acdPipelineMeta.execution_status -ne 'completed' -or $acdPipelineMeta.exit_code -ne 0 -or -not $acdPipelineComplete.completed) {
    throw 'The owned pipeline has not completed successfully.'
}
$acdEvalRoot = Join-Path $acdExperimentRoot 'runs\RUN_PROTO_TAIL_PAIRED_EVAL_S0'
$acdEvalMeta = Get-Content -LiteralPath (Join-Path $acdEvalRoot 'run.json') -Raw | ConvertFrom-Json
$acdEvalSummary = Get-Content -LiteralPath (Join-Path $acdEvalRoot 'SUMMARY.json') -Raw | ConvertFrom-Json
if ($acdEvalMeta.execution_status -ne 'completed' -or $acdEvalMeta.exit_code -ne 0 -or $acdEvalSummary.status -ne 'complete') {
    throw 'The owned evaluation has not completed successfully.'
}
$acdOwnedPids = @([int]$acdPipelineMeta.pid)
$acdEvalProcess = Get-Content -LiteralPath (Join-Path $acdEvalRoot 'PROCESS.json') -Raw | ConvertFrom-Json
$acdOwnedPids += [int]$acdEvalProcess.pid
foreach ($acdOwnedPid in $acdOwnedPids) {
    if (Get-Process -Id $acdOwnedPid -ErrorAction SilentlyContinue) {
        throw "Owned process $acdOwnedPid is still present; do not clean up the scheduler yet."
    }
}
$acdTaskName = 'Codex_ACD_ProtoTail_20261006_01a11060'
$acdTask = Get-ScheduledTask -TaskName $acdTaskName -ErrorAction Stop
$acdTaskInfo = Get-ScheduledTaskInfo -TaskName $acdTaskName -ErrorAction Stop
if ($acdTask.State -eq 'Running' -or $acdTaskInfo.LastTaskResult -ne 0) {
    throw "Scheduler is not finished with success: $($acdTask.State), result $($acdTaskInfo.LastTaskResult)"
}
$acdSchedulerReceipt = [ordered]@{
    task_name = $acdTaskName
    state_before_cleanup = $acdTask.State.ToString()
    last_task_result = $acdTaskInfo.LastTaskResult
    last_run_time = $acdTaskInfo.LastRunTime.ToString('o')
    observed_at = (Get-Date).ToString('o')
    pipeline_run = $acdPipelineMeta.run_id
    evaluation_run = $acdEvalMeta.run_id
    owned_processes_absent = $true
    one_shot_task_removed = $false
}
$acdReceiptPath = Join-Path $acdPipelineRoot 'SCHEDULER_COMPLETE.json'
$acdSchedulerReceipt | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $acdReceiptPath -Encoding UTF8
Unregister-ScheduledTask -TaskName $acdTaskName -Confirm:$false -ErrorAction Stop
$acdSchedulerReceipt.one_shot_task_removed = $true
$acdSchedulerReceipt | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $acdReceiptPath -Encoding UTF8
$acdSchedulerReceipt | ConvertTo-Json -Compress
