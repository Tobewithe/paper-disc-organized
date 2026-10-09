$ErrorActionPreference = 'Stop'
$triFlowRoot = 'D:\coco_wire\experiments\triflow_potential_coefficient_20261006'
$triFlowPipelinePath = Join-Path $triFlowRoot 'runs\RUN_TRIFLOW_PIPELINE_S0'
$triFlowMeta = Get-Content -LiteralPath (Join-Path $triFlowPipelinePath 'run.json') -Raw | ConvertFrom-Json
$triFlowComplete = Get-Content -LiteralPath (Join-Path $triFlowPipelinePath 'PIPELINE_COMPLETE.json') -Raw | ConvertFrom-Json
if ($triFlowMeta.execution_status -ne 'completed' -or $triFlowMeta.exit_code -ne 0 -or -not $triFlowComplete.completed) {
    throw 'Owned TriFlow pipeline has not completed successfully.'
}
$triFlowEvalPath = Join-Path $triFlowRoot ('runs\' + $triFlowComplete.evaluation_run)
$triFlowEvalMeta = Get-Content -LiteralPath (Join-Path $triFlowEvalPath 'run.json') -Raw | ConvertFrom-Json
$triFlowSummary = Get-Content -LiteralPath (Join-Path $triFlowEvalPath 'SUMMARY.json') -Raw | ConvertFrom-Json
if ($triFlowEvalMeta.execution_status -ne 'completed' -or $triFlowEvalMeta.exit_code -ne 0 -or $triFlowSummary.status -ne 'complete') {
    throw 'Owned full evaluation has not completed successfully.'
}
foreach ($triFlowOwnedPid in @([int]$triFlowMeta.pid,[int]$triFlowEvalMeta.pid)) {
    if (Get-Process -Id $triFlowOwnedPid -ErrorAction SilentlyContinue) {
        throw "Owned process $triFlowOwnedPid still present; leave scheduler intact."
    }
}
$triFlowTaskName = 'Codex_TriFlow_20261006_01a11060'
$triFlowTask = Get-ScheduledTask -TaskName $triFlowTaskName -ErrorAction Stop
$triFlowInfo = Get-ScheduledTaskInfo -TaskName $triFlowTaskName -ErrorAction Stop
if ($triFlowTask.State -eq 'Running' -or $triFlowInfo.LastTaskResult -ne 0) {
    throw 'Owned scheduler is not finished with success.'
}
$triFlowReceipt = [ordered]@{task_name=$triFlowTaskName;state_before_cleanup=$triFlowTask.State.ToString();
    last_task_result=$triFlowInfo.LastTaskResult;last_run_time=$triFlowInfo.LastRunTime.ToString('o');
    observed_at=(Get-Date).ToString('o');owned_processes_absent=$true;one_shot_task_removed=$false;
    cleanup_script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()}
$triFlowReceipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $triFlowPipelinePath 'SCHEDULER_COMPLETE.json') -Encoding utf8
Unregister-ScheduledTask -TaskName $triFlowTaskName -Confirm:$false -ErrorAction Stop
$triFlowReceipt.one_shot_task_removed = $true
$triFlowReceipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $triFlowPipelinePath 'SCHEDULER_COMPLETE.json') -Encoding utf8
$triFlowReceipt | ConvertTo-Json -Compress
