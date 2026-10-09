$ErrorActionPreference = 'Stop'
$fullRoot = 'D:\coco_wire\experiments\triflow_fullscale_training_20261007'
$fullRun = Join-Path $fullRoot 'runs\RUN_TRIFLOW_FULL_PIPELINE_S0'
$progress = Get-Content -LiteralPath (Join-Path $fullRun 'PIPELINE_PROGRESS.json') -Raw | ConvertFrom-Json
if ($progress.scientific_training_status -ne 'not_started') { throw 'Refuse waiting-queue cleanup because formal training may have started; inspect exact state' }
$processRecord = Get-Content -LiteralPath (Join-Path $fullRun 'PROCESS.json') -Raw | ConvertFrom-Json
$ownedSource = Join-Path $fullRun 'source\run_fullscale_pipeline.py'
$ownedProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($processRecord.pid)"
if ($ownedProcess) {
    if ($ownedProcess.Name -ne 'python.exe' -or $ownedProcess.CommandLine -notlike "*$ownedSource*") { throw 'Exact owned process identity did not match' }
    @{ reason='user_changed_conclusion_training_scale_to_20000'; requested_at=(Get-Date).ToString('o'); scientific_training_started=$false; preserved_pipeline_pid=$processRecord.pid; next_study='triflow_20k_training_20261007' } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $fullRun 'SCOPE_CHANGE.json') -Encoding utf8
    Stop-Process -Id $processRecord.pid -Force
}
@{ owned_pipeline_child_stopped=$true; reason='user_requested_20000_not_full_original_split'; historical_files_preserved=$true; observed_at=(Get-Date).ToString('o') } | ConvertTo-Json -Compress
