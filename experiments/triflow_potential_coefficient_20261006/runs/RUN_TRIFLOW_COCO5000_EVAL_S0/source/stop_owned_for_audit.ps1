$ErrorActionPreference = 'Stop'
$triAuditRoot = 'D:\coco_wire\experiments\triflow_potential_coefficient_20261006'
$triAuditRun = Join-Path $triAuditRoot 'runs\RUN_TRIFLOW_TRAIN_S0'
$triAuditMeta = Get-Content -LiteralPath (Join-Path $triAuditRun 'run.json') -Raw | ConvertFrom-Json
$triAuditProcess = Get-Content -LiteralPath (Join-Path $triAuditRun 'PROCESS.json') -Raw | ConvertFrom-Json
if ($triAuditMeta.execution_status -ne 'running' -or $triAuditMeta.pid -ne $triAuditProcess.pid) {
    throw 'Expected owned initial training Run is not running with the recorded PID.'
}
$triAuditOwnedPid = [int]$triAuditProcess.pid
$triAuditActual = Get-CimInstance Win32_Process -Filter "ProcessId=$triAuditOwnedPid"
$triAuditExpectedScript = Join-Path $triAuditRun 'source\train_triflow.py'
if (-not $triAuditActual -or -not $triAuditActual.CommandLine.Contains($triAuditExpectedScript)) {
    throw 'PID command line does not match this exact owned source script; stop nothing.'
}
$triAuditReceipt = [ordered]@{run_id=$triAuditMeta.run_id;pid=$triAuditOwnedPid;script=$triAuditExpectedScript;
    actual_command_line=$triAuditActual.CommandLine;requested_at=(Get-Date).ToString('o');
    reason='Strengthen observation of the existing mask-task backward: separate phi output rows from stiffness and require simultaneous phi/attention gradients. Objective, sampling, budget and backward count unchanged. Restart module from seed0 in a distinct Run.';
    cache_run_preserved='RUN_TRIFLOW_FIT_CACHE_S0';stopped=$false;
    stop_script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()}
$triAuditReceipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $triAuditRun 'INTERRUPTION.json') -Encoding utf8
Stop-Process -Id $triAuditOwnedPid -ErrorAction Stop
$triAuditReceipt.stopped=$true
$triAuditReceipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $triAuditRun 'INTERRUPTION.json') -Encoding utf8
$triAuditReceipt | ConvertTo-Json -Compress
