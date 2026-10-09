$ErrorActionPreference = 'Stop'
$triFlowTask = 'Codex_TriFlow_20261006_01a11060'
$triFlowRoot = 'D:\coco_wire\experiments\triflow_potential_coefficient_20261006'
$triFlowPipeline = Join-Path $triFlowRoot 'runs\RUN_TRIFLOW_PIPELINE_S0'
if (Get-ScheduledTask -TaskName $triFlowTask -ErrorAction SilentlyContinue) {
    throw 'This owned TriFlow task already exists; inspect rather than replace.'
}
$triFlowDiagnostic = Get-Content -LiteralPath (Join-Path $triFlowRoot 'runs\RUN_TRIFLOW_NUMERICAL_VERIFY_S0_R1\TRIFLOW_VERIFICATION.json') -Raw | ConvertFrom-Json
$triFlowSmoke = Get-Content -LiteralPath (Join-Path $triFlowRoot 'runs\RUN_TRIFLOW_SMOKE_TRAIN_S0\TRAINING_AUDIT.json') -Raw | ConvertFrom-Json
$triFlowContract = Get-Content -LiteralPath (Join-Path $triFlowRoot 'runs\RUN_TRIFLOW_EVAL_CONTRACT_S0\EVALUATOR_VERIFICATION.json') -Raw | ConvertFrom-Json
if (-not $triFlowDiagnostic.passed -or -not $triFlowSmoke.passed -or -not $triFlowContract.passed) {
    throw 'Mathematical, actual smoke and sample decoder contracts must pass.'
}
New-Item -ItemType Directory -Path $triFlowPipeline -Force | Out-Null
$triFlowUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$triFlowArgs = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File D:\coco_wire\experiments\triflow_potential_coefficient_20261006\scripts\task_runner.ps1'
$triFlowAction = New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument $triFlowArgs -WorkingDirectory $triFlowRoot
$triFlowPrincipal = New-ScheduledTaskPrincipal -UserId $triFlowUser -LogonType Interactive -RunLevel Limited
$triFlowSettings = New-ScheduledTaskSettingsSet -Hidden -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 12)
Register-ScheduledTask -TaskName $triFlowTask -Action $triFlowAction -Principal $triFlowPrincipal -Settings $triFlowSettings -Description 'One manual execution of TriFlow fixed eight-epoch feasibility pipeline, no recurring trigger' | Out-Null
Start-ScheduledTask -TaskName $triFlowTask
$triFlowReceipt = @{task_name=$triFlowTask;principal=$triFlowUser;requested_at=(Get-Date).ToString('o');command=$triFlowArgs;launch_mode='One-shot manual start, hidden'}
$triFlowReceipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $triFlowPipeline 'SCHEDULER_LAUNCH.json') -Encoding utf8
$triFlowReceipt | ConvertTo-Json -Compress
