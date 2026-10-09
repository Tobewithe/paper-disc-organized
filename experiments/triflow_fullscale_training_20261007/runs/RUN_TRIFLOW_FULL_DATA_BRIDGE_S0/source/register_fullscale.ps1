$ErrorActionPreference = 'Stop'
$fullRoot = 'D:\coco_wire\experiments\triflow_fullscale_training_20261007'
$fullTask = 'Codex_TriFlow_FullCOCO_20261007_01a11060'
$fullPipeline = Join-Path $fullRoot 'runs\RUN_TRIFLOW_FULL_PIPELINE_S0'
if (Get-ScheduledTask -TaskName $fullTask -ErrorAction SilentlyContinue) { throw 'Owned fullscale task already exists; do not replace' }
$streamProof = Get-Content -LiteralPath (Join-Path $fullRoot 'runs\RUN_TRIFLOW_FULL_STREAM_VERIFY_S0\VERIFY_STREAM.json') -Raw | ConvertFrom-Json
$resumeProof = Get-Content -LiteralPath (Join-Path $fullRoot 'runs\RUN_TRIFLOW_FULL_RESUME_VERIFY_S0_R1_RECHECK\RESUME_VERIFICATION.json') -Raw | ConvertFrom-Json
if (-not $streamProof.passed -or -not $resumeProof.passed) { throw 'Actual input and optimizer resume contracts must pass before queuing formal execution' }
New-Item -ItemType Directory -Path $fullPipeline -Force | Out-Null
$fullUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$fullArgs = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File D:\coco_wire\experiments\triflow_fullscale_training_20261007\scripts\task_fullscale.ps1'
$fullAction = New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument $fullArgs -WorkingDirectory $fullRoot
$fullPrincipal = New-ScheduledTaskPrincipal -UserId $fullUser -LogonType Interactive -RunLevel Limited
$fullSettings = New-ScheduledTaskSettingsSet -Hidden -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Days 14)
Register-ScheduledTask -TaskName $fullTask -Action $fullAction -Principal $fullPrincipal -Settings $fullSettings -Description 'Single manual full COCO TriFlow execution; waits for complete data, then fixed training/evaluation/readout. No recurring trigger.' | Out-Null
Start-ScheduledTask -TaskName $fullTask
$receipt = @{ task_name=$fullTask; principal=$fullUser; requested_at=(Get-Date).ToString('o'); command=$fullArgs; launch_mode='one-shot hidden scheduler, no recurring trigger'; execution_limit_days=14; full_data_required_before_training=$true; idle_sleep_prevention='process scoped' }
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $fullPipeline 'SCHEDULER_LAUNCH.json') -Encoding utf8
$receipt | ConvertTo-Json -Compress
