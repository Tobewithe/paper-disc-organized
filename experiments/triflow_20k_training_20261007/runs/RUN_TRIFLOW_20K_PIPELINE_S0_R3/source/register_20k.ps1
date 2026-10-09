$ErrorActionPreference = 'Stop'
$fullRoot = 'D:\coco_wire\experiments\triflow_20k_training_20261007'
$fullTask = 'Codex_TriFlow_20K_20261007_01a11060'
$fullPipeline = Join-Path $fullRoot 'runs\RUN_TRIFLOW_20K_PIPELINE_S0'
if (Get-ScheduledTask -TaskName $fullTask -ErrorAction SilentlyContinue) { throw 'Owned 20k task already exists; do not replace' }
$streamProof = Get-Content -LiteralPath (Join-Path $fullRoot 'runs\RUN_TRIFLOW_20K_STREAM_VERIFY_S0\VERIFY_STREAM.json') -Raw | ConvertFrom-Json
$resumeProof = Get-Content -LiteralPath (Join-Path $fullRoot 'runs\RUN_TRIFLOW_20K_RESUME_VERIFY_S0_R1\RESUME_VERIFICATION.json') -Raw | ConvertFrom-Json
$evaluatorProof = Get-Content -LiteralPath (Join-Path $fullRoot 'runs\RUN_TRIFLOW_20K_EVALUATOR_CONTRACT_S0_R1\CONTRACT_VERIFICATION.json') -Raw | ConvertFrom-Json
if (-not $streamProof.passed -or -not $resumeProof.passed -or -not $evaluatorProof.passed) { throw 'Actual input, optimizer resume and evaluator contracts must pass before queuing formal execution' }
New-Item -ItemType Directory -Path $fullPipeline -Force | Out-Null
$fullUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$fullArgs = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File D:\coco_wire\experiments\triflow_20k_training_20261007\scripts\task_20k.ps1'
$fullAction = New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument $fullArgs -WorkingDirectory $fullRoot
$fullPrincipal = New-ScheduledTaskPrincipal -UserId $fullUser -LogonType Interactive -RunLevel Limited
$fullSettings = New-ScheduledTaskSettingsSet -Hidden -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Days 3)
Register-ScheduledTask -TaskName $fullTask -Action $fullAction -Principal $fullPrincipal -Settings $fullSettings -Description 'Single manual20000-image TriFlow execution; eight training epochs then full5000 evaluation/readout. No recurring trigger.' | Out-Null
Start-ScheduledTask -TaskName $fullTask
$receipt = @{ task_name=$fullTask; principal=$fullUser; requested_at=(Get-Date).ToString('o'); command=$fullArgs; launch_mode='one-shot hidden scheduler, no recurring trigger'; execution_limit_days=3; selected20000_data_required_before_training=$true; original_full_split_training=$false; idle_sleep_prevention='process scoped' }
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $fullPipeline 'SCHEDULER_LAUNCH.json') -Encoding utf8
$receipt | ConvertTo-Json -Compress
