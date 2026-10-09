$ErrorActionPreference = 'Stop'
$acdProtoTaskName = 'Codex_ACD_ProtoTail_20261006_01a11060'
$acdProtoExperimentRoot = 'D:\coco_wire\experiments\acd_proto_tail_unfreeze_20261006'
$acdProtoPipelineRoot = Join-Path $acdProtoExperimentRoot 'runs\RUN_PROTO_TAIL_PIPELINE_S0'
if (Get-ScheduledTask -TaskName $acdProtoTaskName -ErrorAction SilentlyContinue) {
    throw 'This owned task already exists; inspect before retrying.'
}
$acdProtoTrainingCheck = Get-Content -LiteralPath (Join-Path $acdProtoExperimentRoot 'runs\RUN_PROTO_TAIL_NUMERICAL_VERIFY_S0\TRAINING_VERIFICATION.json') -Raw | ConvertFrom-Json
$acdProtoLoaderCheck = Get-Content -LiteralPath (Join-Path $acdProtoExperimentRoot 'runs\RUN_PROTO_TAIL_LOADER_VERIFY_S0\LOADER_VERIFICATION.json') -Raw | ConvertFrom-Json
if ($acdProtoTrainingCheck.status -ne 'PASS' -or $acdProtoLoaderCheck.status -ne 'PASS') {
    throw 'Numerical and loader contracts must pass before training.'
}
$acdProtoPipelineSource = Join-Path $acdProtoPipelineRoot 'source'
New-Item -ItemType Directory -Path $acdProtoPipelineSource -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $acdProtoExperimentRoot 'scripts\run_pipeline.py') -Destination (Join-Path $acdProtoPipelineSource 'run_pipeline.py')
Copy-Item -LiteralPath (Join-Path $acdProtoExperimentRoot 'PROTOCOL.md') -Destination (Join-Path $acdProtoPipelineSource 'PROTOCOL.md')
$acdProtoSourceReceipt = @{
    executed_snapshot = (Join-Path $acdProtoPipelineSource 'run_pipeline.py')
    script_sha256 = (Get-FileHash -LiteralPath (Join-Path $acdProtoPipelineSource 'run_pipeline.py')).Hash.ToLowerInvariant()
    protocol_sha256 = (Get-FileHash -LiteralPath (Join-Path $acdProtoPipelineSource 'PROTOCOL.md')).Hash.ToLowerInvariant()
    captured_at = (Get-Date).ToString('o')
}
$acdProtoSourceReceipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $acdProtoPipelineRoot 'SOURCE.json') -Encoding UTF8
$acdProtoUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$acdProtoTaskArgs = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File D:\coco_wire\experiments\acd_proto_tail_unfreeze_20261006\scripts\task_runner.ps1'
$acdProtoAction = New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument $acdProtoTaskArgs -WorkingDirectory $acdProtoExperimentRoot
$acdProtoPrincipal = New-ScheduledTaskPrincipal -UserId $acdProtoUser -LogonType Interactive -RunLevel Limited
$acdProtoSettings = New-ScheduledTaskSettingsSet -Hidden -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 12)
Register-ScheduledTask -TaskName $acdProtoTaskName -Action $acdProtoAction -Principal $acdProtoPrincipal -Settings $acdProtoSettings -Description 'Single execution of paired ACD prototype-tail unfreezing screen; no recurring trigger' | Out-Null
Start-ScheduledTask -TaskName $acdProtoTaskName
$acdProtoReceipt = @{task_name=$acdProtoTaskName; principal=$acdProtoUser; requested_at=(Get-Date).ToString('o');
    launch_mode='One-shot Windows scheduler manual start, hidden'; command=$acdProtoTaskArgs}
$acdProtoReceipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $acdProtoPipelineRoot 'SCHEDULER_LAUNCH.json') -Encoding UTF8
$acdProtoReceipt | ConvertTo-Json -Compress
