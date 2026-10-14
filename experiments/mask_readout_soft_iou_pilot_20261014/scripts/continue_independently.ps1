param([string]$TrainingRunName = 'RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb')
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$pilotRoot = 'D:\coco_wire\baseline_native_20261013'
$pilotStudy = Join-Path $pilotRoot 'experiments\mask_readout_soft_iou_pilot_20261014'
$pilotTraining = Join-Path $pilotStudy ('runs\'+$TrainingRunName)
$pilotPython = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$pilotLog = Join-Path $pilotStudy 'autonomous_execution.log'
$pilotStatusPath = Join-Path $pilotStudy 'AUTONOMOUS_STATUS.json'
$pilotGuard = Join-Path $pilotStudy '.AUTONOMOUS_OWNER.guard'
$pilotOwner = $null
$pilotPowerSet = $false

function Save-PilotStatus([string]$Phase, [object]$Extra = @{}) {
    $pilotStatus = [ordered]@{phase=$Phase; updated_at=(Get-Date -Format o); controller_pid=$PID; host=$env:COMPUTERNAME; training_run_id=$TrainingRunName; executor='Windows Task Scheduler, interactive user; independent of SSH'; details=$Extra}
    $pilotTemporary = $pilotStatusPath+'.tmp'
    [IO.File]::WriteAllText($pilotTemporary, ($pilotStatus | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $pilotTemporary -Destination $pilotStatusPath -Force
}
function Invoke-PilotPython([string[]]$PythonArguments) {
    & $pilotPython @PythonArguments *>> $pilotLog
    if ($LASTEXITCODE -ne 0) { throw ('Python stage failed: '+($PythonArguments -join ' ')+'; exit='+$LASTEXITCODE) }
}

try {
    $pilotOwner=[IO.File]::Open($pilotGuard,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    $pilotOwnerBytes=[Text.Encoding]::UTF8.GetBytes(('pid='+$PID+'; training='+$TrainingRunName))
    $pilotOwner.Write($pilotOwnerBytes,0,$pilotOwnerBytes.Length)
    Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class PilotPower { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
    $pilotPowerResult=[PilotPower]::SetThreadExecutionState([uint32]2147483649)
    if ($pilotPowerResult -eq 0) { throw 'Unable to prevent idle sleep during independent execution.' }
    $pilotPowerSet=$true
    $env:PYTHONPATH = Join-Path $pilotRoot '_site'
    $env:YOLO_OFFLINE='true'
    $env:YOLO_AUTOINSTALL='false'
    $env:YOLO_CONFIG_DIR=Join-Path $pilotRoot '_ultralytics_config'
    $env:OMP_NUM_THREADS='1'
    $env:PYTHONDONTWRITEBYTECODE='1'
    $env:PYTHONIOENCODING='utf-8'
    Set-Location -LiteralPath $pilotRoot
    Save-PilotStatus 'waiting_for_existing_training' @{no_new_training=$true; idle_sleep_temporarily_inhibited=$true}
    $pilotDeadline=(Get-Date).AddSeconds(7800)
    do {
        $pilotTrainState=Get-Content -Raw -LiteralPath (Join-Path $pilotTraining 'run.json') | ConvertFrom-Json
        if ($pilotTrainState.execution_status -eq 'error') { throw 'Existing training failed; retain it and do not restart training.' }
        if ($pilotTrainState.execution_status -eq 'completed') { break }
        if ((Get-Date) -gt $pilotDeadline) { throw 'Existing training did not close within the fixed budget plus bookkeeping margin.' }
        Start-Sleep -Seconds 5
    } while ($true)
    # Wait for __exit__ to finish the artifact inventory and release the scientific stage lock.
    while (Test-Path -LiteralPath (Join-Path $pilotStudy '.S2_SOFT_IOU.lock')) {
        if ((Get-Date) -gt $pilotDeadline) { throw 'Training stage lock did not close.' }
        Start-Sleep -Seconds 2
    }
    if ((Get-Content -Raw -LiteralPath (Join-Path $pilotTraining 'CONFIG_AUDIT.json') | ConvertFrom-Json).status -ne 'passed') { throw 'Training configuration audit failed.' }
    Save-PilotStatus 'independent_training_audit'
    Invoke-PilotPython -PythonArguments @('-B',(Join-Path $pilotStudy 'scripts\collect_evidence.py'),'audit',$pilotTraining)
    if (-not (Test-Path -LiteralPath (Join-Path $pilotTraining 'evidence.zip'))) {
        Invoke-PilotPython -PythonArguments @('-B',(Join-Path $pilotStudy 'scripts\collect_evidence.py'),'package',$pilotTraining)
    }
    $pilotPriorEval=@(Get-ChildItem -LiteralPath (Join-Path $pilotStudy 'runs') -Directory | Where-Object { $_.Name -like 'RUN_SOFT_IOU_EVAL_*' })
    if ($pilotPriorEval.Count -ne 0) { throw 'Existing evaluation found; no duplicate draw or evaluation is allowed.' }
    Save-PilotStatus 'S2_evaluation_starting' @{training_config_audit='passed'; checkpoint=(Join-Path $pilotTraining 'last.pt')}
    Invoke-PilotPython -PythonArguments @('-B',(Join-Path $pilotStudy 'scripts\run_evaluation.py'),'--training-run',$pilotTraining)
    $pilotEval=@(Get-ChildItem -LiteralPath (Join-Path $pilotStudy 'runs') -Directory | Where-Object { $_.Name -like 'RUN_SOFT_IOU_EVAL_*' })
    if ($pilotEval.Count -ne 1) { throw 'Expected exactly one new evaluation Run.' }
    $pilotEvalPath=$pilotEval[0].FullName
    $pilotEvalState=Get-Content -Raw -LiteralPath (Join-Path $pilotEvalPath 'run.json') | ConvertFrom-Json
    if ($pilotEvalState.execution_status -ne 'completed') { throw 'S2 evaluation did not complete.' }
    Save-PilotStatus 'independent_evaluation_audit' @{evaluation_run_id=$pilotEval[0].Name}
    Invoke-PilotPython -PythonArguments @('-B',(Join-Path $pilotStudy 'scripts\collect_evidence.py'),'audit',$pilotEvalPath)
    Invoke-PilotPython -PythonArguments @('-B',(Join-Path $pilotStudy 'scripts\collect_evidence.py'),'package',$pilotEvalPath)
    $pilotDecision=Get-Content -Raw -LiteralPath (Join-Path $pilotEvalPath 'DECISION.json') | ConvertFrom-Json
    $pilotSummary=Get-Content -Raw -LiteralPath (Join-Path $pilotEvalPath 'SUMMARY.json') | ConvertFrom-Json
    $pilotResult=[ordered]@{execution_status='completed'; evaluation_run_id=$pilotEval[0].Name; scientific_decision=$pilotDecision; gates=$pilotSummary.gates; B=$pilotSummary.B; next_stage_automatically_started=$false; evidence_retained_on_laptop=$true; finished_at=(Get-Date -Format o)}
    [IO.File]::WriteAllText((Join-Path $pilotStudy 'AUTONOMOUS_RESULT.json'), ($pilotResult | ConvertTo-Json -Depth 16), [Text.UTF8Encoding]::new($false))
    $pilotStudyMeta=Get-Content -Raw -LiteralPath (Join-Path $pilotStudy 'study.json') | ConvertFrom-Json
    $pilotStudyMeta.status='S2_'+$pilotDecision.status
    $pilotStudyMeta.execution_status='completed'
    $pilotStudyMeta.scientific_status=$pilotDecision.scientific_status
    $pilotStudyMeta.active_run_id=$null
    $pilotStudyMeta.configuration_audit='passed'
    $pilotStudyMeta | Add-Member -NotePropertyName evaluation_run_id -NotePropertyValue $pilotEval[0].Name -Force
    [IO.File]::WriteAllText((Join-Path $pilotStudy 'study.json'), ($pilotStudyMeta | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
    Save-PilotStatus 'completed' @{evaluation_run_id=$pilotEval[0].Name; scientific_status=$pilotDecision.scientific_status; next_action=$pilotDecision.next_action; no_S3_started=$true}
} catch {
    $_ | Out-String | Add-Content -LiteralPath $pilotLog
    Save-PilotStatus 'error' @{error=($_.Exception.Message); no_automatic_retry=$true; outputs_preserved=$true}
    exit 1
} finally {
    if ($pilotPowerSet) { [void][PilotPower]::SetThreadExecutionState([uint32]2147483648) }
    if ($null -ne $pilotOwner) {
        $pilotOwner.Dispose()
        Remove-Item -LiteralPath $pilotGuard
    }
}
