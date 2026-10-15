$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$taskRoot='D:\coco_wire\baseline_native_20261013'
$taskStudy=Join-Path $taskRoot 'experiments\mask_readout_spherical_adapter_20261015'
$taskPython='C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$taskLog=Join-Path $taskStudy 'autonomous_execution.log'
$taskGuard=Join-Path $taskStudy '.AUTONOMOUS_OWNER.guard'
$taskOwner=$null
$taskPowerSet=$false
function Save-State([string]$Phase,[object]$Extra=@{}) {
    $s=[ordered]@{phase=$Phase;updated_at=(Get-Date -Format o);controller_pid=$PID;host=$env:COMPUTERNAME;executor='one-shot Windows Task Scheduler; independent of SSH';details=$Extra}
    [IO.File]::WriteAllText((Join-Path $taskStudy 'AUTONOMOUS_STATUS.json'),($s|ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
}
function Run-Python([string[]]$Arguments) {
    & $taskPython @Arguments *>> $taskLog
    if ($LASTEXITCODE -ne 0) { throw ('Stage failed with exit '+$LASTEXITCODE+': '+($Arguments -join ' ')) }
}
function Completed-Run([string]$Prefix) {
    $items=@(Get-ChildItem -LiteralPath (Join-Path $taskStudy 'runs') -Directory | Where-Object { $_.Name -like ($Prefix+'*') })
    if ($items.Count -ne 1) { throw ('Expected exactly one '+$Prefix+' Run') }
    $s=Get-Content -Raw -LiteralPath (Join-Path $items[0].FullName 'run.json')|ConvertFrom-Json
    if ($s.execution_status -ne 'completed' -or $s.exit_code -ne 0) { throw 'Run did not complete' }
    return $items[0]
}
try {
    $taskOwner=[IO.File]::Open($taskGuard,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class E2Power { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
    if ([E2Power]::SetThreadExecutionState([uint32]2147483649) -eq 0) { throw 'Unable to inhibit idle sleep' }
    $taskPowerSet=$true
    $env:PYTHONPATH=Join-Path $taskRoot '_site';$env:YOLO_OFFLINE='true';$env:YOLO_AUTOINSTALL='false'
    $env:YOLO_CONFIG_DIR=Join-Path $taskRoot '_ultralytics_config';$env:OMP_NUM_THREADS='1'
    $env:PYTHONDONTWRITEBYTECODE='1';$env:PYTHONIOENCODING='utf-8'
    Set-Location -LiteralPath $taskRoot
    if (Test-Path -LiteralPath (Join-Path $taskStudy 'runs')) {
        if (@(Get-ChildItem -LiteralPath (Join-Path $taskStudy 'runs') -Directory).Count -gt 0) { throw 'Prior Run exists; no automatic retry or duplicate training' }
    }
    Save-State 'preflight'
    Run-Python @('-B',(Join-Path $taskStudy 'scripts\test_spherical.py'))
    Run-Python @('-B',(Join-Path $taskStudy 'scripts\run_training.py'),'--preflight')
    Save-State 'smoke'
    Run-Python @('-B',(Join-Path $taskStudy 'scripts\run_training.py'),'--scope','smoke')
    $smoke=Completed-Run 'RUN_SPHERICAL_SMOKE_'
    Save-State 'training' @{smoke_run=$smoke.Name}
    Run-Python @('-B',(Join-Path $taskStudy 'scripts\run_training.py'),'--scope','full','--smoke-run',$smoke.Name)
    $train=Completed-Run 'RUN_SPHERICAL_TRAIN_'
    Save-State 'training_audit' @{training_run=$train.Name}
    Run-Python @('-B',(Join-Path $taskStudy 'scripts\audit_training.py'),$train.FullName)
    Save-State 'evaluation' @{training_run=$train.Name}
    Run-Python @('-B',(Join-Path $taskStudy 'scripts\run_evaluation.py'),'--training-run',$train.Name)
    $eval=Completed-Run 'RUN_SPHERICAL_EVAL_'
    Save-State 'evaluation_audit' @{training_run=$train.Name;evaluation_run=$eval.Name}
    Run-Python @('-B',(Join-Path $taskStudy 'scripts\verify_evaluation.py'),'--run',$eval.FullName,'--out',(Join-Path $taskStudy 'INDEPENDENT_LAPTOP_FULL.json'),'--cache-audit')
    Save-State 'completed' @{training_run=$train.Name;evaluation_run=$eval.Name;no_S3_or_additional_training=$true}
} catch {
    $_|Out-String|Add-Content -LiteralPath $taskLog
    Save-State 'error' @{error=$_.Exception.Message;outputs_preserved=$true;automatic_retry=$false}
    exit 1
} finally {
    if ($taskPowerSet) { [void][E2Power]::SetThreadExecutionState([uint32]2147483648) }
    if ($null -ne $taskOwner) { $taskOwner.Dispose();Remove-Item -LiteralPath $taskGuard }
}
