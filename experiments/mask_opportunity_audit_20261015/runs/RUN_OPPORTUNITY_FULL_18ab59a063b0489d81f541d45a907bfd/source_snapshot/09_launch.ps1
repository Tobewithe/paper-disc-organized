$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$auditRoot='D:\coco_wire\baseline_native_20261013'
$auditStudy=Join-Path $auditRoot 'experiments\mask_opportunity_audit_20261015'
$auditPython='C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$auditLog=Join-Path $auditStudy 'autonomous_execution.log'
$auditPower=$false
function Save-State([string]$Phase,[object]$Details=@{}) {
    $state=[ordered]@{phase=$Phase;updated_at=(Get-Date -Format o);controller_pid=$PID;host=$env:COMPUTERNAME;details=$Details;new_training_steps=0}
    [IO.File]::WriteAllText((Join-Path $auditStudy 'AUTONOMOUS_STATUS.json'),($state|ConvertTo-Json -Depth 10),[Text.UTF8Encoding]::new($false))
}
function Run-Python([string[]]$Arguments) {
    & $auditPython @Arguments *>> $auditLog
    if ($LASTEXITCODE -ne 0) { throw ('Exit '+$LASTEXITCODE+': '+($Arguments -join ' ')) }
}
function Completed-Run([string]$Scope) {
    $meta=Get-Content -Raw -LiteralPath (Join-Path $auditStudy 'study.json')|ConvertFrom-Json
    $name=$meta.($Scope+'_run_id')
    $path=Join-Path $auditStudy ('runs\'+$name)
    $run=Get-Content -Raw -LiteralPath (Join-Path $path 'run.json')|ConvertFrom-Json
    if ($run.execution_status -ne 'completed' -or $run.exit_code -ne 0) { throw 'Run incomplete' }
    return $path
}
try {
    if (Test-Path -LiteralPath $auditLog) { throw 'Existing execution log; no automatic duplicate/retry' }
    Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class OpportunityPower { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
    if ([OpportunityPower]::SetThreadExecutionState([uint32]2147483649) -eq 0) { throw 'Cannot inhibit idle sleep' }
    $auditPower=$true
    $env:PYTHONPATH=Join-Path $auditRoot '_site'
    $env:YOLO_OFFLINE='true'
    $env:YOLO_AUTOINSTALL='false'
    $env:YOLO_CONFIG_DIR=Join-Path $auditRoot '_ultralytics_config'
    $env:OMP_NUM_THREADS='1'
    $env:PYTHONDONTWRITEBYTECODE='1'
    $env:PYTHONIOENCODING='utf-8'
    if ($env:MKL_NUM_THREADS) { throw 'Unexpected MKL_NUM_THREADS override' }
    Set-Location -LiteralPath $auditRoot
    if (Test-Path -LiteralPath (Join-Path $auditStudy 'runs')) { throw 'Prior Run directory exists' }
    Save-State 'smoke'
    Run-Python @('-B',(Join-Path $auditStudy 'scripts\run_audit.py'),'--scope','smoke')
    $smoke=Completed-Run 'smoke'
    Save-State 'full_audit' @{smoke_run=(Split-Path -Leaf $smoke)}
    Run-Python @('-B',(Join-Path $auditStudy 'scripts\run_audit.py'),'--scope','full','--smoke-run',$smoke)
    $full=Completed-Run 'full'
    Save-State 'independent_verification' @{full_run=(Split-Path -Leaf $full)}
    Run-Python @('-B',(Join-Path $auditStudy 'scripts\verify.py'),'--run',$full,'--smoke-run',$smoke,'--out',(Join-Path $auditStudy 'INDEPENDENT_LAPTOP.json'),'--laptop-input-audit')
    Save-State 'completed' @{smoke_run=(Split-Path -Leaf $smoke);full_run=(Split-Path -Leaf $full)}
} catch {
    $_|Out-String|Add-Content -LiteralPath $auditLog
    Save-State 'error' @{error=$_.Exception.Message;outputs_preserved=$true;automatic_retry=$false}
    exit 1
} finally {
    if ($auditPower) { [void][OpportunityPower]::SetThreadExecutionState([uint32]2147483648) }
}
