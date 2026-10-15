$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$s3Root='D:\coco_wire\baseline_native_20261013'
$s3Study=Join-Path $s3Root 'experiments\mask_gate_s3r_20261015'
$s3Python='C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$s3Log=Join-Path $s3Study 'autonomous_execution.log'
$s3Guard=Join-Path $s3Study '.AUTONOMOUS_OWNER.guard'
$s3Owner=$null
$s3Power=$false
function Save-State([string]$Phase,[object]$Details=@{}) {
    $state=[ordered]@{phase=$Phase;updated_at=(Get-Date -Format o);controller_pid=$PID;host=$env:COMPUTERNAME;details=$Details;new_training_steps=0}
    [IO.File]::WriteAllText((Join-Path $s3Study 'AUTONOMOUS_STATUS.json'),($state|ConvertTo-Json -Depth 10),[Text.UTF8Encoding]::new($false))
}
function Run-Python([string[]]$Arguments) {
    & $s3Python @Arguments *>> $s3Log
    if ($LASTEXITCODE -ne 0) { throw ('Exit '+$LASTEXITCODE+': '+($Arguments -join ' ')) }
}
function Find-Completed([string]$Prefix) {
    $items=@(Get-ChildItem -LiteralPath (Join-Path $s3Study 'runs') -Directory | Where-Object {$_.Name -like ($Prefix+'*')})
    if ($items.Count -ne 1) { throw ('Expected one '+$Prefix+' Run') }
    $meta=Get-Content -Raw -LiteralPath (Join-Path $items[0].FullName 'run.json')|ConvertFrom-Json
    if ($meta.execution_status -ne 'completed' -or $meta.exit_code -ne 0) { throw 'Run incomplete' }
    return $items[0]
}
try {
    $s3Owner=[IO.File]::Open($s3Guard,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class S3Power { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }'
    if ([S3Power]::SetThreadExecutionState([uint32]2147483649) -eq 0) { throw 'Cannot inhibit idle sleep' }
    $s3Power=$true
    $env:PYTHONPATH=Join-Path $s3Root '_site'
    $env:YOLO_OFFLINE='true';$env:YOLO_AUTOINSTALL='false'
    $env:YOLO_CONFIG_DIR=Join-Path $s3Root '_ultralytics_config'
    $env:OMP_NUM_THREADS='1';$env:PYTHONDONTWRITEBYTECODE='1';$env:PYTHONIOENCODING='utf-8'
    if ($env:MKL_NUM_THREADS) { throw 'Unexpected MKL_NUM_THREADS override' }
    Set-Location -LiteralPath $s3Root
    if ((Test-Path -LiteralPath (Join-Path $s3Study 'runs')) -and @(Get-ChildItem -LiteralPath (Join-Path $s3Study 'runs') -Directory).Count) { throw 'Prior Run exists; no automatic duplicate/retry' }
    Save-State 'smoke'
    Run-Python @('-B',(Join-Path $s3Study 'scripts\run_s3.py'),'--scope','smoke')
    $smoke=Find-Completed 'RUN_S3_SMOKE_'
    Save-State 'full_evaluation' @{smoke_run=$smoke.Name}
    Run-Python @('-B',(Join-Path $s3Study 'scripts\run_s3.py'),'--scope','full','--smoke-run',$smoke.FullName)
    $full=Find-Completed 'RUN_S3_FULL_'
    Save-State 'independent_verification' @{full_run=$full.Name}
    Run-Python @('-B',(Join-Path $s3Study 'scripts\verify_s3.py'),'--run',$full.FullName,'--smoke-run',$smoke.FullName,'--out',(Join-Path $s3Study 'INDEPENDENT_LAPTOP.json'),'--laptop-input-audit')
    Save-State 'completed' @{smoke_run=$smoke.Name;full_run=$full.Name;no_S4_started=$true}
} catch {
    $_|Out-String|Add-Content -LiteralPath $s3Log
    Save-State 'error' @{error=$_.Exception.Message;outputs_preserved=$true;automatic_retry=$false}
    exit 1
} finally {
    if ($s3Power) { [void][S3Power]::SetThreadExecutionState([uint32]2147483648) }
    if ($null -ne $s3Owner) { $s3Owner.Dispose();Remove-Item -LiteralPath $s3Guard }
}
