$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$taskRoot='D:\coco_wire\baseline_native_20261013'
$taskStudy=Join-Path $taskRoot 'experiments\mask_coefficient_error_transfer_20261017'
$taskPython='C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$taskName='Codex-Error-Transfer-20261017'
$taskStatePath=Join-Path $taskStudy 'AUTONOMOUS_STATUS_V2.json'
$taskClaim=Join-Path $taskStudy '.AUTONOMOUS_OWNER.guard'
$taskUtf8=[Text.UTF8Encoding]::new($false)
$taskOwned=$false
$taskState=@{status='initializing';host=$env:COMPUTERNAME;controller_pid=$PID;started_at=[DateTime]::UtcNow.ToString('o');phases=@();new_training_steps=0}
Set-Location -LiteralPath $taskRoot
$env:PYTHONPATH=Join-Path $taskRoot '_site'
$env:YOLO_OFFLINE='true'
$env:YOLO_AUTOINSTALL='false'
$env:YOLO_CONFIG_DIR=Join-Path $taskRoot '_ultralytics_config'
$env:OMP_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
Remove-Item Env:MKL_NUM_THREADS -ErrorAction SilentlyContinue
function Save-State {
    $taskState.updated_at=[DateTime]::UtcNow.ToString('o')
    [IO.File]::WriteAllText($taskStatePath,($taskState|ConvertTo-Json -Depth 15),$taskUtf8)
}
function Child([string]$Label,[string[]]$Arguments,[int]$Budget) {
    $out=Join-Path $taskStudy ($Label+'_V2.stdout.log');$err=Join-Path $taskStudy ($Label+'_V2.stderr.log')
    if ((Test-Path -LiteralPath $out) -or (Test-Path -LiteralPath $err)) {throw 'Existing stage logs; do not repeat'}
    $child=Start-Process -FilePath $taskPython -ArgumentList $Arguments -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    $null=$child.Handle
    $taskState.status='running';$taskState.current_phase=$Label;$taskState.child_pid=$child.Id;Save-State
    if (-not $child.WaitForExit($Budget*1000)) {Stop-Process -Id $child.Id -Force;throw ('Frozen stage wall budget exceeded: '+$Label)}
    $child.WaitForExit();$child.Refresh();if($null -eq $child.ExitCode){throw ('Missing process exit code: '+$Label)};$taskState.phases+=@{label=$Label;pid=$child.Id;exit_code=$child.ExitCode;arguments=$Arguments;ended_at=[DateTime]::UtcNow.ToString('o')};Save-State
    if ($child.ExitCode -ne 0) {throw ('Stage failed: '+$Label)}
}
function One-Run([string]$Stage) {
    $found=@(Get-ChildItem -LiteralPath (Join-Path $taskStudy 'runs') -Directory -Filter ('RUN_CET_'+$Stage+'_*'))
    if ($found.Count -ne 1) {throw ('Expected one Run: '+$Stage)}
    return $found[0].FullName
}
try {
    if (Test-Path -LiteralPath $taskStatePath) {throw 'Existing task state; inspect rather than restart'}
    $handle=[IO.File]::Open($taskClaim,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    $bytes=$taskUtf8.GetBytes(($taskState|ConvertTo-Json -Depth 10));$handle.Write($bytes,0,$bytes.Length);$handle.Close();$taskOwned=$true
    $freeze=Get-Content -LiteralPath (Join-Path $taskStudy 'EXECUTION_FREEZE.json') -Raw|ConvertFrom-Json
    foreach($item in $freeze.files.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath (Join-Path $taskRoot $item.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.Value) {throw ('Source freeze differs: '+$item.Name)}
    }
    $recovery=Get-Content -LiteralPath (Join-Path $taskStudy 'EXECUTION_LAUNCH_V2.json') -Raw|ConvertFrom-Json
    if((Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $recovery.launcher_sha256){throw 'Recovery launcher hash mismatch'}
    $taskState.launcher_sha256=$recovery.launcher_sha256
    Save-State
    Child 'TESTS' @('-B',(Join-Path $taskStudy 'scripts\test_math.py')) 120
    Child 'FIT' @('-B',(Join-Path $taskStudy 'scripts\run_audit.py'),'fit') 1800
    $fit=One-Run 'FIT';$taskState.fit_run=Split-Path $fit -Leaf;Save-State
    Child 'FIT_VERIFY' @('-B',(Join-Path $taskStudy 'scripts\verify.py'),$fit) 1800
    Child 'SMOKE' @('-B',(Join-Path $taskStudy 'scripts\run_audit.py'),'smoke','--fit-run',$fit) 600
    $smoke=One-Run 'SMOKE';$taskState.smoke_run=Split-Path $smoke -Leaf;Save-State
    Child 'SMOKE_VERIFY' @('-B',(Join-Path $taskStudy 'scripts\verify.py'),$smoke) 600
    Child 'FULL' @('-B',(Join-Path $taskStudy 'scripts\run_audit.py'),'full','--fit-run',$fit) 3600
    $full=One-Run 'FULL';$taskState.full_run=Split-Path $full -Leaf;Save-State
    Child 'FULL_VERIFY' @('-B',(Join-Path $taskStudy 'scripts\verify.py'),$full) 3600
    $taskState.status='completed_and_independently_verified';$taskState.current_phase='closed_no_training';$taskState.child_pid=$null;Save-State
} catch {
    if ($taskOwned) {$taskState.status='failed';$taskState.error=$_.ToString();Save-State};throw
} finally {
    if ($taskOwned) {
        Remove-Item -LiteralPath $taskClaim
        $taskState.ended_at=[DateTime]::UtcNow.ToString('o')
        $unused=Disable-ScheduledTask -TaskName $taskName
        $taskState.task_disabled=((Get-ScheduledTask -TaskName $taskName).State -eq 'Disabled');Save-State
    }
}
