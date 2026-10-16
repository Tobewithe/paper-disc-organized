$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$taskRoot = 'D:\coco_wire\baseline_native_20261013'
$taskStudy = Join-Path $taskRoot 'experiments\mask_direction_learnability_20261016'
$taskPython = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$taskName = 'Codex-CLD-Formal-20261016'
$taskClaim = Join-Path $taskStudy 'FORMAL_CONTROLLER.claim'
$taskStatusPath = Join-Path $taskStudy 'FORMAL_STATUS.json'
$taskUtf8 = [Text.UTF8Encoding]::new($false)
$taskOwned = $false
$taskState = @{status='initializing';host=$env:COMPUTERNAME;controller_pid=$PID;started_at=[DateTime]::UtcNow.ToString('o');run_ids=@{};phases=@();current_phase=$null;child_pid=$null;formal_evaluation_started=$false}
Set-Location -LiteralPath $taskRoot
$env:PYTHONPATH = Join-Path $taskRoot '_site'
$env:YOLO_OFFLINE = 'true'
$env:YOLO_AUTOINSTALL = 'false'
$env:YOLO_CONFIG_DIR = Join-Path $taskRoot '_ultralytics_config'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
Remove-Item Env:MKL_NUM_THREADS -ErrorAction SilentlyContinue

function Save-FormalState {
    $taskState.updated_at = [DateTime]::UtcNow.ToString('o')
    $taskTemporary = Join-Path $taskStudy 'FORMAL_STATUS.write.json'
    [IO.File]::WriteAllText($taskTemporary,($taskState | ConvertTo-Json -Depth 15),$taskUtf8)
    Move-Item -LiteralPath $taskTemporary -Destination $taskStatusPath -Force
}

function Invoke-FrozenChild([string]$Label,[string[]]$Arguments,[int]$BudgetSeconds) {
    $taskOut=Join-Path $taskStudy ($Label+'.stdout.log')
    $taskErr=Join-Path $taskStudy ($Label+'.stderr.log')
    if ((Test-Path -LiteralPath $taskOut) -or (Test-Path -LiteralPath $taskErr)) {throw ('Stage already has output: '+$Label)}
    $taskChild=Start-Process -FilePath $taskPython -ArgumentList $Arguments -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput $taskOut -RedirectStandardError $taskErr -PassThru
    $taskState.current_phase=$Label
    $taskState.child_pid=$taskChild.Id
    $taskState.status='running'
    Save-FormalState
    if (-not $taskChild.WaitForExit($BudgetSeconds*1000)) {
        Stop-Process -Id $taskChild.Id -Force
        $taskState.phases+=@{label=$Label;pid=$taskChild.Id;status='wall_budget_exhausted';arguments=$Arguments}
        throw ('Frozen wall budget exhausted: '+$Label)
    }
    $taskChild.WaitForExit()
    $taskExit=$taskChild.ExitCode
    $taskState.phases+=@{label=$Label;pid=$taskChild.Id;exit_code=$taskExit;arguments=$Arguments;ended_at=[DateTime]::UtcNow.ToString('o')}
    $taskState.child_pid=$null
    Save-FormalState
    if ($taskExit -ne 0) {throw ('Child failed: '+$Label+' exit '+$taskExit)}
}

try {
    if (Test-Path -LiteralPath $taskStatusPath) {throw 'Formal controller already has state; inspect rather than restart'}
    $taskHandle=[IO.File]::Open($taskClaim,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    $taskOwned=$true
    $taskBytes=$taskUtf8.GetBytes(($taskState|ConvertTo-Json -Depth 8))
    $taskHandle.Write($taskBytes,0,$taskBytes.Length)
    $taskHandle.Close()
    $taskFreeze=Get-Content -LiteralPath (Join-Path $taskStudy 'FORMAL_CONTROL_FREEZE.json') -Raw|ConvertFrom-Json
    foreach($taskProperty in $taskFreeze.files.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath (Join-Path $taskStudy $taskProperty.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskProperty.Value) {throw ('Control source hash mismatch: '+$taskProperty.Name)}
    }
    $taskPrecheck=Get-Content -LiteralPath (Join-Path $taskStudy 'FORMAL_PRELAUNCH_CHECK.json') -Raw|ConvertFrom-Json
    if ($taskPrecheck.status -ne 'passed') {throw 'Prelaunch audit incomplete'}
    $taskSnapshot=Join-Path $taskStudy 'formal_control_snapshot'
    New-Item -ItemType Directory -Path $taskSnapshot -ErrorAction Stop|Out-Null
    foreach($taskProperty in $taskFreeze.files.PSObject.Properties) {
        Copy-Item -LiteralPath (Join-Path $taskStudy $taskProperty.Name) -Destination (Join-Path $taskSnapshot ($taskProperty.Name.Replace('/','__')))
    }
    Save-FormalState
    $taskPreflight=Join-Path $taskStudy 'runs\RUN_CLD_PREFLIGHT_a436d52ceed643638c6f32be616f9cfe'
    $taskSmokeReal=Join-Path $taskStudy 'runs\RUN_CLD_SMOKE_REAL_396e8c25780f4357b21bc76c7431db2a'
    $taskSmokePerm=Join-Path $taskStudy 'runs\RUN_CLD_SMOKE_PERM_0761445d8fd2437795f783d6b6acdba8'
    foreach($taskArm in @('REAL','PERM')) {
        if (@(Get-ChildItem -LiteralPath (Join-Path $taskStudy 'runs') -Directory -Filter ('RUN_CLD_FULL_'+$taskArm+'_*')).Count -gt 0) {throw ('Existing formal arm: '+$taskArm)}
        $taskArgs=@('-B',(Join-Path $taskStudy 'scripts\train_cld.py'),'--scope','full','--arm',$taskArm,'--preflight',$taskPreflight,'--smoke-real',$taskSmokeReal,'--smoke-perm',$taskSmokePerm)
        Invoke-FrozenChild ('FORMAL_TRAIN_'+$taskArm) $taskArgs 7200
        $taskRuns=@(Get-ChildItem -LiteralPath (Join-Path $taskStudy 'runs') -Directory -Filter ('RUN_CLD_FULL_'+$taskArm+'_*'))
        if ($taskRuns.Count -ne 1) {throw 'Expected exactly one actual formal Run'}
        $taskRun=$taskRuns[0]
        $taskState.run_ids[$taskArm]=$taskRun.Name
        Save-FormalState
        $taskArgs=@('-B',(Join-Path $taskStudy 'scripts\verify_cld.py'),$taskRun.FullName)
        Invoke-FrozenChild ('FORMAL_VERIFY_'+$taskArm) $taskArgs 3600
        $taskIndependent=Get-Content -LiteralPath (Join-Path $taskRun.FullName 'INDEPENDENT.json') -Raw|ConvertFrom-Json
        if ($taskIndependent.status -ne 'passed') {throw 'Independent check failed'}
    }
    $taskState.status='completed_training_and_verification'
    $taskState.current_phase='closed_no_evaluation'
    $taskState.ended_at=[DateTime]::UtcNow.ToString('o')
    Save-FormalState
} catch {
    if ($taskOwned) {
        $taskState.status='failed'
        $taskState.error=$_.ToString()
        $taskState.ended_at=[DateTime]::UtcNow.ToString('o')
        Save-FormalState
    }
    throw
} finally {
    if ($taskOwned) {
        Remove-Item -LiteralPath $taskClaim -ErrorAction SilentlyContinue
        Disable-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue | Out-Null
    }
}
