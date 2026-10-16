$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$taskRoot = 'D:\coco_wire\baseline_native_20261013'
$taskStudy = Join-Path $taskRoot 'experiments\mask_direction_learnability_20261016'
$taskPython = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$taskName = 'Codex-CLD-Evaluation-20261016'
$taskClaim = Join-Path $taskStudy 'EVALUATION_CONTROLLER.claim'
$taskStatusPath = Join-Path $taskStudy 'EVALUATION_STATUS.json'
$taskUtf8 = [Text.UTF8Encoding]::new($false)
$taskOwned = $false
$taskState = @{status='initializing';host=$env:COMPUTERNAME;controller_pid=$PID;started_at=[DateTime]::UtcNow.ToString('o');run_id=$null;phases=@();current_phase=$null;child_pid=$null;new_optimizer_steps=0}
Set-Location -LiteralPath $taskRoot
$env:PYTHONPATH = Join-Path $taskRoot '_site'
$env:YOLO_OFFLINE = 'true'
$env:YOLO_AUTOINSTALL = 'false'
$env:YOLO_CONFIG_DIR = Join-Path $taskRoot '_ultralytics_config'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
Remove-Item Env:MKL_NUM_THREADS -ErrorAction SilentlyContinue

function Save-EvaluationState {
    $taskState.updated_at = [DateTime]::UtcNow.ToString('o')
    $taskTemporary = Join-Path $taskStudy 'EVALUATION_STATUS.write.json'
    [IO.File]::WriteAllText($taskTemporary,($taskState | ConvertTo-Json -Depth 15),$taskUtf8)
    Move-Item -LiteralPath $taskTemporary -Destination $taskStatusPath -Force
}

function Invoke-EvaluationChild([string]$Label,[string[]]$Arguments,[int]$BudgetSeconds) {
    $taskOut = Join-Path $taskStudy ($Label+'.stdout.log')
    $taskErr = Join-Path $taskStudy ($Label+'.stderr.log')
    if ((Test-Path -LiteralPath $taskOut) -or (Test-Path -LiteralPath $taskErr)) {throw ('Stage already has output: '+$Label)}
    $taskChild = Start-Process -FilePath $taskPython -ArgumentList $Arguments -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput $taskOut -RedirectStandardError $taskErr -PassThru
    $taskState.current_phase = $Label
    $taskState.child_pid = $taskChild.Id
    $taskState.status = 'running'
    Save-EvaluationState
    if (-not $taskChild.WaitForExit($BudgetSeconds*1000)) {
        Stop-Process -Id $taskChild.Id -Force
        $taskState.phases += @{label=$Label;pid=$taskChild.Id;status='wall_budget_exhausted';arguments=$Arguments}
        throw ('Frozen wall budget exhausted: '+$Label)
    }
    $taskChild.WaitForExit()
    $taskExit = $taskChild.ExitCode
    $taskState.phases += @{label=$Label;pid=$taskChild.Id;exit_code=$taskExit;arguments=$Arguments;ended_at=[DateTime]::UtcNow.ToString('o')}
    $taskState.child_pid = $null
    $taskRuns = @(Get-ChildItem -LiteralPath (Join-Path $taskStudy 'runs') -Directory -Filter 'RUN_CLD_EVALUATION_*')
    if ($taskRuns.Count -eq 1) {$taskState.run_id=$taskRuns[0].Name}
    Save-EvaluationState
    if ($taskExit -ne 0) {throw ('Child failed: '+$Label+' exit '+$taskExit)}
}

try {
    if (Test-Path -LiteralPath $taskStatusPath) {throw 'Evaluation already has state; inspect rather than restart'}
    if (@(Get-ChildItem -LiteralPath (Join-Path $taskStudy 'runs') -Directory -Filter 'RUN_CLD_EVALUATION_*').Count -ne 0) {throw 'Existing evaluation Run'}
    $taskHandle = [IO.File]::Open($taskClaim,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    $taskOwned = $true
    $taskBytes = $taskUtf8.GetBytes(($taskState | ConvertTo-Json -Depth 8))
    $taskHandle.Write($taskBytes,0,$taskBytes.Length)
    $taskHandle.Close()
    $taskFreeze = Get-Content -LiteralPath (Join-Path $taskStudy 'EVALUATION_CONTROL_FREEZE.json') -Raw | ConvertFrom-Json
    foreach($taskProperty in $taskFreeze.files.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath (Join-Path $taskStudy $taskProperty.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskProperty.Value) {throw ('Control source mismatch: '+$taskProperty.Name)}
    }
    $taskPrecheck = Get-Content -LiteralPath (Join-Path $taskStudy 'EVALUATION_PRELAUNCH_CHECK.json') -Raw | ConvertFrom-Json
    if ($taskPrecheck.status -ne 'passed') {throw 'Prelaunch audit incomplete'}
    $taskSnapshot = Join-Path $taskStudy 'evaluation_control_snapshot'
    New-Item -ItemType Directory -Path $taskSnapshot -ErrorAction Stop | Out-Null
    foreach($taskProperty in $taskFreeze.files.PSObject.Properties) {
        Copy-Item -LiteralPath (Join-Path $taskStudy $taskProperty.Name) -Destination (Join-Path $taskSnapshot ($taskProperty.Name.Replace('/','__')))
    }
    Save-EvaluationState
    $taskRegistry = Get-Content -LiteralPath (Join-Path $taskStudy 'TRAINED_CHECKPOINTS.json') -Raw | ConvertFrom-Json
    $taskReal = Join-Path $taskStudy ('runs\'+$taskRegistry.checkpoints.REAL.run_id)
    $taskPerm = Join-Path $taskStudy ('runs\'+$taskRegistry.checkpoints.PERM.run_id)
    Invoke-EvaluationChild 'EVALUATION_RUN' @('-B',(Join-Path $taskStudy 'scripts\evaluate_cld.py'),'--real',$taskReal,'--perm',$taskPerm) 7200
    $taskRuns = @(Get-ChildItem -LiteralPath (Join-Path $taskStudy 'runs') -Directory -Filter 'RUN_CLD_EVALUATION_*')
    if ($taskRuns.Count -ne 1) {throw 'Expected exactly one actual evaluation Run'}
    Invoke-EvaluationChild 'EVALUATION_VERIFY' @('-B',(Join-Path $taskStudy 'scripts\verify_cld.py'),$taskRuns[0].FullName) 3600
    $taskIndependent = Get-Content -LiteralPath (Join-Path $taskRuns[0].FullName 'INDEPENDENT.json') -Raw | ConvertFrom-Json
    if ($taskIndependent.status -ne 'passed') {throw 'Independent evaluation check failed'}
    $taskState.status = 'completed_evaluation_and_verification'
    $taskState.current_phase = 'closed_no_automatic_next_stage'
    $taskState.ended_at = [DateTime]::UtcNow.ToString('o')
    Save-EvaluationState
} catch {
    if ($taskOwned) {
        $taskState.status = 'failed'
        $taskState.error = $_.ToString()
        $taskState.ended_at = [DateTime]::UtcNow.ToString('o')
        Save-EvaluationState
    }
    throw
} finally {
    if ($taskOwned) {
        Remove-Item -LiteralPath $taskClaim -ErrorAction SilentlyContinue
        $taskDisableOutput = & "$env:SystemRoot\System32\schtasks.exe" /Change /TN $taskName /Disable 2>&1
        $taskState.task_disable_exit_code = $LASTEXITCODE
        $taskState.task_disable_output = ($taskDisableOutput | Out-String).Trim()
        Save-EvaluationState
    }
}
