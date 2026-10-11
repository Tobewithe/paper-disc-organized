<#
.SYNOPSIS
    Queue the old160 and official640 cross-objective solvers, then evaluate.

.DESCRIPTION
    This PowerShell file is only an execution queue. It does not implement
    either objective, fit parameters, or evaluate masks. The parent process
    detaches one hidden worker; the worker starts both Python solvers in
    parallel, waits for both exit codes, checks their completion sentinels,
    and only then starts the single evaluator.

    By default -FitImages 0 means all 796 fit images. A positive value, such
    as 256, is passed to both solvers and written explicitly to queue metadata.
    The fixed cache and environment paths are part of the audit protocol and
    must remain identical for both arms.
#>
[CmdletBinding()]
param(
    [switch]$AsWorker,
    [string]$StudyRoot = "D:\coco_wire\experiments\coefficient_shared_affine_alignment_audit_20261011",
    [string]$Cache = "D:\coco_wire\data\official_tal_affine_20260930\runs\official_cache",
    [string]$Annotations = "D:\coco_wire\data\annotations",
    [string]$Python = "C:\Users\28358\anaconda3\envs\pytorch\python.exe",
    [string]$PowerShell = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
    [string]$Vendor = "D:\coco_wire\vendor_8.4.100",
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "cuda",
    [int]$FitImages = 0,
    [int]$Seed = 20261011,
    [string]$QueueId = "",
    [string]$QueueDir = "",
    [string]$OldRunName = "",
    [string]$OfficialRunName = "",
    [string]$EvalRunName = ""
)

$ErrorActionPreference = "Stop"
$Runs = Join-Path $StudyRoot "runs"
$Scripts = Join-Path $StudyRoot "scripts"

if ([string]::IsNullOrWhiteSpace($QueueId)) {
    $tag = [DateTime]::UtcNow.ToString("yyyyMMdd_HHmmss_fff")
    $QueueId = "RUN_CROSS_QUEUE_$tag"
    $OldRunName = "RUN_CROSS_OLD160_$tag"
    $OfficialRunName = "RUN_CROSS_OFFICIAL640_$tag"
    $EvalRunName = "RUN_CROSS_EVAL_$tag"
}
if ([string]::IsNullOrWhiteSpace($QueueDir)) { $QueueDir = Join-Path $Runs $QueueId }
if ([string]::IsNullOrWhiteSpace($OldRunName)) { $OldRunName = "RUN_CROSS_OLD160_$($QueueId -replace '^RUN_CROSS_QUEUE_', '')" }
if ([string]::IsNullOrWhiteSpace($OfficialRunName)) { $OfficialRunName = "RUN_CROSS_OFFICIAL640_$($QueueId -replace '^RUN_CROSS_QUEUE_', '')" }
if ([string]::IsNullOrWhiteSpace($EvalRunName)) { $EvalRunName = "RUN_CROSS_EVAL_$($QueueId -replace '^RUN_CROSS_QUEUE_', '')" }

$OldOut = Join-Path $Runs $OldRunName
$OfficialOut = Join-Path $Runs $OfficialRunName
$EvalOut = Join-Path $Runs $EvalRunName
$QueueLog = Join-Path $QueueDir "queue.log"
$QueueStatus = Join-Path $QueueDir "QUEUE_STATUS.json"
$QueueRun = Join-Path $QueueDir "run.json"

function Write-QueueStatus([hashtable]$Value) {
    $Value.queue_id = $QueueId
    $Value.timestamp_utc = [DateTime]::UtcNow.ToString("o")
    ($Value | ConvertTo-Json -Depth 16) | Set-Content -Encoding UTF8 $QueueStatus
    ($Value | ConvertTo-Json -Compress -Depth 16) | Add-Content -Encoding UTF8 $QueueLog
}

function Write-RunJson([hashtable]$Value) {
    $Value.queue_id = $QueueId
    $Value.updated_at_utc = [DateTime]::UtcNow.ToString("o")
    ($Value | ConvertTo-Json -Depth 24) | Set-Content -Encoding UTF8 $QueueRun
}

function Get-ImageCounts {
    $indexPath = Join-Path $Cache "INDEX.json"
    if (-not (Test-Path -LiteralPath $indexPath)) { throw "cache index is missing: $indexPath" }
    $index = Get-Content -LiteralPath $indexPath -Raw | ConvertFrom-Json
    return [ordered]@{ fit = @($index.fit).Count; dev = @($index.dev).Count; val = @($index.val).Count }
}

function Assert-Environment {
    foreach ($path in @($StudyRoot, $Cache, $Annotations, $Vendor, $Python, $Scripts)) {
        if (-not (Test-Path -LiteralPath $path)) { throw "required path is missing: $path" }
    }
    foreach ($scriptName in @("solve_cross.py", "evaluate_cross.py")) {
        $scriptPath = Join-Path $Scripts $scriptName
        if (-not (Test-Path -LiteralPath $scriptPath)) { throw "required Python script is missing: $scriptPath" }
    }
    $counts = Get-ImageCounts
    if ($counts.fit -ne 796 -or $counts.dev -ne 197 -or $counts.val -ne 196) {
        throw "unexpected cache image counts; expected fit/dev/val=796/197/196, got $($counts.fit)/$($counts.dev)/$($counts.val)"
    }
    if ($FitImages -lt 0 -or $FitImages -gt $counts.fit) { throw "-FitImages must be zero or between 1 and $($counts.fit)" }
    return $counts
}

function Start-PythonJob([string]$Mode, [string]$OutDir) {
    if (Test-Path -LiteralPath $OutDir) { throw "refusing to overwrite existing run directory: $OutDir" }
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $solver = Join-Path $Scripts "solve_cross.py"
    $log = Join-Path $OutDir "stdout.log"
    $err = Join-Path $OutDir "stderr.log"
    $args = @($solver, "--cache", $Cache, "--annotations", $Annotations, "--mode", $Mode, "--out", $OutDir, "--device", $Device, "--seed", "$Seed")
    if ($FitImages -gt 0) { $args += @("--fit-images", "$FitImages") }
    # Hidden is the non-interactive mode required on this laptop. Do not add
    # -NoNewWindow here: Windows PowerShell rejects it with -WindowStyle.
    return Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $StudyRoot `
        -RedirectStandardOutput $log -RedirectStandardError $err -PassThru -WindowStyle Hidden
}

function Assert-SolverComplete([string]$Mode, [string]$OutDir) {
    $model = Join-Path $OutDir "SHARED_$Mode.pt"
    $summaryPath = Join-Path $OutDir "SUMMARY.json"
    $runPath = Join-Path $OutDir "run.json"
    foreach ($path in @($model, $summaryPath, $runPath)) {
        if (-not (Test-Path -LiteralPath $path)) { throw "$Mode solver missing required artifact: $path" }
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    $run = Get-Content -LiteralPath $runPath -Raw | ConvertFrom-Json
    if ($summary.status -ne "complete") { throw "$Mode solver summary status is not complete: $($summary.status)" }
    if ($run.status -ne "completed") { throw "$Mode solver run status is not completed: $($run.status)" }
    $expectedFit = if ($FitImages -eq 0) { 796 } else { $FitImages }
    if ([int]$summary.fit_images -ne $expectedFit) { throw "$Mode solver fit_images metadata mismatch: expected $expectedFit, got $($summary.fit_images)" }
    if (-not [bool]$summary.solver.converged) { throw "$Mode solver did not reach the configured gradient convergence" }
    if ([double]$summary.solver.gradient_inf -gt 1e-6) { throw "$Mode solver gradient_inf exceeds 1e-6: $($summary.solver.gradient_inf)" }
}

function Assert-EvaluatorComplete([string]$OutDir) {
    foreach ($path in @((Join-Path $OutDir "SUMMARY.json"), (Join-Path $OutDir "run.json"), (Join-Path $OutDir "manifest.sha256"))) {
        if (-not (Test-Path -LiteralPath $path)) { throw "evaluator missing required artifact: $path" }
    }
    $run = Get-Content -LiteralPath (Join-Path $OutDir "run.json") -Raw | ConvertFrom-Json
    if ($run.status -ne "complete") { throw "evaluator run status is not complete: $($run.status)" }
}

if (-not $AsWorker) {
    New-Item -ItemType Directory -Force -Path $StudyRoot, $Runs | Out-Null
    if (Test-Path -LiteralPath $QueueDir) { throw "refusing to overwrite existing queue directory: $QueueDir" }
    foreach ($path in @($OldOut, $OfficialOut, $EvalOut)) {
        if (Test-Path -LiteralPath $path) { throw "refusing to overwrite existing run directory: $path" }
    }
    New-Item -ItemType Directory -Force -Path $QueueDir | Out-Null
    New-Item -ItemType File -Force -Path $QueueLog | Out-Null
    try {
        $counts = Assert-Environment
        $fitScope = if ($FitImages -eq 0) { "all $($counts.fit) fit images" } else { "subset of $FitImages/$($counts.fit) fit images" }
        $oldCommand = "`"$Python`" `"$(Join-Path $Scripts 'solve_cross.py')`" --cache `"$Cache`" --annotations `"$Annotations`" --mode old160 --out `"$OldOut`" --device $Device --seed $Seed" + $(if ($FitImages -gt 0) { " --fit-images $FitImages" } else { "" })
        $officialCommand = "`"$Python`" `"$(Join-Path $Scripts 'solve_cross.py')`" --cache `"$Cache`" --annotations `"$Annotations`" --mode official640 --out `"$OfficialOut`" --device $Device --seed $Seed" + $(if ($FitImages -gt 0) { " --fit-images $FitImages" } else { "" })
        $evalCommand = "`"$Python`" `"$(Join-Path $Scripts 'evaluate_cross.py')`" --cache `"$Cache`" --annotations `"$Annotations`" --old-map `"$(Join-Path $OldOut 'SHARED_old160.pt')`" --official-map `"$(Join-Path $OfficialOut 'SHARED_official640.pt')`" --out `"$EvalOut`" --groups fit dev val --device $Device"
        $meta = @{
            run_id = $QueueId; study_id = "STUDY_ALIGNMENT_20261011"; kind = "queue"; status = "queued"
            started_at_utc = [DateTime]::UtcNow.ToString("o"); python = $Python; vendor = $Vendor
            cache = $Cache; annotations = $Annotations; fit_images_requested = $(if ($FitImages -eq 0) { $null } else { $FitImages })
            fit_images_scope = $fitScope; expected_images = $counts
            runs = @{ old160 = $OldRunName; official640 = $OfficialRunName; evaluator = $EvalRunName }
            commands = @{ old160 = $oldCommand; official640 = $officialCommand; evaluator = $evalCommand }
            queue_script = $PSCommandPath
        }
        Write-QueueStatus @{ status = "queued"; phase = "preparing"; fit_images_scope = $fitScope; expected_images = $counts; runs = $meta.runs }
        Write-RunJson $meta
        $workerArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath, "-AsWorker", "-StudyRoot", $StudyRoot, "-Cache", $Cache, "-Annotations", $Annotations, "-Python", $Python, "-PowerShell", $PowerShell, "-Vendor", $Vendor, "-Device", $Device, "-FitImages", "$FitImages", "-Seed", "$Seed", "-QueueId", $QueueId, "-QueueDir", $QueueDir, "-OldRunName", $OldRunName, "-OfficialRunName", $OfficialRunName, "-EvalRunName", $EvalRunName)
        $workerLog = Join-Path $QueueDir "worker_stdout.log"
        $workerErr = Join-Path $QueueDir "worker_stderr.log"
        $workerProc = Start-Process -FilePath $PowerShell -ArgumentList $workerArgs -WorkingDirectory $StudyRoot `
            -RedirectStandardOutput $workerLog -RedirectStandardError $workerErr -PassThru -WindowStyle Hidden
        $meta.worker_pid = $workerProc.Id
        $meta.status = "running"
        Write-QueueStatus @{ status = "running"; phase = "worker_detached"; worker_pid = $workerProc.Id; fit_images_scope = $fitScope; expected_images = $counts; runs = $meta.runs }
        Write-RunJson $meta
        Write-Output ("queue_started pid={0} queue_run={1}" -f $workerProc.Id, $QueueDir)
        exit 0
    }
    catch {
        $failure = @{ status = "queue_failed"; phase = "preparing"; error = $_.Exception.ToString() }
        Write-QueueStatus $failure
        Write-RunJson @{ run_id = $QueueId; study_id = "STUDY_ALIGNMENT_20261011"; kind = "queue"; status = "failed"; error = $_.Exception.ToString(); cache = $Cache; annotations = $Annotations; python = $Python; vendor = $Vendor }
        exit 1
    }
}

try {
    $counts = Assert-Environment
    $fitScope = if ($FitImages -eq 0) { "all $($counts.fit) fit images" } else { "subset of $FitImages/$($counts.fit) fit images" }
    $oldCommand = "`"$Python`" `"$(Join-Path $Scripts 'solve_cross.py')`" --cache `"$Cache`" --annotations `"$Annotations`" --mode old160 --out `"$OldOut`" --device $Device --seed $Seed" + $(if ($FitImages -gt 0) { " --fit-images $FitImages" } else { "" })
    $officialCommand = "`"$Python`" `"$(Join-Path $Scripts 'solve_cross.py')`" --cache `"$Cache`" --annotations `"$Annotations`" --mode official640 --out `"$OfficialOut`" --device $Device --seed $Seed" + $(if ($FitImages -gt 0) { " --fit-images $FitImages" } else { "" })
    $evalCommand = "`"$Python`" `"$(Join-Path $Scripts 'evaluate_cross.py')`" --cache `"$Cache`" --annotations `"$Annotations`" --old-map `"$(Join-Path $OldOut 'SHARED_old160.pt')`" --official-map `"$(Join-Path $OfficialOut 'SHARED_official640.pt')`" --out `"$EvalOut`" --groups fit dev val --device $Device"
    $env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) { $Vendor } else { "$Vendor;$($env:PYTHONPATH)" }
    $env:PYTHONUNBUFFERED = "1"
    Write-QueueStatus @{ status = "solvers_starting"; phase = "solvers_starting"; worker_pid = $PID; fit_images_scope = $fitScope; expected_images = $counts; runs = @{ old160 = $OldRunName; official640 = $OfficialRunName; evaluator = $EvalRunName } }
    $old = Start-PythonJob "old160" $OldOut
    $official = Start-PythonJob "official640" $OfficialOut
    Write-QueueStatus @{ status = "solvers_running"; phase = "solvers_running"; worker_pid = $PID; old_pid = $old.Id; official_pid = $official.Id; fit_images_scope = $fitScope }
    Wait-Process -Id @($old.Id, $official.Id)
    $old.Refresh(); $official.Refresh()
    $oldExit = [int]$old.ExitCode
    $officialExit = [int]$official.ExitCode
    if ($oldExit -ne 0 -or $officialExit -ne 0) { throw "solver failure; old160=$oldExit official640=$officialExit" }
    Assert-SolverComplete "old160" $OldOut
    Assert-SolverComplete "official640" $OfficialOut
    Write-QueueStatus @{ status = "solvers_complete"; phase = "solvers_complete"; worker_pid = $PID; old_exit = $oldExit; official_exit = $officialExit; old_model = (Join-Path $OldOut "SHARED_old160.pt"); official_model = (Join-Path $OfficialOut "SHARED_official640.pt") }

    $eval = Join-Path $Scripts "evaluate_cross.py"
    New-Item -ItemType Directory -Force -Path $EvalOut | Out-Null
    $evalLog = Join-Path $EvalOut "stdout.log"
    $evalErr = Join-Path $EvalOut "stderr.log"
    $evalArgs = @($eval, "--cache", $Cache, "--annotations", $Annotations, "--old-map", (Join-Path $OldOut "SHARED_old160.pt"), "--official-map", (Join-Path $OfficialOut "SHARED_official640.pt"), "--out", $EvalOut, "--groups", "fit", "dev", "val", "--device", $Device)
    $e = Start-Process -FilePath $Python -ArgumentList $evalArgs -WorkingDirectory $StudyRoot `
        -RedirectStandardOutput $evalLog -RedirectStandardError $evalErr -PassThru -WindowStyle Hidden
    Write-QueueStatus @{ status = "evaluation_running"; phase = "evaluation_running"; worker_pid = $PID; eval_pid = $e.Id }
    Wait-Process -Id $e.Id
    $e.Refresh()
    $evalExit = [int]$e.ExitCode
    if ($evalExit -ne 0) { throw "evaluator failed with exit code $evalExit" }
    Assert-EvaluatorComplete $EvalOut
    Write-QueueStatus @{ status = "complete"; phase = "complete"; worker_pid = $PID; eval_exit = 0; evaluation = $EvalOut; old_map = (Join-Path $OldOut "SHARED_old160.pt"); official_map = (Join-Path $OfficialOut "SHARED_official640.pt") }
    Write-RunJson @{ run_id = $QueueId; study_id = "STUDY_ALIGNMENT_20261011"; kind = "queue"; status = "completed"; completed_at_utc = [DateTime]::UtcNow.ToString("o"); cache = $Cache; annotations = $Annotations; python = $Python; vendor = $Vendor; fit_images_requested = $(if ($FitImages -eq 0) { $null } else { $FitImages }); fit_images_scope = $fitScope; expected_images = $counts; runs = @{ old160 = $OldRunName; official640 = $OfficialRunName; evaluator = $EvalRunName }; commands = @{ old160 = $oldCommand; official640 = $officialCommand; evaluator = $evalCommand }; worker_pid = $PID; exit_codes = @{ old160 = $oldExit; official640 = $officialExit; evaluator = $evalExit } }
}
catch {
    $message = $_.Exception.ToString()
    try { Write-QueueStatus @{ status = "queue_failed"; phase = "failed"; worker_pid = $PID; error = $message } } catch { }
    try { Write-RunJson @{ run_id = $QueueId; study_id = "STUDY_ALIGNMENT_20261011"; kind = "queue"; status = "failed"; error = $message; cache = $Cache; annotations = $Annotations; python = $Python; vendor = $Vendor } } catch { }
    exit 1
}
