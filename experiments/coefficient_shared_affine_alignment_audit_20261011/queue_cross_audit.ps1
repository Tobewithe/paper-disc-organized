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
    [int]$Seed = 20261011
)

$ErrorActionPreference = "Stop"
$QueueLog = Join-Path $StudyRoot "queue.log"
$QueueStatus = Join-Path $StudyRoot "queue_status.json"
$Runs = Join-Path $StudyRoot "runs"
$Scripts = Join-Path $StudyRoot "scripts"
$OldOut = Join-Path $Runs "RUN_CROSS_OLD160_FULL_20261011"
$OfficialOut = Join-Path $Runs "RUN_CROSS_OFFICIAL640_FULL_20261011"
$EvalOut = Join-Path $Runs "RUN_CROSS_EVAL_FULL_20261011"

function Write-QueueStatus([hashtable]$Value) {
    $Value.timestamp_utc = [DateTime]::UtcNow.ToString("o")
    ($Value | ConvertTo-Json -Depth 8) | Set-Content -Encoding UTF8 $QueueStatus
    ($Value | ConvertTo-Json -Compress -Depth 8) | Add-Content -Encoding UTF8 $QueueLog
}

function Start-PythonJob([string]$Mode, [string]$OutDir) {
    $solver = Join-Path $Scripts "solve_cross.py"
    $log = Join-Path $OutDir "stdout.log"
    $err = Join-Path $OutDir "stderr.log"
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $args = @($solver, "--cache", $Cache, "--annotations", $Annotations, "--mode", $Mode, "--out", $OutDir, "--device", $Device, "--seed", "$Seed")
    if ($FitImages -gt 0) { $args += @("--fit-images", "$FitImages") }
    return Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $StudyRoot -RedirectStandardOutput $log -RedirectStandardError $err -PassThru -WindowStyle Hidden
}

function Assert-Solver([string]$OutDir, [string]$Mode) {
    $summaryPath = Join-Path $OutDir "SUMMARY.json"
    $modelPath = Join-Path $OutDir ("SHARED_{0}.pt" -f $Mode)
    if (-not (Test-Path -LiteralPath $summaryPath) -or -not (Test-Path -LiteralPath $modelPath)) {
        throw "solver $Mode did not produce SUMMARY.json and SHARED_$Mode.pt"
    }
    $summary = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
    if (-not [bool]$summary.solver.converged -or [double]$summary.solver.gradient_inf -ge 1e-6) {
        throw ("solver {0} failed gradient gate: converged={1}, gradient_inf={2}" -f $Mode, $summary.solver.converged, $summary.solver.gradient_inf)
    }
}

if (-not $AsWorker) {
    New-Item -ItemType Directory -Force -Path $StudyRoot, $Runs | Out-Null
    $start = [DateTime]::UtcNow.ToString("o")
    $initial = @{ status = "queued"; study_root = $StudyRoot; cache = $Cache; annotations = $Annotations; device = $Device; fit_images = $FitImages; seed = $Seed; started_at_utc = $start; worker_pid = $null }
    Write-QueueStatus $initial
    $workerArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath, "-AsWorker", "-StudyRoot", $StudyRoot, "-Cache", $Cache, "-Annotations", $Annotations, "-Python", $Python, "-PowerShell", $PowerShell, "-Vendor", $Vendor, "-Device", $Device, "-FitImages", "$FitImages", "-Seed", "$Seed")
    $workerLog = Join-Path $StudyRoot "worker_stdout.log"
    $workerErr = Join-Path $StudyRoot "worker_stderr.log"
    $workerProc = Start-Process -FilePath $PowerShell -ArgumentList $workerArgs -WorkingDirectory $StudyRoot -RedirectStandardOutput $workerLog -RedirectStandardError $workerErr -PassThru -WindowStyle Hidden
    $initial.worker_pid = $workerProc.Id
    $initial.status = "running"
    Write-QueueStatus $initial
    Write-Output ("queue_started pid={0} status={1}" -f $workerProc.Id, $QueueStatus)
    exit 0
}

try {
    $env:PYTHONPATH = $Vendor
    $env:PYTHONUNBUFFERED = "1"
    Write-QueueStatus @{ status = "solvers_starting"; worker_pid = $PID; old_out = $OldOut; official_out = $OfficialOut }
    $old = Start-PythonJob "old160" $OldOut
    $official = Start-PythonJob "official640" $OfficialOut
    Write-QueueStatus @{ status = "solvers_running"; worker_pid = $PID; old_pid = $old.Id; official_pid = $official.Id }
    Wait-Process -Id @($old.Id, $official.Id)
    $old.Refresh(); $official.Refresh()
    $oldExit = $old.ExitCode
    $officialExit = $official.ExitCode
    if ($oldExit -ne 0 -or $officialExit -ne 0) {
        Write-QueueStatus @{ status = "solver_failed"; worker_pid = $PID; old_exit = $oldExit; official_exit = $officialExit; next = "evaluation_not_started" }
        exit 1
    }
    Assert-Solver $OldOut "old160"
    Assert-Solver $OfficialOut "official640"
    Write-QueueStatus @{ status = "solvers_complete"; worker_pid = $PID; old_exit = $oldExit; official_exit = $officialExit }

    $eval = Join-Path $Scripts "evaluate_cross.py"
    $evalLog = Join-Path $EvalOut "stdout.log"
    $evalErr = Join-Path $EvalOut "stderr.log"
    New-Item -ItemType Directory -Force -Path $EvalOut | Out-Null
    $evalArgs = @($eval, "--cache", $Cache, "--annotations", $Annotations, "--old-map", (Join-Path $OldOut "SHARED_old160.pt"), "--official-map", (Join-Path $OfficialOut "SHARED_official640.pt"), "--out", $EvalOut, "--groups", "fit", "dev", "val", "--device", $Device)
    $e = Start-Process -FilePath $Python -ArgumentList $evalArgs -WorkingDirectory $StudyRoot -RedirectStandardOutput $evalLog -RedirectStandardError $evalErr -PassThru -WindowStyle Hidden
    Write-QueueStatus @{ status = "evaluation_running"; worker_pid = $PID; eval_pid = $e.Id }
    Wait-Process -Id $e.Id
    $e.Refresh()
    if ($e.ExitCode -ne 0) {
        Write-QueueStatus @{ status = "evaluation_failed"; worker_pid = $PID; eval_exit = $e.ExitCode }
        exit 1
    }
    if (-not (Test-Path -LiteralPath (Join-Path $EvalOut "SUMMARY.json")) -or -not (Test-Path -LiteralPath (Join-Path $EvalOut "manifest.sha256"))) {
        throw "evaluation completed without SUMMARY.json and manifest.sha256"
    }
    Write-QueueStatus @{ status = "complete"; worker_pid = $PID; eval_exit = 0; evaluation = $EvalOut; old_map = (Join-Path $OldOut "SHARED_old160.pt"); official_map = (Join-Path $OfficialOut "SHARED_official640.pt") }
} catch {
    Write-QueueStatus @{ status = "queue_failed"; worker_pid = $PID; error = $_.Exception.ToString() }
    exit 1
}
