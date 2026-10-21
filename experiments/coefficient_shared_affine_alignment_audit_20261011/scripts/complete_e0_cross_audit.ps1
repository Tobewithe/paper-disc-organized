$ErrorActionPreference = 'Stop'
# E0 completion: the interrupted official640 solver instance (exact queued arguments) followed by
# the cross evaluator over the existing old160 map and the new official640 map.
# Historical runs are not touched; both executions land in new run directories.
$python = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$root = 'D:\coco_wire\experiments\coefficient_shared_affine_alignment_audit_20261011'
$cache = 'D:\coco_wire\data\official_tal_affine_20260930\runs\official_cache'
$annotations = 'D:\coco_wire\data\annotations'
$vendor = 'D:\coco_wire\vendor_8.4.100'
$solverOut = Join-Path $root 'runs/RUN_CROSS_OFFICIAL640_COMPLETE_20261012'
$evalOut = Join-Path $root 'runs/RUN_CROSS_EVAL_COMPLETE_20261012'
$oldMap = Join-Path $root 'runs/RUN_CROSS_OLD160_20261011_053905_129/SHARED_old160.pt'

New-Item -ItemType Directory -Force -Path $solverOut, $evalOut | Out-Null
$log = Join-Path $solverOut 'stdout.log'
$err = Join-Path $solverOut 'stderr.log'

# Stage 1: official640 solver with the exact queued arguments (new run directory)
& $python "$root\scripts\solve_cross.py" --cache $cache --annotations $annotations --mode official640 --out $solverOut --device cuda --seed 20261011 *> $log
$solverExit = $LASTEXITCODE
"solver_exit=$solverExit at $(Get-Date -Format o)" | Set-Content -LiteralPath (Join-Path $solverOut 'TASK_EXIT.txt')
if ($solverExit -ne 0) { exit $solverExit }

# Stage 2: assertions identical in spirit to the queue script
$model = Join-Path $solverOut 'SHARED_official640.pt'
$summary = Get-Content -LiteralPath (Join-Path $solverOut 'SUMMARY.json') -Raw | ConvertFrom-Json
$run = Get-Content -LiteralPath (Join-Path $solverOut 'run.json') -Raw | ConvertFrom-Json
foreach ($p in @($model, (Join-Path $solverOut 'SUMMARY.json'), (Join-Path $solverOut 'run.json'))) {
    if (-not (Test-Path -LiteralPath $p)) { throw "solver missing artifact: $p" }
}
if ($summary.status -ne 'complete') { throw "solver summary status not complete" }
if ($run.status -ne 'completed') { throw "solver run status not completed" }
if ([int]$summary.fit_images -ne 796) { throw "fit_images mismatch: $($summary.fit_images)" }
if (-not [bool]$summary.solver.converged) { throw "solver did not converge" }
if ([double]$summary.solver.gradient_inf -gt 1e-6) { throw "gradient_inf exceeds 1e-6: $($summary.solver.gradient_inf)" }

# Stage 3: evaluator over both maps
$env:PYTHONPATH = "$vendor;$env:PYTHONPATH"
$env:PYTHONUNBUFFERED = '1'
& $python "$root\scripts\evaluate_cross.py" --cache $cache --annotations $annotations `
  --old-map $oldMap --official-map $model --out $evalOut --groups fit dev val --device cuda *> (Join-Path $evalOut 'stdout.log')
$evalExit = $LASTEXITCODE
"evaluator_exit=$evalExit at $(Get-Date -Format o)" | Set-Content -LiteralPath (Join-Path $evalOut 'TASK_EXIT.txt')
if ($evalExit -ne 0) { exit $evalExit }
foreach ($p in @((Join-Path $evalOut 'SUMMARY.json'), (Join-Path $evalOut 'run.json'), (Join-Path $evalOut 'manifest.sha256'))) {
    if (-not (Test-Path -LiteralPath $p)) { throw "evaluator missing artifact: $p" }
}
$eRun = Get-Content -LiteralPath (Join-Path $evalOut 'run.json') -Raw | ConvertFrom-Json
if ($eRun.status -ne 'complete') { throw "evaluator run status not complete" }
"e0_completion_done at $(Get-Date -Format o)" | Set-Content -LiteralPath (Join-Path $evalOut 'COMPLETION_DONE.txt')
