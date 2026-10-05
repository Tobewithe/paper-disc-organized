param()
$ErrorActionPreference = "Stop"
$base = "D:\coco_wire\research\mask_quality_boundary_branch_20261004"
$methodTrain = Join-Path $base "runs\RUN_boundary_full_available_e3_s0_v2\run.json"
$baselineEval = Join-Path $base "scripts\launch_eval_standard_baseline.ps1"
$methodEval = Join-Path $base "scripts\launch_eval_standard_method.ps1"
$baseEvalRun = Join-Path $base "runs\RUN_eval_standard_val5k_baseline\run.json"
$methodEvalRun = Join-Path $base "runs\RUN_eval_standard_val5k_method\run.json"
$exe = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
function Wait-Run($path) {
  while (-not (Test-Path $path)) { Start-Sleep -Seconds 30 }
  do {
    try { $r = Get-Content $path -Raw | ConvertFrom-Json; $status = $r.status } catch { $status = "reading" }
    if ($status -eq "running" -or $status -eq "preparing") { Start-Sleep -Seconds 30 }
  } while ($status -eq "running" -or $status -eq "preparing")
  return $r
}
$r = Wait-Run $methodTrain
if ($r.status -eq "completed" -and $r.return_code -eq 0) {
  Start-Process -WindowStyle Hidden -FilePath $exe -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File",$baselineEval) -RedirectStandardOutput (Join-Path $base "runs\eval_baseline_chain.out.log") -RedirectStandardError (Join-Path $base "runs\eval_baseline_chain.err.log")
  $b = Wait-Run $baseEvalRun
  if ($b.status -eq "completed" -and $b.return_code -eq 0) {
    Start-Process -WindowStyle Hidden -FilePath $exe -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File",$methodEval) -RedirectStandardOutput (Join-Path $base "runs\eval_method_chain.out.log") -RedirectStandardError (Join-Path $base "runs\eval_method_chain.err.log")
  } else {
    "baseline_eval_status=$($b.status) return_code=$($b.return_code)" | Set-Content (Join-Path $base "runs\method_eval_blocked.txt")
  }
} else {
  "method_train_status=$($r.status) return_code=$($r.return_code)" | Set-Content (Join-Path $base "runs\method_eval_blocked.txt")
}

