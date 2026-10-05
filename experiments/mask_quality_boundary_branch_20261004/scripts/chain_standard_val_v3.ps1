param()
$ErrorActionPreference = "Stop"
$base = "D:\coco_wire\research\mask_quality_boundary_branch_20261004"
$baselineEval = Join-Path $base "runs\RUN_eval_standard_val5k_baseline_v3\run.json"
$method = Join-Path $base "scripts\launch_eval_standard_method_v3.ps1"
$methodEvalRun = Join-Path $base "runs\RUN_eval_standard_val5k_method_v3\run.json"
$exe = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
while (-not (Test-Path $baselineEval)) { Start-Sleep -Seconds 15 }
do {
  try { $r = Get-Content $baselineEval -Raw | ConvertFrom-Json; $status = $r.status } catch { $status = "reading" }
  if ($status -eq "running" -or $status -eq "preparing") { Start-Sleep -Seconds 20 }
} while ($status -eq "running" -or $status -eq "preparing")
if ($status -eq "completed" -and $r.return_code -eq 0) {
  Start-Process -WindowStyle Hidden -FilePath $exe -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File",$method) -RedirectStandardOutput (Join-Path $base "runs\eval_method_v3_chain.out.log") -RedirectStandardError (Join-Path $base "runs\eval_method_v3_chain.err.log")
} else {
  "baseline_eval_status=$status return_code=$($r.return_code)" | Set-Content (Join-Path $base "runs\method_eval_v3_blocked.txt")
}

