param()
$ErrorActionPreference = "Stop"
$base = "D:\coco_wire\research\mask_quality_boundary_branch_20261004"
$baseline = Join-Path $base "runs\RUN_baseline_full_available_e3_s0_v3\run.json"
$method = Join-Path $base "scripts\launch_boundary_full_available_e3_s0_v2.ps1"
$logOut = Join-Path $base "runs\method_chain.out.log"
$logErr = Join-Path $base "runs\method_chain.err.log"
while (-not (Test-Path $baseline)) { Start-Sleep -Seconds 20 }
do {
  try { $r = Get-Content $baseline -Raw | ConvertFrom-Json; $status = $r.status } catch { $status = "reading" }
  if ($status -eq "running" -or $status -eq "preparing") { Start-Sleep -Seconds 30 }
} while ($status -eq "running" -or $status -eq "preparing")
if ($status -eq "completed" -and $r.return_code -eq 0) {
  Start-Process -WindowStyle Hidden -FilePath "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File",$method) -RedirectStandardOutput $logOut -RedirectStandardError $logErr
} else {
  "baseline_status=$status return_code=$($r.return_code)" | Set-Content (Join-Path $base "runs\method_chain_blocked.txt")
}

