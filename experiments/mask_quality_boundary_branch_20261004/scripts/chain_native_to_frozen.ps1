param()
$ErrorActionPreference="Stop"
$base="D:\coco_wire\research\mask_quality_boundary_branch_20261004"
$native=Join-Path $base "runs\RUN_protocol_method_native_v2\run.json"
$frozen=Join-Path $base "scripts\launch_boundary_frozen_full_available_e3_s0_v1.ps1"
$logOut=Join-Path $base "runs\frozen_chain.out.log"; $logErr=Join-Path $base "runs\frozen_chain.err.log"
while(-not (Test-Path $native)){Start-Sleep -Seconds 20}
while($true){
  try{$r=Get-Content $native -Raw|ConvertFrom-Json}catch{$r=$null}
  if($r -and $r.status -eq "completed"){break}
  if($r -and $r.status -eq "failed"){throw "native method evaluation failed"}
  Start-Sleep -Seconds 30
}
Start-Process -WindowStyle Hidden -FilePath "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File",$frozen) -RedirectStandardOutput $logOut -RedirectStandardError $logErr

