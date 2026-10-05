$ErrorActionPreference = 'Stop'
$root = 'C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911'
$python = 'C:\Dpan\envsfiles\CondaData\envs\pytorch\python.exe'
$logRoot = Join-Path $root 'runs\boundary_ownership_20260914\queue_logs'
New-Item -ItemType Directory -Force $logRoot | Out-Null
$arms = @('baseline','self_out','self_out_neighbor')
foreach ($arm in $arms) {
  foreach ($seed in 0,1,2) {
    $tag = "${arm}_s${seed}"
    $log = Join-Path $logRoot "$tag.log"
    Add-Content $log "START $(Get-Date -Format o)"
    & $python -u (Join-Path $root 'train_boundary_ownership.py') --arm $arm --seed $seed --epochs 15 *>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    Add-Content $log "EXIT_CODE $code $(Get-Date -Format o)"
    if ($code -ne 0) { Add-Content $log "QUEUE_STOPPED_AFTER_FAILURE $tag"; exit $code }
  }
}
Set-Content (Join-Path $logRoot 'QUEUE_COMPLETE.txt') "COMPLETE $(Get-Date -Format o)"
