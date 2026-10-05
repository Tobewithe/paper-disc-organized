param()
$ErrorActionPreference="Stop"
$base="D:\coco_wire\research\mask_quality_boundary_branch_20261004\scripts"
$exe="C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
foreach($n in @("launch_protocol_pretrained_native.ps1","launch_protocol_baseline_native.ps1","launch_protocol_method_native.ps1")){
  & $exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $base $n)
}

