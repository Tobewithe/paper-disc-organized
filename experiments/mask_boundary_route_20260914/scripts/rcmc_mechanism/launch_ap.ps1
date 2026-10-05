$ErrorActionPreference='Stop'
$root='D:/coco_wire/scripts/rcmc_mechanism_20260920'
$runs='D:/coco_wire/runs/mask_boundary_route_20260914'
$deadline=(Get-Date).AddHours(3)
foreach($dependency in @('RUN_c70956b46a3d4e769b7982a4d9267afd','RUN_9fa3374a7519409fb2b09995ce37affb')) {
 while($true) {
  if((Get-Date) -gt $deadline){throw "Dependency timeout: $dependency"}
  if(Test-Path "$runs/$dependency/run.json") {
   $r=Get-Content "$runs/$dependency/run.json" -Raw|ConvertFrom-Json
   if($r.status -eq 'completed' -and $r.return_code -eq 0){break}
   if($r.status -in @('failed','interrupted','cancelled')){throw "Dependency failed: $dependency"}
  }
  Start-Sleep -Seconds 15
 }
}
$python='C:/Users/28358/anaconda3/envs/pytorch/python.exe'
$env:PYTHONPATH='D:/coco_wire/py';$env:PYTHONUNBUFFERED='1';$env:OMP_NUM_THREADS='4';$env:MKL_NUM_THREADS='4'
$id='RUN_a720b6dc7b7448a7badb5bfd94df1d9e';$output="$runs/$id";$protocol="$root/PROTOCOL_MECHANISM_V6_20260920.json"
$snapshots=@('--snapshot',"$root/conditional_ap.py",'--snapshot',"$root/common.py",'--snapshot',$PSCommandPath,'--snapshot',$protocol)
$inputs=@('--input',$protocol,'--input',"$root/inputs/decisions.csv",'--input',"$root/inputs/predictions_risk_response.json",'--input','D:/coco_wire/data/annotations/instances_val2017.json')
foreach($name in @('candidate_records.csv','instance_records.csv','image_ids.json','predictions_official_zero.json')){$inputs+=@('--input',"$runs/RUN_2ae67d6556a14848bceb5783a72e5790/$name")}
& $python "$root/runner.py" --study STUDY_8fb3468ebb704682a2225ebed0e16206 --run-id $id --output $output --cwd D:/coco_wire @inputs @snapshots --expect "$output/SUMMARY.json" --metrics "$output/SUMMARY.json" -- $python -u "$root/conditional_ap.py" --protocol $protocol --output $output --response-predictions "$root/inputs/predictions_risk_response.json"
if($LASTEXITCODE -ne 0){throw 'Conditional AP failed'}
