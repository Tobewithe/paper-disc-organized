$ErrorActionPreference='Stop'
$scriptRoot='D:/coco_wire/scripts/rcmc_mechanism_20260920'
$python='C:/Users/28358/anaconda3/envs/pytorch/python.exe'
$env:PYTHONPATH='D:/coco_wire/py'
$env:PYTHONUNBUFFERED='1'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$runRoot='D:/coco_wire/runs/mask_boundary_route_20260914'
$protocol="$scriptRoot/PROTOCOL_MECHANISM_V6_20260920.json"
$panel="$runRoot/RUN_c70956b46a3d4e769b7982a4d9267afd/panel.json"
$smokeId='RUN_7bfe6b97306e4a56b56c1bd538e628ad'
$fullId='RUN_9fa3374a7519409fb2b09995ce37affb'
foreach($task in @(@{id=$smokeId;limit=3},@{id=$fullId;limit=0})) {
    $runPath="$runRoot/$($task.id)"
    $snapshots=@('--snapshot',"$scriptRoot/threshold_probe.py",'--snapshot',"$scriptRoot/common.py",'--snapshot',$protocol,'--snapshot',$PSCommandPath)
    foreach($name in @('mask_calibration.py','risk_calibration.py','portable_risk.py')) { $snapshots+=@('--snapshot',"$scriptRoot/$name") }
    $inputs=@('--input',$protocol,'--input',$panel,'--input',"$scriptRoot/inputs/response.json",'--input','D:/coco_wire/data/annotations/instances_val2017.json','--input','D:/mdoeldata/pigcv-task05/models/yolo26m-seg.pt','--input',"$runRoot/RUN_2ae67d6556a14848bceb5783a72e5790/predictions_official_zero.json")
    & $python "$scriptRoot/runner.py" --study STUDY_8fb3468ebb704682a2225ebed0e16206 --run-id $task.id --output $runPath --cwd D:/coco_wire @inputs @snapshots --expect "$runPath/SUMMARY.json" --expect "$runPath/threshold_pixels.csv" --expect "$runPath/instances.json" --metrics "$runPath/SUMMARY.json" -- $python -u "$scriptRoot/threshold_probe.py" --protocol $protocol --panel $panel --output $runPath --limit-images $task.limit
    if($LASTEXITCODE -ne 0) { throw "Threshold run failed: $($task.id)" }
}
