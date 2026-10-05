$ErrorActionPreference='Stop'
$scriptRoot='D:/coco_wire/scripts/rcmc_mechanism_20260920'
$python='C:/Users/28358/anaconda3/envs/pytorch/python.exe'
$env:PYTHONPATH='D:/coco_wire/py'
$env:PYTHONUNBUFFERED='1'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$runRoot='D:/coco_wire/runs/mask_boundary_route_20260914'
$protocol="$scriptRoot/PROTOCOL_MECHANISM_V6_20260920.json"
$smokeId='RUN_7fb7971e45a64569a94ed6f194a89c2c'
$fullId='RUN_c70956b46a3d4e769b7982a4d9267afd'
foreach($task in @(@{id=$smokeId;limit=8},@{id=$fullId;limit=0})) {
    $runPath="$runRoot/$($task.id)"
    $snapshots=@('--snapshot',"$scriptRoot/spatial.py",'--snapshot',"$scriptRoot/common.py",'--snapshot',$protocol,'--snapshot',$PSCommandPath)
    $inputs=@('--input',$protocol,'--input',"$scriptRoot/inputs/decisions.csv",'--input','D:/coco_wire/data/annotations/instances_val2017.json')
    foreach($name in @('candidate_records.csv','instance_records.csv','image_ids.json','predictions_official_zero.json','predictions_smooth_gated.json')) {
        $inputs+=@('--input',"$runRoot/RUN_2ae67d6556a14848bceb5783a72e5790/$name")
    }
    & $python "$scriptRoot/runner.py" --study STUDY_8fb3468ebb704682a2225ebed0e16206 --run-id $task.id --output $runPath --cwd D:/coco_wire @inputs @snapshots --expect "$runPath/SUMMARY.json" --expect "$runPath/spatial_instances.csv" --expect "$runPath/panel.json" --metrics "$runPath/SUMMARY.json" -- $python -u "$scriptRoot/spatial.py" --protocol $protocol --output $runPath --limit-images $task.limit
    if($LASTEXITCODE -ne 0) { throw "Spatial run failed: $($task.id)" }
}
