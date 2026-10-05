$ErrorActionPreference='Stop'
$root='D:/coco_wire/scripts/rcmc_mechanism_20260920'
$runs='D:/coco_wire/runs/mask_boundary_route_20260914'
$python='C:/Users/28358/anaconda3/envs/pytorch/python.exe'
$env:PYTHONPATH='D:/coco_wire/py';$env:PYTHONUNBUFFERED='1';$env:OMP_NUM_THREADS='4';$env:MKL_NUM_THREADS='4'
$id='RUN_2ed665c5a7194fc9a564a9709c464101';$output="$runs/$id";$protocol="$root/PROTOCOL_MECHANISM_V6_20260920.json"
$inputDir="$runs/RUN_9fa3374a7519409fb2b09995ce37affb"
$snapshots=@('--snapshot',"$root/threshold_analysis.py",'--snapshot',"$root/common.py",'--snapshot',$PSCommandPath,'--snapshot',$protocol)
$inputs=@('--input',$protocol,'--input',"$inputDir/instances.json",'--input',"$inputDir/threshold_pixels.csv",'--input',"$inputDir/SUMMARY.json")
& $python "$root/runner.py" --study STUDY_8fb3468ebb704682a2225ebed0e16206 --run-id $id --output $output --cwd D:/coco_wire @inputs @snapshots --expect "$output/SUMMARY.json" --expect "$output/threshold_diagnostics.png" --metrics "$output/SUMMARY.json" -- $python -u "$root/threshold_analysis.py" --protocol $protocol --output $output --input $inputDir
if($LASTEXITCODE -ne 0){throw 'Threshold analysis failed'}
