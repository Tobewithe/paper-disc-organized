$ErrorActionPreference='Stop'
$root='D:/coco_wire/scripts/boundary_response_20260921'
$python='C:/Users/28358/anaconda3/envs/pytorch/python.exe'
$env:PYTHONPATH='D:/coco_wire/py';$env:PYTHONUNBUFFERED='1';$env:OMP_NUM_THREADS='4';$env:MKL_NUM_THREADS='4'
$protocol="$root/PROTOCOL_BOUNDARY_RESPONSE_EVAL_20260921.json"
$p=Get-Content -LiteralPath $protocol -Raw | ConvertFrom-Json
$runs='D:/coco_wire/runs/mask_boundary_route_20260914';$output="$runs/$($p.evaluation.run_id)"
$snapshots=@('--snapshot',$PSCommandPath,'--snapshot',$protocol)
foreach($name in @('evaluate.py','local_features.py','common.py','data_access.py','calibrator.py','portable_risk.py','mask_calibration.py')){$snapshots+=@('--snapshot',"$root/$name")}
$inputs=@('--input',$p.weights,'--input',$p.evaluation.annotations,'--input',"$($p.evaluation.bank)/candidate_records.csv",'--input',"$($p.evaluation.bank)/instance_records.csv",'--input',"$($p.evaluation.bank)/predictions_official_zero.json",'--input',"$runs/$($p.evaluation.model_run)/SUMMARY.json",'--input',"$runs/$($p.evaluation.model_run)/single_local.json",'--input',"$runs/$($p.evaluation.model_run)/multi_local.json")
& $python "$root/runner.py" --study $p.study_id --run-id $p.evaluation.run_id --output $output --cwd D:/coco_wire @inputs @snapshots --expect "$output/SUMMARY.json" --expect "$output/instance_decisions.csv" --metrics "$output/SUMMARY.json" -- $python -u "$root/evaluate.py" --protocol $protocol --output $output
if($LASTEXITCODE -ne 0){throw 'Frozen response evaluation failed'}
