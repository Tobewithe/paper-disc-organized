$ErrorActionPreference='Stop'
$root='D:/coco_wire/scripts/boundary_response_20260921'
$python='C:/Users/28358/anaconda3/envs/pytorch/python.exe'
$env:PYTHONPATH='D:/coco_wire/py';$env:PYTHONUNBUFFERED='1';$env:OMP_NUM_THREADS='4';$env:MKL_NUM_THREADS='4'
$protocol="$root/PROTOCOL_BOUNDARY_RESPONSE_RETRY_20260921.json"
$p=Get-Content -LiteralPath $protocol -Raw | ConvertFrom-Json
$runroot='D:/coco_wire/runs/mask_boundary_route_20260914'
$extract="$runroot/$($p.runs.extract)";$fit="$runroot/$($p.runs.fit_select)"
$snapshots=@('--snapshot',$PSCommandPath,'--snapshot',$protocol)
foreach($name in @('extract.py','fit_select.py','local_features.py','common.py','data_access.py','calibrator.py','portable_risk.py','mask_calibration.py')){$snapshots+=@('--snapshot',"$root/$name")}
$inputs=@('--input',$protocol,'--input',$p.weights,'--input',$p.split,'--input',$p.annotations,'--input',"$($p.bank)/candidate_records.csv",'--input',"$($p.bank)/instance_records.csv")
& $python "$root/runner.py" --study $p.study_id --run-id $p.runs.extract --output $extract --cwd D:/coco_wire @inputs @snapshots --expect "$extract/SUMMARY.json" --expect "$extract/features.npz" --metrics "$extract/SUMMARY.json" -- $python -u "$root/extract.py" --protocol $protocol --output $extract
if($LASTEXITCODE -ne 0){throw 'Boundary feature extraction failed'}
& $python "$root/runner.py" --study $p.study_id --run-id $p.runs.fit_select --output $fit --cwd D:/coco_wire --input "$extract/features.npz" --input "$extract/instances.json" --input $p.frozen_rcmc --input $p.split @snapshots --expect "$fit/SUMMARY.json" --expect "$fit/selection_decisions.csv" --metrics "$fit/SUMMARY.json" -- $python -u "$root/fit_select.py" --protocol $protocol --output $fit --features $extract
if($LASTEXITCODE -ne 0){throw 'Boundary pilot fitting failed'}
