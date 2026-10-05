$ErrorActionPreference='Stop'
$root='D:/coco_wire/scripts/boundary_response_20260921'
$python='C:/Users/28358/anaconda3/envs/pytorch/python.exe'
$env:PYTHONPATH='D:/coco_wire/py';$env:PYTHONUNBUFFERED='1';$env:OMP_NUM_THREADS='4';$env:MKL_NUM_THREADS='4'
$protocol="$root/PROTOCOL_BOUNDARY_RESPONSE_EVAL_20260921.json"
$p=Get-Content -LiteralPath $protocol -Raw | ConvertFrom-Json
$runs='D:/coco_wire/runs/mask_boundary_route_20260914'
$id='RUN_9631795f20c34be7893a02b94de7cfc2';$output="$runs/$id";$evaluation="$runs/$($p.evaluation.run_id)"
& $python "$root/runner.py" --study $p.study_id --run-id $id --output $output --cwd D:/coco_wire --input "$evaluation/SUMMARY.json" --input "$evaluation/instance_decisions.csv" --input "$($p.evaluation.reference_evaluation)/instance_decisions.csv" --snapshot $protocol --snapshot "$root/analyze.py" --snapshot "$root/common.py" --snapshot $PSCommandPath --expect "$output/SUMMARY.json" --expect "$output/REPORT.md" --metrics "$output/SUMMARY.json" -- $python -u "$root/analyze.py" --protocol $protocol --output $output
if($LASTEXITCODE -ne 0){throw 'Response analysis failed'}
