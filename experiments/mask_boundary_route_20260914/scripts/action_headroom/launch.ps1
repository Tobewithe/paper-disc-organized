$ErrorActionPreference='Stop'
$root='D:/coco_wire/scripts/action_headroom_20260921'
$python='C:/Users/28358/anaconda3/envs/pytorch/python.exe'
$env:PYTHONPATH='D:/coco_wire/py';$env:PYTHONUNBUFFERED='1';$env:OMP_NUM_THREADS='4';$env:MKL_NUM_THREADS='4'
$protocol="$root/PROTOCOL_ACTION_HEADROOM_V1_20260921.json"
$p=Get-Content -LiteralPath $protocol -Raw | ConvertFrom-Json
$output="D:/coco_wire/runs/mask_boundary_route_20260914/$($p.run_id)"
& $python "$root/runner.py" --study $p.study_id --run-id $p.run_id --output $output --cwd D:/coco_wire --input $p.decisions --input $p.image_ids --input $p.threshold_pixels --input $protocol --snapshot "$root/headroom.py" --snapshot "$root/common.py" --snapshot $protocol --snapshot $PSCommandPath --expect "$output/SUMMARY.json" --expect "$output/oracle_decisions.csv" --metrics "$output/SUMMARY.json" -- $python -u "$root/headroom.py" --protocol $protocol --output $output
if($LASTEXITCODE -ne 0){throw 'Action headroom analysis failed'}
