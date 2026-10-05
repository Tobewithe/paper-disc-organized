param()
$ErrorActionPreference = "Stop"
$py = "C:\Users\28358\anaconda3\envs\pytorch\python.exe"
$base = "D:\coco_wire\research\mask_quality_boundary_branch_20261004"
$runner = Join-Path $base "runner.py"
$fixed = Join-Path $base "scripts\run_fixed_runtime.py"
$eval = Join-Path $base "scripts\eval_standard_val.py"
$yaml = Join-Path $base "data\coco_val5k.yaml"
$weights = "D:\coco_wire\research\mask_quality_boundary_branch_20261004\runs\boundary_full_available_e3_s0_v2\weights\best.pt"
$run = "RUN_eval_standard_val5k_method"
$out = Join-Path $base ("runs\" + $run)
$project = Join-Path $base "runs"
& $py $runner --study mask_quality_boundary_branch_20261004 --output $out --cwd $base --input $yaml --snapshot "scripts\eval_standard_val.py" --snapshot "scripts\run_fixed_runtime.py" --snapshot "runner.py" --expect "runs\standard_val5k_method\results.csv" -- $py $fixed $eval --weights $weights --data $yaml --project $project --name standard_val5k_method --method
exit $LASTEXITCODE

