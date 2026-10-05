param()
$ErrorActionPreference="Stop"
$py="C:\Users\28358\anaconda3\envs\pytorch\python.exe"
$base="D:\coco_wire\research\mask_quality_boundary_branch_20261004"
$runner=Join-Path $base "runner.py"; $fixed=Join-Path $base "scripts\run_fixed_runtime.py"; $eval=Join-Path $base "scripts\eval_protocol_compare.py"
$yaml=Join-Path $base "data\coco_val5k.yaml"; $weights="D:\coco_wire\research\mask_quality_score_training_20261004\runs\baseline_full_available_e3_s0_v3\weights\best.pt"
$run="RUN_protocol_baseline_native_v2"; $out=Join-Path $base ("runs\"+$run); $project=Join-Path $base "runs"; $metrics=Join-Path $base "runs\protocol_baseline_native_v2_metrics.json"
& $py $runner --study mask_quality_boundary_branch_20261004 --output $out --cwd $base --input $yaml --snapshot "scripts\eval_protocol_compare.py" --snapshot "scripts\run_fixed_runtime.py" --snapshot "runner.py" --expect $metrics -- $py $fixed $eval --weights $weights --data $yaml --project $project --name protocol_baseline_native_v2 --metrics-out $metrics --mode default --native
exit $LASTEXITCODE



