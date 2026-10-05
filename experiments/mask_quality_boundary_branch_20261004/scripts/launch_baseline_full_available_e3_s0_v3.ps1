param()
$ErrorActionPreference = "Stop"
$py = "C:\Users\28358\anaconda3\envs\pytorch\python.exe"
$base = "D:\coco_wire\research\mask_quality_score_training_20261004"
$runner = "D:\coco_wire\research\mask_quality_boundary_branch_20261004\runner.py"
$fixed = "D:\coco_wire\research\mask_quality_boundary_branch_20261004\scripts\run_fixed_runtime.py"
$train = Join-Path $base "scripts\train_official_baseline.py"
$yaml = Join-Path $base "coco_full_available.yaml"
$run = "RUN_baseline_full_available_e3_s0_v3"
$out = "D:\coco_wire\research\mask_quality_boundary_branch_20261004\runs\$run"
$project = Join-Path $base "runs"
& $py $runner --study mask_quality_boundary_branch_20261004 --output $out --cwd "D:\coco_wire\research\mask_quality_boundary_branch_20261004" --input $yaml --snapshot $train --snapshot $fixed --snapshot $runner --expect "D:\coco_wire\research\mask_quality_score_training_20261004\runs\baseline_full_available_e3_s0_v3\results.csv" -- $py $fixed $train --weights "D:\coco_wire\models\yolo26m-seg.pt" --data $yaml --project $project --name baseline_full_available_e3_s0_v3 --epochs 3 --batch 2 --imgsz 640
exit $LASTEXITCODE



