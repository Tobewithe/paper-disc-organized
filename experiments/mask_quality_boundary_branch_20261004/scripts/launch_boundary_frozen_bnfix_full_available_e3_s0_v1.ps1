param()
$ErrorActionPreference="Stop"
$py="C:\Users\28358\anaconda3\envs\pytorch\python.exe"
$base="D:\coco_wire\research\mask_quality_boundary_branch_20261004"
$runner=Join-Path $base "runner.py"; $fixed=Join-Path $base "scripts\run_fixed_runtime.py"; $train=Join-Path $base "scripts\train_boundary_quality.py"
$yaml="D:\coco_wire\research\mask_quality_score_training_20261004\coco_full_available.yaml"
$run="RUN_boundary_frozen_bnfix_full_available_e3_s0_v1"; $out=Join-Path $base ("runs\"+$run); $project=Join-Path $base "runs"
$expected=Join-Path $base "runs\boundary_frozen_bnfix_full_available_e3_s0_v1\results.csv"
& $py $runner --study mask_quality_boundary_branch_20261004 --output $out --cwd $base --input $yaml --snapshot "scripts\train_boundary_quality.py" --snapshot "scripts\run_fixed_runtime.py" --snapshot "runner.py" --expect $expected -- $py $fixed $train --weights "D:\coco_wire\models\yolo26m-seg.pt" --data $yaml --project $project --name boundary_frozen_bnfix_full_available_e3_s0_v1 --epochs 3 --batch 2 --imgsz 640 --quality-weight 0.1 --freeze-original
exit $LASTEXITCODE
