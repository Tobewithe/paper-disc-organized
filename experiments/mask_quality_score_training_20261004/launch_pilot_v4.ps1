$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'D:\coco_wire\py'
$py = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$base = 'D:\coco_wire\research\mask_quality_score_training_20261004'
& $py "$base\runner.py" --study mask_quality_score_training_20261004 --output "$base\runs\RUN_quality_100_e3_v4" --cwd $base --snapshot "$base\scripts\train_quality_smoke.py" --snapshot "$base\coco_100.yaml" --input "$base\coco_100.yaml" --expect "$base\runs\quality_100_e3_v4\results.csv" -- $py "$base\scripts\train_quality_smoke.py" --weights D:\coco_wire\models\yolo26m-seg.pt --data "$base\coco_100.yaml" --project "$base\runs" --name quality_100_e3_v4 --epochs 3 --batch 2 --imgsz 640
exit $LASTEXITCODE
