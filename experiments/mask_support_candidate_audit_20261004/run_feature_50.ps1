$env:PYTHONPATH = 'D:\coco_wire\py'
$env:PYTHONUTF8 = '1'
$env:COCO_SPLIT = 'val2017'
$python = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$out = 'D:\coco_wire\mask_support_candidate_audit_20261004\val_50_v2'
New-Item -ItemType Directory -Force -Path $out | Out-Null
& $python D:\coco_wire\mask_support_audit.py --root D:\coco_wire --out $out --images 50 --start 0 --imgsz 640
