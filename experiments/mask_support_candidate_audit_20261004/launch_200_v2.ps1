$env:PYTHONPATH = 'D:\coco_wire\py'
$env:PYTHONUTF8 = '1'
$python = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$out = 'D:\coco_wire\mask_support_candidate_audit_20261004\run_200_v2'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$p = Start-Process -FilePath $python -ArgumentList @('D:\coco_wire\mask_support_audit.py','--root','D:\coco_wire','--out',$out,'--images','200','--start','0','--imgsz','640') -WorkingDirectory 'D:\coco_wire' -WindowStyle Hidden -RedirectStandardOutput "$out\stdout.log" -RedirectStandardError "$out\stderr.log" -PassThru
$p.Id | Set-Content "$out\pid.txt"
Write-Output "PID=$($p.Id)"
