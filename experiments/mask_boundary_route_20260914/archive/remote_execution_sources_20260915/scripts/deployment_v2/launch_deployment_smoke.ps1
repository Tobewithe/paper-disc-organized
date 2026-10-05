$ErrorActionPreference = 'Stop'
$pythonExe = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$runId = 'RUN_' + [guid]::NewGuid().ToString('N')
$runDirectory = "D:\coco_wire\runs\mask_boundary_route_20260914\$runId"
$scriptDirectory = 'D:\coco_wire\scripts\deployment_v2'
New-Item -ItemType Directory -Path $runDirectory | Out-Null
$env:PYTHONPATH = 'D:\coco_wire\py'
$env:PYTHONUNBUFFERED = '1'
$runnerArguments = @(
    'D:\coco_wire\scripts\runner.py', '--study', 'STUDY_8fb3468ebb704682a2225ebed0e16206',
    '--run-id', $runId, '--output', $runDirectory, '--cwd', 'D:\coco_wire',
    '--input', 'D:\mdoeldata\pigcv-task05\models\yolo26m-seg.pt',
    '--input', 'D:\coco_wire\data\images\val2017\000000000139.jpg',
    '--snapshot', "$scriptDirectory\mask_calibration.py", '--snapshot', "$scriptDirectory\deployment_smoke.py",
    '--expect', "$runDirectory\SUMMARY.json", '--expect', "$runDirectory\calibrated_preview.jpg",
    '--metrics', "$runDirectory\SUMMARY.json",
    '--', $pythonExe, '-u', "$scriptDirectory\deployment_smoke.py",
    '--package-root', 'D:\coco_wire\py', '--weights', 'D:\mdoeldata\pigcv-task05\models\yolo26m-seg.pt',
    '--source', 'D:\coco_wire\data\images\val2017\000000000139.jpg', '--output', $runDirectory
)
$launched = Start-Process -FilePath $pythonExe -ArgumentList $runnerArguments -WorkingDirectory 'D:\coco_wire' `
    -WindowStyle Hidden -RedirectStandardOutput "$runDirectory\launcher_stdout.log" `
    -RedirectStandardError "$runDirectory\launcher_stderr.log" -PassThru
[pscustomobject]@{RunId=$runId; RecorderPid=$launched.Id; Directory=$runDirectory} | ConvertTo-Json -Compress
