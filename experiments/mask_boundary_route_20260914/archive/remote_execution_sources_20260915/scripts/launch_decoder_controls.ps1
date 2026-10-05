param(
    [string]$RunId = 'RUN_63ca40e6dc0748e6b95e92f5f272c0d1',
    [int]$Start = 0,
    [int]$Limit = 500,
    [string]$Variants = 'official_zero,official_gated,official_global,official_erode,smooth_gated,input_area_gated,legacy_zero,legacy_gated'
)
$ErrorActionPreference = 'Stop'
$pythonExe = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$studyRoot = 'D:\coco_wire'
$runDirectory = Join-Path $studyRoot "runs\mask_boundary_route_20260914\$RunId"
if (Test-Path (Join-Path $runDirectory 'run.json')) { throw 'Existing Run; choose another ID.' }
New-Item -ItemType Directory -Force -Path $runDirectory | Out-Null
$env:PYTHONPATH = 'D:\coco_wire\py'
$env:PYTHONUNBUFFERED = '1'
$runnerArguments = @(
    'D:\coco_wire\scripts\runner.py', '--study', 'STUDY_8fb3468ebb704682a2225ebed0e16206',
    '--run-id', $RunId, '--output', $runDirectory, '--cwd', $studyRoot,
    '--input', 'D:\coco_wire\data\annotations\instances_val2017.json',
    '--input', 'D:\mdoeldata\pigcv-task05\models\yolo26m-seg.pt',
    '--snapshot', 'D:\coco_wire\scripts\mask_calibration.py',
    '--snapshot', 'D:\coco_wire\scripts\decoder_calibration_experiment.py',
    '--snapshot', 'D:\coco_wire\scripts\PROTOCOL_DECODER_V2.json',
    '--snapshot', 'D:\coco_wire\py\ultralytics\utils\ops.py',
    '--snapshot', 'D:\coco_wire\py\ultralytics\models\yolo\segment\predict.py',
    '--snapshot', 'D:\coco_wire\py\ultralytics\models\yolo\segment\val.py',
    '--expect', (Join-Path $runDirectory 'SUMMARY.json'),
    '--expect', (Join-Path $runDirectory 'instance_records.csv'),
    '--metrics', (Join-Path $runDirectory 'SUMMARY.json'),
    '--', $pythonExe, '-u', 'D:\coco_wire\scripts\decoder_calibration_experiment.py',
    '--package-root', 'D:\coco_wire\py', '--images', 'D:\coco_wire\data\images\val2017',
    '--annotations', 'D:\coco_wire\data\annotations\instances_val2017.json',
    '--weights', 'D:\mdoeldata\pigcv-task05\models\yolo26m-seg.pt',
    '--protocol', 'D:\coco_wire\scripts\PROTOCOL_DECODER_V2.json',
    '--output', $runDirectory, '--branch', 'one2one', '--start', $Start, '--limit', $Limit,
    '--variants', $Variants
)
$launched = Start-Process -FilePath $pythonExe -ArgumentList $runnerArguments -WorkingDirectory $studyRoot `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runDirectory 'launcher_stdout.log') `
    -RedirectStandardError (Join-Path $runDirectory 'launcher_stderr.log') -PassThru
[pscustomobject]@{RunId=$RunId; RecorderPid=$launched.Id; Directory=$runDirectory} | ConvertTo-Json -Compress
