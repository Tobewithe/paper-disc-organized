$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$rfcRoot = 'D:\coco_wire\baseline_native_20261013'
$rfcStudy = Join-Path $rfcRoot 'experiments\mask_region_f_correspondence_20261017'
$rfcPython = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$env:PYTHONPATH = Join-Path $rfcRoot '_site'
$env:YOLO_OFFLINE = 'true'
$env:YOLO_AUTOINSTALL = 'false'
$env:YOLO_CONFIG_DIR = Join-Path $rfcRoot '_ultralytics_config'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
if ($env:CUDA_VISIBLE_DEVICES -or $env:MKL_NUM_THREADS) { throw 'Unexpected numeric environment variables' }
if (Test-Path -LiteralPath (Join-Path $rfcStudy '.sequence.lock')) { throw 'Inspect existing RFC sequence first' }
if (@(Get-CimInstance Win32_Process -Filter "name='python.exe'").Count -gt 0) { throw 'Existing Python process requires ownership inspection' }
$rfcStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$rfcLog = Join-Path $rfcStudy ('launch_' + $rfcStamp)
$rfcArguments = @('-B', (Join-Path $rfcStudy 'scripts\rfc_sequence.py'))
$rfcProcess = Start-Process -FilePath $rfcPython -ArgumentList $rfcArguments -WorkingDirectory $rfcRoot -WindowStyle Hidden -RedirectStandardOutput ($rfcLog+'.stdout.log') -RedirectStandardError ($rfcLog+'.stderr.log') -PassThru
$rfcReceipt = [ordered]@{pid=$rfcProcess.Id;launched_at=(Get-Date -Format o);stdout=($rfcLog+'.stdout.log');stderr=($rfcLog+'.stderr.log');interpreter=$rfcPython;arguments=$rfcArguments;scheduled_task=$false;finite_sequence=$true}
$rfcReceipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath ($rfcLog+'.json') -Encoding UTF8
$rfcReceipt | ConvertTo-Json -Depth 6 -Compress
