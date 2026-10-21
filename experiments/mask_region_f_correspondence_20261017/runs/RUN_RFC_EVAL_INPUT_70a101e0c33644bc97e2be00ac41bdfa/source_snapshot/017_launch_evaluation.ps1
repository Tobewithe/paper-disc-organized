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
if (@(Get-CimInstance Win32_Process -Filter "name='python.exe'").Count -gt 0) { throw 'Existing Python process requires ownership inspection' }
foreach ($rfcStage in @('freeze','input','evaluate')) {
    $rfcLog = Join-Path $rfcStudy ('evaluation_' + $rfcStage + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
    $rfcArgs = @('-B',(Join-Path $rfcStudy 'scripts\rfc_evaluation.py'),'--stage',$rfcStage)
    $rfcProcess = Start-Process -FilePath $rfcPython -ArgumentList $rfcArgs -WorkingDirectory $rfcRoot -WindowStyle Hidden -RedirectStandardOutput ($rfcLog+'.stdout.log') -RedirectStandardError ($rfcLog+'.stderr.log') -PassThru
    $rfcProcess.WaitForExit()
    $rfcProcess.Refresh()
    if ($rfcProcess.ExitCode -ne 0) { throw ('Evaluation stage failed: '+$rfcStage+' exit '+$rfcProcess.ExitCode) }
}
