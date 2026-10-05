# Finite dependent postprocessing for the already-running confirmation evaluation.
# No parameters are fitted and no GPU job is started here.
$ErrorActionPreference = 'Stop'
$sourceRun = 'D:\coco_wire\runs\mask_boundary_route_20260914\RUN_8fe82201101b4e9d941cb07a3496c110'
$analysisId = 'RUN_a175c8a5120f49278a183d43965d795e'
$analysisRun = "D:\coco_wire\runs\mask_boundary_route_20260914\$analysisId"
$deadline = (Get-Date).AddHours(3)
while ($true) {
    $status = Get-Content -LiteralPath "$sourceRun\run.json" -Raw | ConvertFrom-Json
    if ($status.status -eq 'completed') {
        if ($status.return_code -ne 0) { throw 'Evaluation did not return zero.' }
        break
    }
    if ($status.status -in @('failed','cancelled','interrupted')) { throw "Evaluation stopped: $($status.status)" }
    if ((Get-Date) -gt $deadline) { throw 'Timed out waiting for the evaluation; no analysis was fabricated.' }
    Start-Sleep -Seconds 10
}
$pythonExe = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
& $pythonExe 'D:\coco_wire\scripts\runner.py' --study STUDY_8fb3468ebb704682a2225ebed0e16206 `
    --run-id $analysisId --output $analysisRun --cwd D:\coco_wire `
    --input "$sourceRun\SUMMARY.json" --input "$sourceRun\image_ids.json" --input "$sourceRun\instance_records.csv" `
    --input D:\coco_wire\data\annotations\instances_val2017.json `
    --snapshot D:\coco_wire\scripts\summarize_decoder_controls.py --snapshot $PSCommandPath `
    --expect "$analysisRun\ANALYSIS.json" --expect "$analysisRun\decoder_controls.png" `
    --metrics "$analysisRun\ANALYSIS.json" -- $pythonExe D:\coco_wire\scripts\summarize_decoder_controls.py `
    --input $sourceRun --annotations D:\coco_wire\data\annotations\instances_val2017.json --output $analysisRun
exit $LASTEXITCODE
