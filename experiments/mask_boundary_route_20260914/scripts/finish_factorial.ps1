# Finite postprocessing of one named evaluation; no retries or additional training.
param(
    [Parameter(Mandatory=$true)][string]$SourceRunId,
    [Parameter(Mandatory=$true)][string]$StatisticsRunId,
    [Parameter(Mandatory=$true)][string]$ObjectsRunId,
    [string]$ScriptRoot = 'D:\coco_wire\scripts\factorial_v3_20260915'
)
$ErrorActionPreference = 'Stop'
$runsRoot = 'D:\coco_wire\runs\mask_boundary_route_20260914'
$sourceRun = Join-Path $runsRoot $SourceRunId
$pythonExe = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$annotations = 'D:\coco_wire\data\annotations\instances_val2017.json'
$deadline = (Get-Date).AddHours(3)
while ($true) {
    $status = Get-Content -LiteralPath (Join-Path $sourceRun 'run.json') -Raw | ConvertFrom-Json
    if ($status.status -eq 'completed') {
        if ($status.return_code -ne 0 -or $status.artifact_completeness -ne 'complete') { throw 'Evaluation incomplete.' }
        break
    }
    if ($status.status -in @('failed','cancelled','interrupted')) { throw "Evaluation stopped: $($status.status)" }
    if ((Get-Date) -gt $deadline) { throw 'Timed out; dependent results were not fabricated.' }
    Start-Sleep -Seconds 30
}
$statisticsRun = Join-Path $runsRoot $StatisticsRunId
$objectsRun = Join-Path $runsRoot $ObjectsRunId
$statisticsScript = Join-Path $ScriptRoot 'summarize_decoder_controls.py'
$objectsScript = Join-Path $ScriptRoot 'analyze_repair_damage_objects.py'
& $pythonExe 'D:\coco_wire\scripts\runner.py' --study STUDY_8fb3468ebb704682a2225ebed0e16206 `
    --run-id $StatisticsRunId --output $statisticsRun --cwd D:\coco_wire `
    --input "$sourceRun\SUMMARY.json" --input "$sourceRun\instance_records.csv" --input $annotations `
    --snapshot $statisticsScript --snapshot $PSCommandPath `
    --expect "$statisticsRun\ANALYSIS.json" --expect "$statisticsRun\decoder_controls.png" `
    --metrics "$statisticsRun\ANALYSIS.json" -- $pythonExe $statisticsScript `
    --input $sourceRun --annotations $annotations --output $statisticsRun
$statisticsExit = $LASTEXITCODE
& $pythonExe 'D:\coco_wire\scripts\runner.py' --study STUDY_8fb3468ebb704682a2225ebed0e16206 `
    --run-id $ObjectsRunId --output $objectsRun --cwd D:\coco_wire `
    --input "$sourceRun\SUMMARY.json" --input "$sourceRun\instance_records.csv" --input "$sourceRun\candidate_records.csv" --input $annotations `
    --snapshot $objectsScript --snapshot $PSCommandPath `
    --expect "$objectsRun\ANALYSIS.json" --expect "$objectsRun\objects.csv" `
    --metrics "$objectsRun\ANALYSIS.json" -- $pythonExe $objectsScript `
    --input $sourceRun --annotations $annotations --output $objectsRun
if ($statisticsExit -ne 0 -or $LASTEXITCODE -ne 0) { throw 'One or more postprocessing runs failed; see individual logs.' }
