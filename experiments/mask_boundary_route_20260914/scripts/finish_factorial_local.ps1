param([Parameter(Mandatory=$true)][string]$ScoringRunId)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
$studyRoot=Split-Path $PSScriptRoot -Parent
$runsRoot=Join-Path $studyRoot 'runs'
$scoringRun=Join-Path $runsRoot $ScoringRunId
$sourceRun=Join-Path $runsRoot 'RUN_2ae67d6556a14848bceb5783a72e5790'
$deadline=(Get-Date).AddMinutes(30)
while ($true) {
    $record=Get-Content -LiteralPath (Join-Path $scoringRun 'run.json') -Raw | ConvertFrom-Json
    if ($record.status -eq 'completed') { break }
    if ($record.status -in @('failed','cancelled','interrupted')) { throw 'Scoring recovery failed.' }
    if ((Get-Date) -gt $deadline) { throw 'Scoring recovery timed out.' }
    Start-Sleep -Seconds 10
}
$pythonExe='C:\Dpan\envsfiles\CondaData\envs\pytorch\python.exe'
$runner=Join-Path $projectRoot 'shared\tools\research_runner\runner.py'
$annotations=Join-Path $projectRoot 'assets\datasets\coco\annotations\instances_val2017.json'
$summary=Join-Path $scoringRun 'SUMMARY.json'
$specs=@(
    @{id='RUN_97df28c1ff3f44de84b1cff6f5daa128';script='summarize_decoder_controls.py'},
    @{id='RUN_cca418767c2240d0acadcdbbb3adbf48';script='analyze_repair_damage_objects.py'}
)
foreach ($spec in $specs) {
    $outputPath=Join-Path $runsRoot $spec.id
    $scriptPath=Join-Path $PSScriptRoot $spec.script
    $arguments=@($runner,'--study','STUDY_8fb3468ebb704682a2225ebed0e16206','--run-id',$spec.id,
        '--output',$outputPath,'--cwd',$projectRoot,'--input',$summary,
        '--input',(Join-Path $sourceRun 'instance_records.csv'),'--input',(Join-Path $sourceRun 'candidate_records.csv'),
        '--input',$annotations,'--snapshot',$scriptPath,'--snapshot',$PSCommandPath,
        '--expect',(Join-Path $outputPath 'ANALYSIS.json'),'--metrics',(Join-Path $outputPath 'ANALYSIS.json'),
        '--',$pythonExe,'-u',$scriptPath,'--input',$sourceRun,'--summary',$summary,'--annotations',$annotations,'--output',$outputPath)
    if ($spec.script -eq 'analyze_repair_damage_objects.py') {
        $arguments+=@('--geometry-cache',(Join-Path $runsRoot 'RUN_97dbbbcb29754a04a314516ac868d684\objects.csv'))
    }
    & 'C:\Dpan\envsfiles\miniconda3\Scripts\conda.exe' run --no-capture-output -n pytorch python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Analysis failed: $($spec.id)" }
}
