$ErrorActionPreference = 'Stop'
$segStudy = 'C:/Dpan/codexproject/paper-disc-organized/experiments/segrefiner_official_content_baseline_20261009'
$segPython = 'C:/Dpan/envsfiles/CondaData/envs/pytorch/python.exe'
$segRun = 'RUN_SEGREFINER_BATCH8_FIXED5K_INFERENCE_S0_02'
$segOutput = $segStudy + '/runs/' + $segRun
$segEngineering = $segStudy + '/runs/RUN_SEGREFINER_BATCH8_ENGINEERING32_S0_01'
$segEngineeringRecord = Get-Content -LiteralPath ($segEngineering + '/run.json') -Raw | ConvertFrom-Json
$segEngineeringSummary = Get-Content -LiteralPath ($segEngineering + '/SUMMARY.json') -Raw | ConvertFrom-Json
$segDecision = Get-Content -LiteralPath ($segEngineering + '/RESOURCE_FEASIBILITY.json') -Raw | ConvertFrom-Json
if ($segEngineeringRecord.status -ne 'completed' -or $segEngineeringRecord.return_code -ne 0 -or $segEngineeringRecord.artifact_completeness -ne 'complete' -or $segEngineeringSummary.engineering_passed -ne $true -or $segEngineeringSummary.hardware_batch_max -ne 8 -or $segDecision.execution_feasible -ne $true) { throw 'Actual completed batch8 engineering and resource decision required' }
$segLockedSources = @{
    'segrefiner_runtime.py' = '5a296b083991adb6918db09a9f69204a0a2e4f65b582d5c9a2f2335d4183c107'
    'run_segrefiner_inference.py' = '7f550762d2285453098e2d99b9b3211cdc0e2f9c6cb7e233307d85445d855445'
    'segrefiner_batch8_runtime.py' = '274b6b3b84813b81737cdb760a89aadad11b08469397205f4da5f7513b26f737'
    'run_segrefiner_batch8.py' = 'c6a799f67b991a0aa6c263d52239093b3462187f7f10718157bc75a141416528'
    'wddm_memory_probe.py' = 'acb110856d89842d28e2b79ffdf20d5f769b4c3372e7bcc753b110e518a8c69f'
}
foreach ($segEntry in $segLockedSources.GetEnumerator()) {
    $segActualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath ($segStudy + '/scripts/' + $segEntry.Key)).Hash.ToLowerInvariant()
    if ($segActualHash -ne $segEntry.Value) { throw ('Locked source differs: ' + $segEntry.Key) }
}
$segArguments = @('C:/Dpan/codexproject/paper-disc-organized/shared/tools/research_runner/runner.py', '--study', 'STUDY_SEGREFINER_OFFICIAL_CONTENT_BASELINE_20261009', '--output', $segOutput, '--run-id', $segRun, '--cwd', $segStudy,
    '--input', ($segStudy + '/assets/models/segrefiner_lr_latest.pth'),
    '--input', ($segStudy + '/runs/RUN_SEGREFINER_RGB_CACHE_PREPARATION_20261009_01/IMAGE_LIST.txt'),
    '--input', ($segStudy + '/runs/RUN_SEGREFINER_RGB_CACHE_PREPARATION_20261009_01/IMAGE_META.json'),
    '--input', ($segStudy + '/runs/RUN_SEGREFINER_BATCH8_GPU_VERIFY_S0_01/VERIFICATION.json'),
    '--input', ($segEngineering + '/SUMMARY.json'), '--input', ($segEngineering + '/RESOURCE_FEASIBILITY.json'))
foreach ($segName in $segLockedSources.Keys) { $segArguments += @('--snapshot', ($segStudy + '/scripts/' + $segName)) }
foreach ($segName in @('PROTOCOL.md', 'EXECUTION_SUPPLEMENT.md', 'EXECUTION_BATCH8_ADAPTATION.md')) { $segArguments += @('--snapshot', ($segStudy + '/' + $segName)) }
$segArguments += @('--snapshot', $PSCommandPath, '--expect', ($segOutput + '/SUMMARY.json'), '--expect', ($segOutput + '/INFERENCE_COMPLETE.json'), '--metrics', ($segOutput + '/SUMMARY.json'),
    '--scope', '{"type":"gtfree_fixed5000_batch8_officialLR_inference","images":5000,"native_supported_first64":true,"no_yolo":true,"no_gt":true,"no_quality_selection":true,"single_fixed_hardware_adaptation":true}',
    '--', $segPython, ($segStudy + '/scripts/run_segrefiner_batch8.py'), '--root', $segStudy, '--run-id', $segRun,
    '--upstream', ($segStudy + '/assets/upstream/SegRefiner-53419a2d38ea3da0b6e2be77e5b45e139195a0b3'), '--checkpoint', ($segStudy + '/assets/models/segrefiner_lr_latest.pth'),
    '--native-run', 'C:/Dpan/codexproject/paper-disc-organized/experiments/frozen_native_mask_comparison/runs/RUN_FROZEN_NATIVE_5K_INFERENCE_S0_02',
    '--images-list', ($segStudy + '/runs/RUN_SEGREFINER_RGB_CACHE_PREPARATION_20261009_01/IMAGE_LIST.txt'), '--image-meta', ($segStudy + '/runs/RUN_SEGREFINER_RGB_CACHE_PREPARATION_20261009_01/IMAGE_META.json'),
    '--runtime-verification', ($segStudy + '/runs/RUN_SEGREFINER_BATCH8_GPU_VERIFY_S0_01/VERIFICATION.json'), '--engineering-receipt', ($segEngineering + '/SUMMARY.json'))
& $segPython @segArguments
exit $LASTEXITCODE
