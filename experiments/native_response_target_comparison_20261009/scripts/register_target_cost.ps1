$ErrorActionPreference = 'Stop'
$taskPython = 'C:/Users/28358/anaconda3/envs/pytorch/python.exe'
$taskRoot = 'D:/coco_wire/experiments/native_response_target_comparison_20261009'
$taskRunner = 'D:/coco_wire/experiments/frozen_native_mask_comparison/scripts/runner.py'
$taskProtocolSha = 'cc0d3169e32fc1d42dbd8f1027861ef29fa75f18bb145addd73e6c77d6a94057'
$taskCostProtocolSha = 'f3c22f25334316c429027df81e6250fe874dc44daf9a08f7ab8da365b71c11f1'
$taskConfig = "$taskRoot/runs/RUN_NATIVE_TARGET_PREFLIGHT_S0_01/EXECUTION_CONFIG.json"
$taskReference = "$taskRoot/runs/RUN_NATIVE_TARGET_FORMAL_INFERENCE_S0_01"
$taskEngineering = "$taskRoot/runs/RUN_NATIVE_TARGET_COST_ENGINEERING_S0_01"
$taskPanel = "$taskRoot/runs/RUN_NATIVE_TARGET_COST_PANEL_S0_01"
foreach ($taskPhase in @('ENGINEERING','PANEL')) {
    $taskOutput = if ($taskPhase -eq 'ENGINEERING') { $taskEngineering } else { $taskPanel }
    $taskScope = @{ environment='local_laptop_28358lan'; phase='isolated_native_target_cost'; engineering=($taskPhase -eq 'ENGINEERING');
        model_profile='actual_new20k_numeric_HGB_only'; no_gt=$true; no_ap=$true; serial_fresh_children=$true;
        protocol_sha256=$taskProtocolSha; cost_protocol_sha256=$taskCostProtocolSha } | ConvertTo-Json -Compress
    $taskArgs = @($taskRunner,'--study','STUDY_NATIVE_RESPONSE_TARGET_COMPARISON_20261009','--output',$taskOutput,
        '--run-id',"RUN_NATIVE_TARGET_COST_${taskPhase}_S0_01",'--cwd',$taskRoot,'--scope',$taskScope,
        '--input',"$taskRoot/PROTOCOL.md",'--input',"$taskRoot/COST_PROTOCOL.md",'--input',$taskConfig,
        '--input',"$taskReference/SUMMARY.json",'--input',"$taskReference/COMPLETE.json",
        '--snapshot',"$taskRoot/scripts/benchmark_target_cost.py",'--snapshot',"$taskRoot/scripts/target_common.py",'--snapshot',"$taskRoot/scripts/register_target_cost.ps1",
        '--snapshot','D:/coco_wire/experiments/frozen_native_mask_comparison/scripts/benchmark_individual_cost.py','--snapshot',$taskRunner,
        '--expect',"$taskOutput/COST_COMPLETE.json",'--expect',"$taskOutput/COST_SAMPLES.json",'--expect',"$taskOutput/SUMMARY.json",'--expect',"$taskOutput/RESOURCE_BEFORE.json",'--metrics',"$taskOutput/SUMMARY.json",
        '--',$taskPython,"$taskRoot/scripts/benchmark_target_cost.py",'--config',$taskConfig,'--protocol-sha256',$taskProtocolSha,
        '--cost-protocol',"$taskRoot/COST_PROTOCOL.md",'--cost-protocol-sha256',$taskCostProtocolSha,
        '--reference',$taskReference,'--output',$taskOutput)
    if ($taskPhase -eq 'ENGINEERING') { $taskArgs += '--engineering' }
    else { $taskArgs += @('--engineering-receipt',"$taskEngineering/SUMMARY.json") }
    & $taskPython @taskArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
exit 0
