$ErrorActionPreference = 'Stop'
$taskPython = 'C:/Users/28358/anaconda3/envs/pytorch/python.exe'
$taskRoot = 'D:/coco_wire/experiments/native_response_target_comparison_20261009'
$taskRunner = 'D:/coco_wire/experiments/frozen_native_mask_comparison/scripts/runner.py'
$taskOutput = "$taskRoot/runs/RUN_NATIVE_TARGET_FOLLOWUP_PIPELINE_S0_01"
$taskScope = @{ environment='local_laptop_28358lan'; phase='finite_foreground_followup'; detached=$false; source_wait_only_before_terminal=$true;
    stages='source_seal_then_isolated_cost_engineering_then32x3_then_cost_seal'; scientific_retry=$false; no_scoring=$true; max_wait_hours=12 } | ConvertTo-Json -Compress
$taskArgs = @($taskRunner,'--study','STUDY_NATIVE_RESPONSE_TARGET_COMPARISON_20261009','--output',$taskOutput,
    '--run-id','RUN_NATIVE_TARGET_FOLLOWUP_PIPELINE_S0_01','--cwd',$taskRoot,'--scope',$taskScope,
    '--input',"$taskRoot/PROTOCOL.md",'--input',"$taskRoot/COST_PROTOCOL.md",'--input',"$taskRoot/runs/RUN_NATIVE_TARGET_PREFLIGHT_S0_01/EXECUTION_CONFIG.json",
    '--snapshot',"$taskRoot/scripts/run_target_followup.py",'--snapshot',"$taskRoot/scripts/register_target_followup.ps1",'--snapshot',"$taskRoot/scripts/register_target_cost.ps1",
    '--snapshot',"$taskRoot/scripts/benchmark_target_cost.py",'--snapshot',"$taskRoot/scripts/collect_target_runs.py",'--snapshot',"$taskRoot/scripts/seal_extra_runs.py",
    '--expect',"$taskOutput/SUMMARY.json",'--expect',"$taskOutput/COMPLETE.json",'--metrics',"$taskOutput/SUMMARY.json",
    '--',$taskPython,"$taskRoot/scripts/run_target_followup.py",'--config',"$taskRoot/runs/RUN_NATIVE_TARGET_PREFLIGHT_S0_01/EXECUTION_CONFIG.json",
    '--protocol-sha256','cc0d3169e32fc1d42dbd8f1027861ef29fa75f18bb145addd73e6c77d6a94057','--output',$taskOutput,'--max-hours','12')
& $taskPython @taskArgs
exit $LASTEXITCODE
