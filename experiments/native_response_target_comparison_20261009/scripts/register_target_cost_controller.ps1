$ErrorActionPreference = 'Stop'
$taskPython = 'C:/Users/28358/anaconda3/envs/pytorch/python.exe'
$taskRoot = 'D:/coco_wire/experiments/native_response_target_comparison_20261009'
$taskRunner = 'D:/coco_wire/experiments/frozen_native_mask_comparison/scripts/runner.py'
$taskOutput = "$taskRoot/runs/RUN_NATIVE_TARGET_COST_PIPELINE_S0_01"
$taskScope = @{ environment='local_laptop_28358lan'; phase='finite_isolated_cost_controller'; wmi_created_hidden=$true;
    no_gt=$true; serial_fresh_children=$true; launcher_return_independent=$true } | ConvertTo-Json -Compress
$taskArgs = @($taskRunner,'--study','STUDY_NATIVE_RESPONSE_TARGET_COMPARISON_20261009','--output',$taskOutput,
    '--run-id','RUN_NATIVE_TARGET_COST_PIPELINE_S0_01','--cwd',$taskRoot,'--scope',$taskScope,
    '--input',"$taskRoot/COST_PROTOCOL.md",'--snapshot',"$taskRoot/scripts/register_target_cost.ps1",
    '--snapshot',"$taskRoot/scripts/register_target_cost_controller.ps1",'--snapshot',"$taskRoot/scripts/launch_wmi_hidden.ps1",'--snapshot',$taskRunner,
    '--expect',"$taskRoot/runs/RUN_NATIVE_TARGET_COST_ENGINEERING_S0_01/COST_COMPLETE.json",
    '--expect',"$taskRoot/runs/RUN_NATIVE_TARGET_COST_PANEL_S0_01/COST_COMPLETE.json",
    '--','C:/Program Files/PowerShell/7/pwsh.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',"$taskRoot/scripts/register_target_cost.ps1")
& $taskPython @taskArgs
exit $LASTEXITCODE
