$ErrorActionPreference = 'Stop'
$taskPython = 'C:/Users/28358/anaconda3/envs/pytorch/python.exe'
$taskRoot = 'D:/coco_wire/experiments/native_response_target_comparison_20261009'
$taskRunner = 'D:/coco_wire/experiments/frozen_native_mask_comparison/scripts/runner.py'
$taskOutput = "$taskRoot/runs/RUN_NATIVE_TARGET_LIFECYCLE_ENGINEERING_S0_02"
$taskScope = @{ environment='local_laptop_28358lan'; phase='detached_lifecycle_contract'; cpu_only=$true; no_gpu=$true; no_gt=$true; no_models=$true } | ConvertTo-Json -Compress
$taskArgs = @($taskRunner,'--study','STUDY_NATIVE_RESPONSE_TARGET_COMPARISON_20261009','--output',$taskOutput,
    '--run-id','RUN_NATIVE_TARGET_LIFECYCLE_ENGINEERING_S0_02','--cwd',$taskRoot,'--scope',$taskScope,
    '--snapshot',"$taskRoot/scripts/check_detached_lifecycle.py",'--snapshot',"$taskRoot/scripts/register_lifecycle_engineering02.ps1",'--snapshot',"$taskRoot/scripts/launch_wmi_hidden.ps1",
    '--expect',"$taskOutput/SUMMARY.json",'--expect',"$taskOutput/STARTED.json",'--metrics',"$taskOutput/SUMMARY.json",
    '--',$taskPython,"$taskRoot/scripts/check_detached_lifecycle.py",'--output',$taskOutput)
& $taskPython @taskArgs
exit $LASTEXITCODE
