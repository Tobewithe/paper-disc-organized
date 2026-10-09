$ErrorActionPreference = 'Stop'
$taskPython = 'C:/Users/28358/anaconda3/envs/pytorch/python.exe'
$taskRoot = 'D:/coco_wire/experiments/native_response_target_comparison_20261009'
$taskRunner = 'D:/coco_wire/experiments/frozen_native_mask_comparison/scripts/runner.py'
$taskOutput = "$taskRoot/runs/RUN_NATIVE_TARGET_LIFECYCLE_COLLECTION_S0_01"
$taskScope = @{ environment='local_laptop_28358lan'; phase='preserve_failed_cpu_lifecycle_runs'; no_gpu=$true; no_gt=$true; source_runs=@('05','06'); science_sources_unchanged=$true } | ConvertTo-Json -Compress
$taskArgs = @($taskRunner,'--study','STUDY_NATIVE_RESPONSE_TARGET_COMPARISON_20261009','--output',$taskOutput,
    '--run-id','RUN_NATIVE_TARGET_LIFECYCLE_COLLECTION_S0_01','--cwd',$taskRoot,'--scope',$taskScope,
    '--snapshot',"$taskRoot/scripts/seal_extra_runs.py",'--snapshot',"$taskRoot/scripts/register_lifecycle_collection.ps1",
    '--input',"$taskRoot/runs/RUN_NATIVE_TARGET_LIFECYCLE_ENGINEERING_S0_05/run.json",
    '--input',"$taskRoot/runs/RUN_NATIVE_TARGET_LIFECYCLE_ENGINEERING_S0_06/run.json",
    '--expect',"$taskOutput/EXTRA_RUNS_BUNDLE.zip",'--expect',"$taskOutput/BUNDLE_MANIFEST.json",'--expect',"$taskOutput/COMPLETE.json",'--expect',"$taskOutput/SUMMARY.json",'--metrics',"$taskOutput/SUMMARY.json",
    '--',$taskPython,"$taskRoot/scripts/seal_extra_runs.py",'--run',"$taskRoot/runs/RUN_NATIVE_TARGET_LIFECYCLE_ENGINEERING_S0_05",
    '--run',"$taskRoot/runs/RUN_NATIVE_TARGET_LIFECYCLE_ENGINEERING_S0_06",'--output',$taskOutput)
& $taskPython @taskArgs
exit $LASTEXITCODE
