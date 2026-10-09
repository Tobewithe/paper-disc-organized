$ErrorActionPreference = 'Stop'
$taskPython = 'C:/Users/28358/anaconda3/envs/pytorch/python.exe'
$taskRoot = 'D:/coco_wire/experiments/native_response_target_comparison_20261009'
$taskRunner = 'D:/coco_wire/experiments/frozen_native_mask_comparison/scripts/runner.py'
$taskProtocolSha = 'cc0d3169e32fc1d42dbd8f1027861ef29fa75f18bb145addd73e6c77d6a94057'
$taskPreflight = "$taskRoot/runs/RUN_NATIVE_TARGET_PREFLIGHT_S0_01"
$taskPipeline = "$taskRoot/runs/RUN_NATIVE_TARGET_PIPELINE_S0_01"
$taskScope = @{ environment='local_laptop_28358lan'; phase='preflight_identity'; no_forward=$true; no_gt_parse=$true } | ConvertTo-Json -Compress
$taskArgs = @($taskRunner,'--study','STUDY_NATIVE_RESPONSE_TARGET_COMPARISON_20261009','--output',$taskPreflight,
    '--run-id','RUN_NATIVE_TARGET_PREFLIGHT_S0_01','--cwd',$taskRoot,'--scope',$taskScope,
    '--input',"$taskRoot/PROTOCOL.md",'--snapshot',"$taskRoot/scripts/preflight_targets.py",'--snapshot',"$taskRoot/scripts/target_common.py",
    '--snapshot',"$taskRoot/scripts/register_target_pipeline.ps1",'--snapshot',$taskRunner,
    '--expect',"$taskPreflight/COMPLETE.json",'--expect',"$taskPreflight/EXECUTION_CONFIG.json",'--expect',"$taskPreflight/SUMMARY.json",'--metrics',"$taskPreflight/SUMMARY.json",
    '--',$taskPython,"$taskRoot/scripts/preflight_targets.py",'--root',$taskRoot,'--output',$taskPreflight,'--protocol-sha256',$taskProtocolSha,
    '--materialize-run','D:/coco_wire/experiments/triflow_20k_training_20261007/runs/RUN_TRIFLOW_20K_DATA_MATERIALIZE_S0',
    '--native-reference','D:/coco_wire/experiments/frozen_native_mask_comparison/runs/RUN_FROZEN_NATIVE_5K_INFERENCE_S0_02',
    '--frozen-scripts','D:/coco_wire/experiments/frozen_native_mask_comparison/scripts',
    '--weights','D:/coco_wire/models/yolo26m-seg.pt','--vendor','D:/coco_wire/vendor_8.4.100','--runner',$taskRunner)
& $taskPython @taskArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$taskScope = @{ environment='local_laptop_28358lan'; phase='finite_engineering_then20k_two_targets_native5k'; gpu_serial=$true; no_yolo_training=$true; no_selection=$true; protocol_sha256=$taskProtocolSha } | ConvertTo-Json -Compress
$taskArgs = @($taskRunner,'--study','STUDY_NATIVE_RESPONSE_TARGET_COMPARISON_20261009','--output',$taskPipeline,
    '--run-id','RUN_NATIVE_TARGET_PIPELINE_S0_01','--cwd',$taskRoot,'--scope',$taskScope,
    '--input',"$taskRoot/PROTOCOL.md",'--input',"$taskPreflight/EXECUTION_CONFIG.json",
    '--snapshot',"$taskRoot/scripts/run_target_pipeline.py",'--snapshot',"$taskRoot/scripts/target_common.py",'--snapshot',"$taskRoot/scripts/register_target_pipeline.ps1",'--snapshot',$taskRunner,
    '--expect',"$taskPipeline/PIPELINE_COMPLETE.json",'--expect',"$taskPipeline/ENGINEERING_GATE.json",'--expect',"$taskPipeline/SUMMARY.json",'--metrics',"$taskPipeline/SUMMARY.json",
    '--',$taskPython,"$taskRoot/scripts/run_target_pipeline.py",'--config',"$taskPreflight/EXECUTION_CONFIG.json",'--protocol-sha256',$taskProtocolSha,
    '--output',$taskPipeline,'--suffix','01')
& $taskPython @taskArgs
exit $LASTEXITCODE
