param([Parameter(Mandatory=$true)][ValidateSet('production_return','cost_return','engineering_cost_verify','formal_cost_verify')][string]$Phase)
$ErrorActionPreference = 'Stop'
$taskPython = 'C:/Dpan/envsfiles/CondaData/envs/pytorch/python.exe'
$taskRoot = 'C:/Dpan/codexproject/paper-disc-organized/experiments/native_response_target_comparison_20261009'
$taskRunner = 'C:/Dpan/codexproject/paper-disc-organized/experiments/frozen_native_mask_comparison/scripts/runner.py'
$taskHelper = 'C:/Dpan/codexproject/paper-disc-organized/experiments/frozen_native_mask_comparison/scripts/verify_cost_panel.py'
$taskNames = @{
    production_return='RUN_NATIVE_TARGET_PRODUCTION_RETURN_VERIFY_S0_01'
    cost_return='RUN_NATIVE_TARGET_COST_RETURN_VERIFY_S0_01'
    engineering_cost_verify='RUN_NATIVE_TARGET_COST_ENGINEERING_VERIFY_S0_01'
    formal_cost_verify='RUN_NATIVE_TARGET_COST_PANEL_VERIFY_S0_01'
}
$taskName = $taskNames[$Phase]
$taskOutput = "$taskRoot/runs/$taskName"
if (Test-Path -LiteralPath $taskOutput) { throw 'Existing independent Run is preserved; retry requires a newly reviewed Run ID' }
$taskScope = @{
    environment='desktop_cpu_saved_evidence_consumption'; phase=$Phase; gt_parsed=$false; gpu_used=$false;
    new_forward=$false; original_wall_clocks_remeasured=$false; original_source_host='28358lan';
    scientific_parameters_changed=$false; complete_return_required=$true
} | ConvertTo-Json -Compress
$taskArgs = @($taskRunner,'--study','STUDY_NATIVE_RESPONSE_TARGET_COMPARISON_20261009','--output',$taskOutput,
    '--run-id',$taskName,'--cwd',$taskRoot,'--scope',$taskScope,
    '--input',"$taskRoot/PROTOCOL.md",'--input',"$taskRoot/COST_PROTOCOL.md",
    '--snapshot',"$taskRoot/scripts/register_target_consumption.ps1",'--snapshot',$taskRunner,
    '--expect',"$taskOutput/SUMMARY.json",'--metrics',"$taskOutput/SUMMARY.json")
if ($Phase -in @('production_return','cost_return')) {
    $taskKind = if ($Phase -eq 'production_return') { 'production' } else { 'cost' }
    $taskCollectionName = if ($taskKind -eq 'production') { 'RUN_NATIVE_TARGET_COLLECTION_S0_01' } else { 'RUN_NATIVE_TARGET_COST_COLLECTION_S0_01' }
    $taskCollection = "$taskRoot/runs/$taskCollectionName"
    $taskArgs += @('--input',"$taskCollection/run.json",'--input',"$taskCollection/SUMMARY.json",'--input',"$taskCollection/COMPLETE.json",
        '--input',"$taskCollection/BUNDLE_MANIFEST.json",'--snapshot',"$taskRoot/scripts/receive_target_bundle_v2.py",
        '--snapshot',"$taskRoot/scripts/receive_bundle.py",'--snapshot',"$taskRoot/scripts/target_common.py",
        '--expect',"$taskOutput/COMPLETE.json",'--',$taskPython,"$taskRoot/scripts/receive_target_bundle_v2.py",
        '--base-receiver',"$taskRoot/scripts/receive_bundle.py",'--collection',$taskCollection,'--destination',"$taskRoot/runs",
        '--kind',$taskKind,'--output',$taskOutput)
} else {
    $taskMode = if ($Phase -eq 'engineering_cost_verify') { 'engineering' } else { 'formal' }
    $taskSourceName = if ($taskMode -eq 'engineering') { 'RUN_NATIVE_TARGET_COST_ENGINEERING_S0_01' } else { 'RUN_NATIVE_TARGET_COST_PANEL_S0_01' }
    $taskSource = "$taskRoot/runs/$taskSourceName"
    $taskReference = "$taskRoot/runs/RUN_NATIVE_TARGET_FORMAL_INFERENCE_S0_01"
    $taskArgs += @('--input',"$taskSource/run.json",'--input',"$taskSource/SUMMARY.json",'--input',"$taskSource/COST_INPUTS.json",
        '--input',"$taskReference/SUMMARY.json",'--input',"$taskReference/COMPLETE.json",'--snapshot',$taskHelper,
        '--snapshot',"$taskRoot/scripts/verify_target_cost.py",'--snapshot',"$taskRoot/scripts/verify_target_cost_v2.py",
        '--expect',"$taskOutput/VERIFICATION.json",'--',$taskPython,"$taskRoot/scripts/verify_target_cost_v2.py",
        '--base-checker',"$taskRoot/scripts/verify_target_cost.py",'--helper',$taskHelper,'--study',$taskRoot,
        '--input',$taskSource,'--reference',$taskReference,'--mode',$taskMode,'--output',$taskOutput)
}
& $taskPython @taskArgs
exit $LASTEXITCODE
