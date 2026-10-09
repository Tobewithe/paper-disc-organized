$ErrorActionPreference = 'Stop'
$taskPython = 'C:/Dpan/envsfiles/CondaData/envs/pytorch/python.exe'
$taskRoot = 'C:/Dpan/codexproject/paper-disc-organized/experiments/native_response_target_comparison_20261009'
$taskRunner = 'C:/Dpan/codexproject/paper-disc-organized/experiments/frozen_native_mask_comparison/scripts/runner.py'
$taskOutput = "$taskRoot/runs/RUN_NATIVE_TARGET_FIT_CACHE_CONTRACT_S0_01"
if (Test-Path -LiteralPath $taskOutput) { throw 'Existing cache contract is preserved; retry requires a new Run' }
$taskCost = Get-Content -LiteralPath "$taskRoot/runs/RUN_NATIVE_TARGET_COST_PANEL_S0_01/run.json" -Raw | ConvertFrom-Json
if ($taskCost.source_kind -ne 'runner_observed' -or $taskCost.status -ne 'completed' -or $taskCost.return_code -ne 0 -or $taskCost.artifact_completeness -ne 'complete') { throw 'Actual isolated cost must end before the CPU cache contract' }
$taskExtract = "$taskRoot/runs/RUN_NATIVE_TARGET_FORMAL_EXTRACT_S0_01"
$taskFit = "$taskRoot/runs/RUN_NATIVE_TARGET_FORMAL_FIT_I_S0_01"
foreach ($taskSource in @($taskExtract,$taskFit,"$taskRoot/runs/RUN_NATIVE_TARGET_COST_PANEL_S0_01")) {
    $taskTransfer = Get-Content -LiteralPath "$taskSource/transfer.json" -Raw | ConvertFrom-Json
    if ($taskTransfer.status -ne 'complete_scientific_consumption' -or $taskTransfer.all_manifest_sha256_verified -ne $true) { throw 'Full original-byte return must be verified before the cache contract' }
}
$taskScope = @{ environment='desktop_cpu_saved_npz_contract'; gt_json_parsed=$false; gpu_used=$false; model_fit=$false;
    ap_measured=$false; current_production_used_cached_fit=$false; original_wall_clocks_remeasured=$false;
    all_part_bytes_digests_and_cached_integer_labels=$true; legacy_repeated_row_access='fixed endpoints plus first actual fallback per part' } | ConvertTo-Json -Compress
$taskArgs = @($taskRunner,'--study','STUDY_NATIVE_RESPONSE_TARGET_COMPARISON_20261009','--output',$taskOutput,
    '--run-id','RUN_NATIVE_TARGET_FIT_CACHE_CONTRACT_S0_01','--cwd',$taskRoot,'--scope',$taskScope,
    '--input',"$taskExtract/SUMMARY.json",'--input',"$taskExtract/COMPLETE.json",'--input',"$taskExtract/ROWS_PARTS.json",
    '--input',"$taskFit/SUMMARY.json",'--input',"$taskFit/COMPLETE.json",'--input',"$taskRoot/runs/RUN_NATIVE_TARGET_COST_PANEL_S0_01/run.json",
    '--snapshot',"$taskRoot/scripts/check_fit_cache_contract.py",'--snapshot',"$taskRoot/scripts/fit_target.py",
    '--snapshot',"$taskRoot/scripts/fit_target_v2.py",'--snapshot',"$taskRoot/scripts/target_common.py",
    '--snapshot',"$taskRoot/scripts/register_fit_cache_contract.ps1",'--snapshot',$taskRunner,
    '--expect',"$taskOutput/SUMMARY.json",'--expect',"$taskOutput/COMPLETE.json",'--metrics',"$taskOutput/SUMMARY.json",
    '--',$taskPython,"$taskRoot/scripts/check_fit_cache_contract.py",'--extraction',$taskExtract,'--fit-reference',$taskFit,
    '--fit-original',"$taskRoot/scripts/fit_target.py",'--fit-cached',"$taskRoot/scripts/fit_target_v2.py",'--output',$taskOutput)
& $taskPython @taskArgs
exit $LASTEXITCODE
