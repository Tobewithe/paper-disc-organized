$ErrorActionPreference = 'Stop'
$fullRoot = 'D:\coco_wire\experiments\triflow_fullscale_training_20261007'
$fullRun = Join-Path $fullRoot 'runs\RUN_TRIFLOW_FULL_PIPELINE_S0'
New-Item -ItemType Directory -Path $fullRun -Force | Out-Null
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class TriFlowAwake {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
# Process-scoped idle-sleep prevention; display and persistent power plan unchanged.
$awake = [TriFlowAwake]::SetThreadExecutionState([uint32]2147483649)
if ($awake -eq 0) { throw 'Unable to keep laptop awake for this owned training execution' }
try {
    & 'C:\Users\28358\anaconda3\envs\pytorch\python.exe' -X utf8 (Join-Path $fullRoot 'scripts\run_stage.py') --root $fullRoot --run-id RUN_TRIFLOW_FULL_PIPELINE_S0 --purpose full_original_COCO_training_then_evaluation --script run_fullscale_pipeline.py -- --root $fullRoot --run-id RUN_TRIFLOW_FULL_PIPELINE_S0 *> (Join-Path $fullRun 'scheduler_stdout.log')
    $fullCode = $LASTEXITCODE
    @{ exit_code=$fullCode; ended_at=(Get-Date).ToString('o'); idle_sleep_request_released=$true } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $fullRun 'SCHEDULER_EXIT.json') -Encoding utf8
    exit $fullCode
} finally {
    [TriFlowAwake]::SetThreadExecutionState([uint32]2147483648) | Out-Null
}
