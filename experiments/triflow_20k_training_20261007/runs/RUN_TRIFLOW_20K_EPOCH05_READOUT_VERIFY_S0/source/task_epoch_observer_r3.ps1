$ErrorActionPreference = 'Stop'
$epochRoot = 'D:\coco_wire\experiments\triflow_20k_training_20261007'
$epochRun = Join-Path $epochRoot 'runs\RUN_TRIFLOW_20K_PIPELINE_S0_R3'
New-Item -ItemType Directory -Path $epochRun -Force | Out-Null
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class TriFlowDiagnosticsAwake {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
if ([TriFlowDiagnosticsAwake]::SetThreadExecutionState([uint32]2147483649) -eq 0) { throw 'Unable to keep this owned execution awake' }
try {
    & 'C:\Users\28358\anaconda3\envs\pytorch\python.exe' -X utf8 (Join-Path $epochRoot 'scripts\run_stage.py') --root $epochRoot --run-id RUN_TRIFLOW_20K_PIPELINE_S0_R3 --purpose verified_minimal_diagnostics_continuation --script run_epoch_pipeline_r3.py -- --root $epochRoot --run-id RUN_TRIFLOW_20K_PIPELINE_S0_R3 --resume-from (Join-Path $epochRoot 'runs\RUN_TRIFLOW_20K_DIAGNOSTICS_REVISION_S0\resume_snapshot.pt') --parity-receipt (Join-Path $epochRoot 'runs\RUN_TRIFLOW_20K_DIAGNOSTICS_PARITY_S0\DIAGNOSTICS_PARITY.json') *> (Join-Path $epochRun 'scheduler_stdout.log')
    $epochExit = $LASTEXITCODE
    @{exit_code=$epochExit;ended_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $epochRun 'SCHEDULER_EXIT.json') -Encoding utf8
    exit $epochExit
} finally {
    [TriFlowDiagnosticsAwake]::SetThreadExecutionState([uint32]2147483648) | Out-Null
}
