# One-shot continuation of the already launched evaluation, not a recurring automation.
$ErrorActionPreference='Stop'
$project='C:/Dpan/codexproject/paper-disc-organized'
$study="$project/experiments/mask_boundary_route_20260914"
$execution="$study/execution/boundary_response_20260921"
New-Item -ItemType Directory -Force -Path $execution | Out-Null
$evaluation='RUN_b88ab674f90c488cbc185216fa15402c'
$analysis='RUN_9631795f20c34be7893a02b94de7cfc2'
$remoteRoot='D:/coco_wire/runs/mask_boundary_route_20260914'
function Save-State([string]$stage,[string]$detail) {
    [ordered]@{stage=$stage;detail=$detail;updated_at=(Get-Date).ToString('o');evaluation_run=$evaluation;analysis_run=$analysis;helper_pid=$PID} | ConvertTo-Json | Set-Content -LiteralPath "$execution/completion_status.json" -Encoding utf8
}
try {
    Save-State 'waiting_for_existing_evaluation' 'Evaluation continues in its existing SSH runner; no restart.'
    $failedReads=0
    while($true) {
        $json=& ssh -o BatchMode=yes -o ConnectTimeout=10 28358lan "powershell -NoProfile -Command Get-Content -LiteralPath $remoteRoot/$evaluation/run.json -Raw" 2> "$execution/status_read.stderr.log"
        if($LASTEXITCODE -ne 0) {
            $failedReads++
            Save-State 'remote_status_unavailable' "Read failures: $failedReads. No process termination or restart."
            if($failedReads -ge 20){throw 'Remote unavailable for 20 consecutive reads'}
            Start-Sleep -Seconds 30
            continue
        }
        $failedReads=0;$record=($json -join "`n") | ConvertFrom-Json
        if($record.status -eq 'completed'){break}
        if($record.status -in @('failed','cancelled','interrupted')){throw "Evaluation ended: $($record.status)"}
        Start-Sleep -Seconds 30
    }
    Save-State 'analysis' 'Running paired image analysis once after completed evaluation.'
    & ssh -o BatchMode=yes -o ConnectTimeout=10 28358lan 'powershell -NoProfile -File D:/coco_wire/scripts/boundary_response_20260921/launch_analysis.ps1' > "$execution/analysis_launcher.stdout.log" 2> "$execution/analysis_launcher.stderr.log"
    if($LASTEXITCODE -ne 0){throw 'Analysis launcher failed; do not refit or retune'}
    Save-State 'returning' 'Copying evaluation and analysis artifacts over LAN.'
    & scp -r "28358lan:$remoteRoot/$evaluation" "28358lan:$remoteRoot/$analysis" "$study/runs/" > "$execution/return.stdout.log" 2> "$execution/return.stderr.log"
    if($LASTEXITCODE -ne 0){throw 'Artifact return failed; source remains on laptop'}
    $result=Get-Content -LiteralPath "$study/runs/$evaluation/SUMMARY.json" -Raw | ConvertFrom-Json
    Save-State 'completed' $result.decision
} catch {
    Save-State 'needs_attention' $_.Exception.Message
    $_ | Out-String | Set-Content -LiteralPath "$execution/failure.txt" -Encoding utf8
    exit 1
}
