# Authorized one-shot sequence. Scientific pass is not required between independent experiments;
# completion and independent configuration/evidence checks are required. No retries.
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$queueRoot='D:\coco_wire\baseline_native_20261013'
$queueStudy=Join-Path $queueRoot 'experiments\mask_readout_sphere_recovery_20261015'
$queueStatus=Join-Path $queueStudy 'SEQUENCE_34_STATUS.json'
$queueGuard=Join-Path $queueStudy '.QUEUE34_OWNER.guard'
$queueOwner=$null
function Save-Queue([string]$Phase,[object]$Extra=@{}) {
    $state=[ordered]@{phase=$Phase;updated_at=(Get-Date -Format o);controller_pid=$PID;host=$env:COMPUTERNAME;no_automatic_retry=$true;details=$Extra}
    [IO.File]::WriteAllText($queueStatus,($state|ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
}
function Invoke-Study([string]$Name,[string]$Stage) {
    $study=Join-Path $queueRoot ('experiments\'+$Name)
    Save-Queue ($Stage+'_running') @{study=$Name}
    $child=Start-Process -FilePath (Join-Path $PSHOME 'pwsh.exe') -ArgumentList @('-NoProfile','-WindowStyle','Hidden','-File',(Join-Path $study 'scripts\continue_independently.ps1')) -WindowStyle Hidden -PassThru -Wait -RedirectStandardOutput (Join-Path $queueStudy ($Stage+'_controller.stdout.log')) -RedirectStandardError (Join-Path $queueStudy ($Stage+'_controller.stderr.log'))
    if ($child.ExitCode -ne 0) { throw ($Stage+' controller failed; no next stage or retry') }
    $s=Get-Content -Raw (Join-Path $study 'AUTONOMOUS_STATUS.json')|ConvertFrom-Json
    if ($s.phase -ne 'completed') { throw ($Stage+' not completed') }
}
try {
    $queueOwner=[IO.File]::Open($queueGuard,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    Save-Queue 'waiting_for_E2_independent_audit'
    $deadline=(Get-Date).AddHours(4)
    do {
        $s=Get-Content -Raw (Join-Path $queueStudy 'AUTONOMOUS_STATUS.json')|ConvertFrom-Json
        if ($s.phase -eq 'error') { throw 'E2 controller failed; E3/E4 not started' }
        if ($s.phase -eq 'completed') { break }
        if ((Get-Date) -gt $deadline) { throw 'E2 completion wait exceeded four-hour bound' }
        Start-Sleep -Seconds 5
    } while ($true)
    $audit=Get-Content -Raw (Join-Path $queueStudy 'INDEPENDENT_LAPTOP_FULL.json')|ConvertFrom-Json
    if ($audit.status -ne 'passed') { throw 'E2 independent evaluation audit required' }
    while (Test-Path (Join-Path $queueStudy '.AUTONOMOUS_OWNER.guard')) { Start-Sleep -Seconds 1 }
    Invoke-Study 'mask_readout_quadrant_oracle_20261015' 'E3'
    $e3=Join-Path $queueRoot 'experiments\mask_readout_quadrant_oracle_20261015'
    if ((Get-Content -Raw (Join-Path $e3 'INDEPENDENT_LAPTOP_FULL.json')|ConvertFrom-Json).status -ne 'passed') { throw 'E3 independent audit required' }
    Invoke-Study 'mask_readout_aligned_repulsion_20261015' 'E4'
    Save-Queue 'completed' @{E2_E3_E4_finished_with_laptop_independent_checks=$true;transfer_to_primary_host='separate byte-verification step';no_S3_or_scaleup_started=$true}
} catch {
    Save-Queue 'error' @{error=$_.Exception.Message;outputs_preserved=$true;automatic_retry=$false}
    exit 1
} finally {
    if ($null -ne $queueOwner) { $queueOwner.Dispose();Remove-Item -LiteralPath $queueGuard }
}
