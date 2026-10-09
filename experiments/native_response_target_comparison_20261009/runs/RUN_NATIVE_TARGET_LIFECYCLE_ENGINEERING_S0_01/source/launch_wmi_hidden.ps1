param([Parameter(Mandatory=$true)][string]$TaskScript)
$ErrorActionPreference = 'Stop'
$taskRoot = 'D:/coco_wire/experiments/native_response_target_comparison_20261009'
$taskScriptPath = (Resolve-Path -LiteralPath $TaskScript).Path
if (-not $taskScriptPath.StartsWith((Resolve-Path -LiteralPath "$taskRoot/scripts").Path, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Launcher target must remain in this Study scripts directory'
}
$taskStartup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow=[uint16]0 }
$taskCommand = '"C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "' + $taskScriptPath + '"'
$taskResult = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine=$taskCommand; CurrentDirectory=$taskRoot; ProcessStartupInformation=$taskStartup
}
@{ return_value=$taskResult.ReturnValue; process_id=$taskResult.ProcessId; command=$taskCommand; hidden=$true;
   scope='WMI-created fresh process, independent of SSH parent job; no existing process moved or stopped' } | ConvertTo-Json -Compress
if ($taskResult.ReturnValue -ne 0) { exit 1 }
