$ErrorActionPreference = 'Stop'
$taskRoot = 'D:\coco_wire\experiments\native_response_target_comparison_20261009'
$taskPython = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$taskRunner = 'D:\coco_wire\experiments\frozen_native_mask_comparison\scripts\runner.py'
$taskOutput = "$taskRoot\runs\RUN_NATIVE_TARGET_LIFECYCLE_ENGINEERING_S0_06"
$taskCommand = '"' + $taskPython + '" "' + $taskRunner + '" --study STUDY_NATIVE_RESPONSE_TARGET_COMPARISON_20261009 --output "' + $taskOutput + '" --run-id RUN_NATIVE_TARGET_LIFECYCLE_ENGINEERING_S0_06 --cwd "' + $taskRoot + '" --snapshot "' + $taskRoot + '\scripts\check_detached_lifecycle.py" --snapshot "' + $taskRoot + '\scripts\launch_lifecycle_python06.ps1" --expect "' + $taskOutput + '\SUMMARY.json" --expect "' + $taskOutput + '\STARTED.json" --metrics "' + $taskOutput + '\SUMMARY.json" -- "' + $taskPython + '" "' + $taskRoot + '\scripts\check_detached_lifecycle.py" --output "' + $taskOutput + '"'
$taskStartup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{
    ShowWindow=[uint16]0; CreateFlags=[uint32]16777216
}
$taskResult = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine=$taskCommand; CurrentDirectory=$taskRoot; ProcessStartupInformation=$taskStartup
}
@{ return_value=$taskResult.ReturnValue; process_id=$taskResult.ProcessId; command=$taskCommand; hidden=$true;
   create_flags=$taskStartup.CreateFlags; show_window=$taskStartup.ShowWindow;
   scope='direct Python original runner CPU-only5second; solely CREATE_BREAKAWAY_FROM_JOB' } | ConvertTo-Json -Compress
if ($taskResult.ReturnValue -ne 0) { exit 1 }
