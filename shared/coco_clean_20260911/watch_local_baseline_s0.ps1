$ErrorActionPreference = 'SilentlyContinue'
$log = 'C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911\runs\boundary_ownership_20260914\queue_logs\baseline_s0.log'
$parentId = 25036
$watchLog = 'C:\Dpan\codexproject\paper-disc\experiments\coco_clean_20260911\runs\boundary_ownership_20260914\LOCAL_WATCHER.log'

while ($true) {
  if (Test-Path -LiteralPath $log) {
    $fs = $null
    $sr = $null
    try {
      $fs = [IO.File]::Open($log, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
      $sr = New-Object IO.StreamReader($fs)
      $txt = $sr.ReadToEnd()
      if ($txt -match 'START 2026-09-14T01:01:55' -and $txt -match 'EXIT_CODE 0') {
        Start-Sleep -Seconds 5
        Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $parentId } | ForEach-Object {
          Stop-Process -Id $_.ProcessId -Force
        }
        Stop-Process -Id $parentId -Force
        Add-Content -LiteralPath $watchLog -Value "Stopped local queue after baseline_s0 at $(Get-Date -Format o)"
        break
      }
    } finally {
      if ($sr) { $sr.Dispose() }
      if ($fs) { $fs.Dispose() }
    }
  }
  Start-Sleep -Seconds 30
}
