$ErrorActionPreference = 'Stop'
$python = 'C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$root = 'D:\coco_wire\experiments\coefficient_recoverability_rederive_20261010'
$run = Join-Path $root 'runs/RUN_REDERIVE_S0_R1'
New-Item -ItemType Directory -Force -Path $run | Out-Null
& $python "$root\rederive_recoverability.py" `
  --cache "$root\cache" `
  --annotations 'D:\coco_wire\data\annotations\instances_val2017.json' `
  --out $run *> "$run\stdout.log"
"exit_code=$LASTEXITCODE at $(Get-Date -Format o)" | Set-Content -LiteralPath "$run\TASK_EXIT.txt"
