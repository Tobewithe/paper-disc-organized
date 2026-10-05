$ErrorActionPreference='Stop'
$py='C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$root='D:\coco_wire\bgcr_residual_20261002'
$env:PYTHONPATH='D:\coco_wire\py'
$rid='RUN_5d8a367e2f574aa18c47c26660a2a110'
$out="$root\runs\$rid"
$rec=@('D:\coco_wire\runner.py','--study','STUDY_3854d243bdd24c82af2504035a931882','--run-id',$rid,'--cwd',$root,'--output',$out,'--input',"$root\RUN_CONFIG.json",'--input','D:\coco_wire\data\official_tal_affine_20260930\runs\official_cache\INDEX.json','--input','D:\coco_wire\pcdcr_20261002\source_PER_CANDIDATE.jsonl','--snapshot',"$root\PROTOCOL.md",'--snapshot',"$root\RUN_CONFIG.json")
Get-ChildItem "$root\scripts\*.py" | ForEach-Object { $rec += @('--snapshot',$_.FullName) }
& $py @rec --expect "$out\COMPLETE.json" -- $py -u "$root\scripts\run_bgcr.py" --config "$root\RUN_CONFIG.json" --out $out
exit $LASTEXITCODE
