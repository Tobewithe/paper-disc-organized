$ErrorActionPreference='Stop'
$py='C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$root='D:\coco_wire\pcdcr_20261002'
$base=@('D:\coco_wire\runner.py','--study','STUDY_ee75286e-6996-4cad-8ab3-a4d6fa066db6','--cwd',$root,'--input',"$root\RUN_CONFIG.json",'--input',"$root\source_OFFICIAL_MANIFEST.json",'--snapshot',"$root\RUN_CONFIG.json",'--snapshot',"$root\PROTOCOL.md",'--snapshot',"$root\scripts\run_pcdcr.py",'--snapshot',"$root\scripts\pcdcr_models.py",'--snapshot',"$root\scripts\pcdcr_evaluation.py")
$out="$root\runs\RUN_stageI_seed0_20261002"
& $py @base --output "$root\runs\RUN_prepare_retry1_20261002" --run-id RUN_pcdcr_prepare_retry1_20261002 --expect "$out\PREPARE_COMPLETE.json" -- $py -u "$root\scripts\run_pcdcr.py" --config "$root\RUN_CONFIG.json" --out $out --stage prepare
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
& $py @base --input "$out\MISMATCH_MANIFEST.json" --input "$out\ASSET_MANIFEST.json" --output $out --run-id RUN_pcdcr_stageI_seed0_20261002 --expect "$out\COMPLETE.json" -- $py -u "$root\scripts\run_pcdcr.py" --config "$root\RUN_CONFIG.json" --out $out --stage finish
exit $LASTEXITCODE
