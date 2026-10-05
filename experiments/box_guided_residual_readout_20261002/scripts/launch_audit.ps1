$ErrorActionPreference='Stop'
$py='C:\Users\28358\anaconda3\envs\pytorch\python.exe'
$root='D:\coco_wire\bgcr_residual_20261002'
$env:PYTHONPATH='D:\coco_wire\py'
$rid='RUN_c2f9a64b7e124c5da191b2c25374ec90'
$out="$root\runs\$rid"
$source="$root\runs\RUN_5d8a367e2f574aa18c47c26660a2a110"
$rec=@('D:\coco_wire\runner.py','--study','STUDY_3854d243bdd24c82af2504035a931882','--run-id',$rid,'--cwd',$root,'--output',$out,'--input',"$root\RUN_CONFIG.json",'--input',"$source\B_epoch10.pt",'--input',"$source\C_epoch10.pt",'--input',"$source\ROI_FEATURES.pt",'--input',"$source\FINAL_COEFFICIENTS.pt",'--snapshot',"$root\AUDIT_PROTOCOL.md",'--snapshot',"$root\scripts\audit_training.py",'--snapshot',"$root\scripts\run_bgcr.py",'--snapshot',"$root\scripts\bgcr_models.py")
& $py @rec --expect "$out\COMPLETE.json" -- $py -u "$root\scripts\audit_training.py" --config "$root\RUN_CONFIG.json" --source-run $source --out $out
exit $LASTEXITCODE
