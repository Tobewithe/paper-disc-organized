$ErrorActionPreference='Stop'
$root='D:/coco_wire/scripts/coverage_calibration_20260920'
$runs='D:/coco_wire/runs/mask_boundary_route_20260914'
$python='C:/Users/28358/anaconda3/envs/pytorch/python.exe'
$protocol="$root/PROTOCOL_COVERAGE_V7_20260920.json"
$p=Get-Content $protocol -Raw|ConvertFrom-Json
$env:PYTHONPATH='D:/coco_wire/py';$env:PYTHONUNBUFFERED='1';$env:OMP_NUM_THREADS='4';$env:MKL_NUM_THREADS='4'
$snapshots=@('--snapshot',$protocol,'--snapshot',$PSCommandPath)
foreach($name in @('calibrator.py','data_access.py','fit_calibrator.py','evaluate_calibrator.py','common.py','portable_risk.py','risk_calibration.py','mask_calibration.py')){$snapshots+=@('--snapshot',"$root/$name")}
$models="$runs/$($p.runs.fit)"
foreach($phase in @('fit','selection','evaluation')) {
    $id=$p.runs.$phase;$output="$runs/$id"
    $inputs=@('--input',$protocol,'--input',$p.split,'--input',$p.frozen_rcmc)
    $bank=if($phase -eq 'evaluation'){$p.val_bank}else{$p.train_bank}
    foreach($name in @('candidate_records.csv','instance_records.csv','image_ids.json')){$inputs+=@('--input',"$bank/$name")}
    if($phase -ne 'fit') {
        foreach($name in @('direct.json','purity.json','coverage.json','removed_purity.json')){$inputs+=@('--input',"$models/$name")}
        foreach($name in @('predictions_official_zero.json','predictions_smooth_gated.json')){$inputs+=@('--input',"$bank/$name")}
        $ann=if($phase -eq 'evaluation'){$p.val_annotations}else{$p.train_annotations}
        $inputs+=@('--input',$ann)
    }
    if($phase -eq 'fit') {
        $command=@($python,'-u',"$root/fit_calibrator.py",'--protocol',$protocol,'--output',$output)
    }else{
        $command=@($python,'-u',"$root/evaluate_calibrator.py",'--protocol',$protocol,'--output',$output,'--models',$models,'--phase',$phase)
        if($phase -eq 'evaluation') {
            $selection="$runs/$($p.runs.selection)/selection.json"
            $inputs+=@('--input',$selection);$command+=@('--selection',$selection)
        }
    }
    & $python "$root/runner.py" --study $p.study_id --run-id $id --output $output --cwd D:/coco_wire @inputs @snapshots --expect "$output/SUMMARY.json" --metrics "$output/SUMMARY.json" -- @command
    if($LASTEXITCODE -ne 0){throw "Coverage calibration failed in $phase"}
}
