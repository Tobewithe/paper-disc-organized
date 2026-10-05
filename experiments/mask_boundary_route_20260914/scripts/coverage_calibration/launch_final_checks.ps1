$ErrorActionPreference='Stop'
$root='D:/coco_wire/scripts/coverage_calibration_20260920'
$runs='D:/coco_wire/runs/mask_boundary_route_20260914'
$python='C:/Users/28358/anaconda3/envs/pytorch/python.exe'
$protocol="$root/PROTOCOL_COVERAGE_V7_20260920.json"
$p=Get-Content $protocol -Raw|ConvertFrom-Json
$ids=Get-Content "$root/execution_runs.json" -Raw|ConvertFrom-Json
$env:PYTHONPATH='D:/coco_wire/py';$env:PYTHONUNBUFFERED='1';$env:OMP_NUM_THREADS='4';$env:MKL_NUM_THREADS='4'
$deadline=(Get-Date).AddHours(2)
while($true) {
    $f="$runs/$($ids.evaluation)/run.json"
    if(Test-Path $f){$r=Get-Content $f -Raw|ConvertFrom-Json
        if($r.status -eq 'completed' -and $r.return_code -eq 0){break}
        if($r.status -in @('failed','interrupted','cancelled')){throw 'Evaluation dependency did not succeed'}
    }
    if((Get-Date) -gt $deadline){throw 'Evaluation dependency timeout'}
    Start-Sleep -Seconds 15
}
$models="$runs/$($ids.fit)";$selection="$runs/$($ids.selection)/selection.json";$evaluation="$runs/$($ids.evaluation)";$inference="$runs/$($ids.inference_check)"
$snapshots=@('--snapshot',$protocol,'--snapshot',$PSCommandPath)
foreach($name in @('check_inference.py','analyze_results.py','calibrator.py','data_access.py','common.py','portable_risk.py','risk_calibration.py','mask_calibration.py')){$snapshots+=@('--snapshot',"$root/$name")}
foreach($phase in @('inference_check','analysis')) {
    $id=$ids.$phase;$output="$runs/$id"
    $inputs=@('--input',$protocol,'--input',$selection,'--input',"$evaluation/SUMMARY.json",'--input',"$models/SUMMARY.json",'--input',$p.frozen_rcmc)
    foreach($name in @('direct.json','purity.json','coverage.json','removed_purity.json')){$inputs+=@('--input',"$models/$name")}
    if($phase -eq 'inference_check') {
        foreach($name in @('candidate_records.csv','image_ids.json','predictions_official_zero.json','predictions_smooth_gated.json')){$inputs+=@('--input',"$($p.val_bank)/$name")}
        $inputs+=@('--input',"$evaluation/candidate_decisions.npz",'--input',$p.weights,'--input',$p.val_annotations)
        $command=@($python,'-u',"$root/check_inference.py",'--protocol',$protocol,'--output',$output,'--models',$models,'--selection',$selection,'--evaluation',$evaluation)
    }else{
        $inputs+=@('--input',"$evaluation/instance_decisions.csv",'--input',"$inference/SUMMARY.json",'--input',$p.val_annotations)
        $command=@($python,'-u',"$root/analyze_results.py",'--protocol',$protocol,'--output',$output,'--fit',$models,'--selection',"$runs/$($ids.selection)",'--evaluation',$evaluation,'--inference',$inference)
    }
    & $python "$root/runner.py" --study $p.study_id --run-id $id --output $output --cwd D:/coco_wire @inputs @snapshots --expect "$output/SUMMARY.json" --metrics "$output/SUMMARY.json" -- @command
    if($LASTEXITCODE -ne 0){throw "Final check failed in $phase"}
}
