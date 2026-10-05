"""Finite collection, fitting and frozen-gate validation after a named bank run."""
import argparse,json,os,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--bank-run',required=True)
    p.add_argument('--prepare-run',required=True);p.add_argument('--fit-run',required=True);p.add_argument('--eval-run',required=True)
    args=p.parse_args();root=Path(args.project);study=root/'experiments/mask_boundary_route_20260914';scripts=study/'scripts'
    runs=study/'runs';bank=runs/args.bank_run;prep=runs/args.prepare_run;fit=runs/args.fit_run;evalout=runs/args.eval_run
    env=os.environ.copy();env['OMP_NUM_THREADS']='4';env['MKL_NUM_THREADS']='4'
    collector=[sys.executable,str(scripts/'return_factorial_outputs.py'),'--runs-root',str(runs),'--source',args.bank_run,'--timeout-hours','3']
    subprocess.run(collector,env=env,cwd=root,check=True)
    runner=root/'shared/tools/research_runner/runner.py'
    runscript=scripts/'fit_risk_calibration.py'
    jobs=[('fit',fit,bank,prep/'calibration_annotations.json'),
          ('evaluate',evalout,runs/'RUN_2ae67d6556a14848bceb5783a72e5790',root/'assets/datasets/coco/annotations/instances_val2017.json')]
    for phase,out,source,annotations in jobs:
        command=[sys.executable,str(runner),'--study','STUDY_8fb3468ebb704682a2225ebed0e16206',
            '--run-id',out.name,'--output',str(out),'--cwd',str(root),
            '--input',str(source/'candidate_records.csv'),'--input',str(source/'instance_records.csv'),
            '--input',str(annotations),'--input',str(prep/'split.json'),
            '--snapshot',str(runscript),'--snapshot',str(scripts/'continue_risk_pipeline.py'),
            '--snapshot',str(study/'PROTOCOL_RISK_V4.json'),'--expect',str(out/'SUMMARY.json'),
            '--metrics',str(out/'SUMMARY.json'),'--',sys.executable,'-u',str(runscript),'--phase',phase,
            '--input',str(source),'--annotations',str(annotations),'--output',str(out)]
        if phase=='fit':command+=['--split',str(prep/'split.json')]
        else:command+=['--model-run',str(fit),'--global-bank',str(runs/'RUN_8fe82201101b4e9d941cb07a3496c110')]
        subprocess.run(command,env=env,cwd=root,check=True)
    print(json.dumps(dict(status='completed',fit_run=args.fit_run,evaluation_run=args.eval_run)),flush=True)

if __name__=='__main__':main()
