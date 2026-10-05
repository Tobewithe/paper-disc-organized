import argparse,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--bank-run',required=True);p.add_argument('--score-run',required=True)
a=p.parse_args();root=Path(a.project);study=root/'experiments/mask_boundary_route_20260914';scripts=study/'scripts';runs=study/'runs'
subprocess.run([sys.executable,str(scripts/'return_factorial_outputs.py'),'--runs-root',str(runs),'--source',a.bank_run,'--timeout-hours','3'],check=True,cwd=root)
out=runs/a.score_run;source=runs/a.bank_run;ann=root/'assets/datasets/coco/annotations/instances_val2017.json'
command=[sys.executable,str(root/'shared/tools/research_runner/runner.py'),'--study','STUDY_8fb3468ebb704682a2225ebed0e16206',
    '--run-id',a.score_run,'--output',str(out),'--cwd',str(root),'--input',str(source/'predictions_global_0.25.json'),
    '--input',str(ann),'--snapshot',str(scripts/'score_exported_variant.py'),'--snapshot',str(scripts/'continue_global_control.py'),
    '--expect',str(out/'SUMMARY.json'),'--metrics',str(out/'SUMMARY.json'),'--',sys.executable,'-u',str(scripts/'score_exported_variant.py'),
    '--input',str(source),'--variant','global_0.25','--annotations',str(ann),'--output',str(out),
    '--baseline-reference',str(runs/'RUN_2ae67d6556a14848bceb5783a72e5790/predictions_official_zero.json')]
subprocess.run(command,check=True,cwd=root)
