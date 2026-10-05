"""Run one finite remote queue, return artifacts, then score frozen local gates."""
import argparse,base64,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--transfer-only',action='store_true');a=p.parse_args()
root=Path(a.project);study=root/'experiments/mask_boundary_route_20260914';scripts=study/'scripts';runs=study/'runs'
protocol=json.loads((study/'PROTOCOL_TRANSFER_V5.json').read_text(encoding='utf-8'));ids=protocol['run_ids']
for key in ('download','speed','transfer_bank'):(runs/ids[key]).mkdir(parents=True,exist_ok=True)
control=runs/ids['transfer_bank']
code="$ErrorActionPreference='Stop'\n$env:PYTHONPATH='D:/coco_wire/py'\n& 'C:/Users/28358/anaconda3/envs/pytorch/python.exe' -u 'D:/coco_wire/scripts/transfer_v5_20260915/run_transfer_remote.py' --scripts 'D:/coco_wire/scripts/transfer_v5_20260915'\nexit $LASTEXITCODE"
encoded=base64.b64encode(code.encode('utf-16-le')).decode('ascii')
if a.transfer_only:
    code=code.replace(" --scripts 'D:/coco_wire/scripts/transfer_v5_20260915'", " --scripts 'D:/coco_wire/scripts/transfer_v5_20260915' --stage transfer")
    encoded=base64.b64encode(code.encode('utf-16-le')).decode('ascii')
suffix='_transfer' if a.transfer_only else ''
with (control/('remote_queue'+suffix+'_stdout.log')).open('wb') as stdout,(control/('remote_queue'+suffix+'_stderr.log')).open('wb') as stderr:
    remote=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=30','28358lan',
        'powershell -NoProfile -EncodedCommand '+encoded],stdout=stdout,stderr=stderr)
    (control/'supervisor.json').write_text(json.dumps(dict(pid=os.getpid(),ssh_pid=remote.pid,queue='download,speed,transfer_bank,local_transfer_eval',
        run_ids=ids),indent=2),encoding='utf-8')
    for key in (('transfer_bank',) if a.transfer_only else ('download','speed','transfer_bank')):
        subprocess.run([sys.executable,str(scripts/'return_factorial_outputs.py'),'--runs-root',str(runs),'--source',ids[key],'--timeout-hours','4'],cwd=root,check=True)
        print(json.dumps(dict(returned=key,run_id=ids[key])),flush=True)
    exit_code=remote.wait(timeout=60)
    if exit_code:raise RuntimeError('Remote queue exit '+str(exit_code))
subprocess.run(['scp','-o','BatchMode=yes','28358lan:D:/coco_wire/models/yolo26s-seg.pt',str(root/'assets/models/coco_clean_20260911/yolo26s-seg.pt')],check=True)
source=runs/ids['transfer_bank'];out=runs/ids['transfer_eval'];ann=root/'assets/datasets/coco/annotations/instances_val2017.json'
models=runs/protocol['frozen_model_run']
command=[sys.executable,str(root/'shared/tools/research_runner/runner.py'),'--study',protocol['study_id'],'--run-id',ids['transfer_eval'],
    '--output',str(out),'--cwd',str(root),'--input',str(ann),'--input',str(source/'candidate_records.csv'),
    '--input',str(source/'predictions_official_zero.json'),'--input',str(source/'predictions_smooth_gated.json'),
    '--input',str(source/'predictions_global_0.25.json')]
for mode in ('area','shape','response'):command+=['--input',str(models/(mode+'.joblib'))]
for path in (scripts/'fit_risk_calibration.py',scripts/'continue_transfer_pipeline.py',study/'PROTOCOL_TRANSFER_V5.json'):
    command+=['--snapshot',str(path)]
command+=['--expect',str(out/'SUMMARY.json'),'--metrics',str(out/'SUMMARY.json'),'--',sys.executable,'-u',str(scripts/'fit_risk_calibration.py'),
    '--phase','evaluate','--input',str(source),'--annotations',str(ann),'--model-run',str(models),'--output',str(out)]
subprocess.run(command,cwd=root,check=True)
print('TRANSFER_PIPELINE_COMPLETED',flush=True)
