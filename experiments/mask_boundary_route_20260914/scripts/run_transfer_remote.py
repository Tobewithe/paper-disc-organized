"""Finite sequential execution on notebook, held by one local SSH supervisor."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scripts',required=True);p.add_argument('--stage',choices=['all','speed','transfer'],default='all');a=p.parse_args();scripts=Path(a.scripts)
protocol=json.loads((scripts/'PROTOCOL_TRANSFER_V5.json').read_text(encoding='utf-8'));ids=protocol['run_ids']
root=Path('D:/coco_wire');runs=root/'runs/mask_boundary_route_20260914';weights=root/'models/yolo26s-seg.pt'
os.environ['PYTHONPATH']='D:/coco_wire/py';os.environ['PYTHONUNBUFFERED']='1'
def run(key,script,args,inputs=()):
    out=runs/ids[key]
    command=[sys.executable,str(root/'scripts/runner.py'),'--study',protocol['study_id'],'--run-id',ids[key],'--output',str(out),'--cwd',str(root)]
    for f in inputs:command+=['--input',str(f)]
    for name in (script,'mask_calibration.py','risk_calibration.py','portable_risk.py','PROTOCOL_TRANSFER_V5.json'):
        command+=['--snapshot',str(scripts/name)]
    command+=['--expect',str(out/'SUMMARY.json'),'--metrics',str(out/'SUMMARY.json'),'--',sys.executable,'-u',str(scripts/script),*map(str,args),'--output',str(out)]
    subprocess.run(command,cwd=root,check=True)
if a.stage=='all':run('download','download_transfer_weight.py',['--package-root',root/'py','--weights',weights])
if a.stage in ('all','speed'):run('speed','benchmark_risk_inference.py',['--package-root',root/'py','--weights','D:/mdoeldata/pigcv-task05/models/yolo26m-seg.pt',
    '--model',scripts/'response.json','--images',root/'data/images/val2017','--ids',scripts/'image_ids.json'],
    ['D:/mdoeldata/pigcv-task05/models/yolo26m-seg.pt',scripts/'response.json'])
if a.stage in ('all','transfer'):run('transfer_bank','decoder_calibration_experiment.py',['--package-root',root/'py','--images',root/'data/images/val2017',
    '--annotations',root/'data/annotations/instances_val2017.json','--weights',weights,'--protocol',scripts/'PROTOCOL_TRANSFER_V5.json',
    '--branch','one2one','--start','500','--limit','4500','--variants','official_zero,smooth_gated,global_0.25','--benchmark-repeats','0','--export-only'],
    [weights,root/'data/annotations/instances_val2017.json'])
