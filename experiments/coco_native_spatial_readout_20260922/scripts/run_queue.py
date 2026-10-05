"""Finite train-three/evaluate-five queue; no test-set based winner selection."""
import json
from pathlib import Path
import subprocess
import sys
import uuid

root=Path(__file__).resolve().parent.parent;study=json.loads((root/'study.json').read_text())['study_id']
old=Path('/root/autodl-tmp/coco_mask_spatial_probe_20260916');previous=Path('/root/coco_spatial_calibration_mechanism_20260922')
bank=old/'runs/RUN_e83cf453bd6e47feb025ef4259bc8499/training_bank.pt';split=old/'runs/RUN_690c1580277a44679752ff727705f2e6/SPLIT.json'
features=previous/'runs/RUN_098ba724d6034a1eab6ae058362d2ba0/coefficient_features.pt'
geometry=Path('/root/coco_template_source_20260921/runs/RUN_143f4861823342c8afdca4251ee764d8/geometry.pt')
weights=old/'yolo26m-seg.pt';data=Path('/root/autodl-tmp/datasets/coco2017')
reference=previous/'runs/RUN_984bb9ff1cd2432bae2c2c08c3042c66/MATCHED_GT75.json'
assert not (root/'execution_ids.json').exists()
ids={name:'RUN_'+uuid.uuid4().hex for name in ['native_scalar','native_spatial','response_spatial','smoke','evaluate']}
(root/'execution_ids.json').write_text(json.dumps(ids,indent=2))

def execute(name,action,args,inputs,expected):
    out=root/'runs'/ids[name];script=root/'scripts/native_spatial_readout.py'
    cmd=[sys.executable,str(root/'runner.py'),'--study',study,'--run-id',ids[name],'--output',str(out),'--cwd',str(root)]
    for path in inputs:cmd+=['--input',str(path)]
    for path in [root/'PROTOCOL.json',*sorted((root/'scripts').glob('*.py'))]:cmd+=['--snapshot',str(path)]
    for file in expected:cmd+=['--expect',str(out/file)]
    cmd+=['--',sys.executable,'-u',str(script),action,*map(str,args),'--out',str(out)]
    (root/'queue_status.json').write_text(json.dumps({'status':'running','stage':name,'run_id':ids[name]}))
    subprocess.run(cmd,check=True);return out

try:
    trained={}
    for name in ['native_scalar','native_spatial','response_spatial']:
        trained[name]=str(execute(name,'train',['--mode',name,'--bank',bank,'--features',features,'--geometry',geometry,'--split',split],
            [bank,features,geometry,split],['epoch8.pt','SELECTION.json','COMPLETE.json']))
    config={'heads':trained,'native_checkpoint':str(previous/'runs/RUN_8ad0b76e6b9f456fac99b476f6cb35bf/epoch8.pt')}
    (root/'MODELS.json').write_text(json.dumps(config,indent=2));common=['--models',root/'MODELS.json','--data',data,'--weights',weights,'--reference',reference]
    inputs=[root/'MODELS.json',weights,reference,Path(config['native_checkpoint'])]+[Path(p)/'epoch8.pt' for p in trained.values()]
    execute('smoke','evaluate',[*common,'--smoke'],inputs,['COMPLETE.json'])
    result=execute('evaluate','evaluate',common,inputs,['COMPLETE.json','RESULTS.json','MATCHED_GT75.json'])
    (root/'queue_status.json').write_text(json.dumps({'status':'completed','result':str(result),'run_id':ids['evaluate']}))
except Exception as exc:
    status=json.loads((root/'queue_status.json').read_text());status.update(status='failed',error=repr(exc))
    (root/'queue_status.json').write_text(json.dumps(status));raise
