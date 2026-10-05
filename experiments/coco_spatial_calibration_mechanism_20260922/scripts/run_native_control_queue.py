"""Bounded native-readout control, evaluate after the support queue releases the GPU."""
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

root=Path(__file__).resolve().parent.parent
study=json.loads((root/'study.json').read_text())['study_id']
old=Path('/root/autodl-tmp/coco_mask_spatial_probe_20260916')
bank=old/'runs/RUN_e83cf453bd6e47feb025ef4259bc8499/training_bank.pt'
split=old/'runs/RUN_690c1580277a44679752ff727705f2e6/SPLIT.json'
weights=old/'yolo26m-seg.pt';data=Path('/root/autodl-tmp/datasets/coco2017')
reference=Path('/root/coco_template_source_20260921/runs/RUN_ec412bbfcfac49519b5f0cc27384e3e4/MATCHED_GT75.json')
assert not (root/'native_ids.json').exists()
ids={n:'RUN_'+uuid.uuid4().hex for n in ['prepare','train','evaluate']}
(root/'native_ids.json').write_text(json.dumps(ids,indent=2))
def execute(name,args,inputs,expected):
    out=root/'runs'/ids[name];script=root/'scripts/native_coefficient_control.py'
    command=[sys.executable,str(root/'runner.py'),'--study',study,'--run-id',ids[name],'--output',str(out),'--cwd',str(root)]
    for p in inputs:command+=['--input',str(p)]
    for p in [root/'NATIVE_CONTROL_PROTOCOL.json',script,*[root/'scripts'/n for n in ['learn_refinement.py','repair_refinement.py','component_seed_probe.py','shared_shape_head.py']]]:command+=['--snapshot',str(p)]
    for n in expected:command+=['--expect',str(out/n)]
    command+=['--',sys.executable,'-u',str(script),name,*map(str,args),'--out',str(out)]
    (root/'native_status.json').write_text(json.dumps({'status':'running','stage':name,'run_id':ids[name]}))
    subprocess.run(command,check=True);return out
try:
    prepare=execute('prepare',['--bank',bank,'--split',split,'--data',data,'--weights',weights],[bank,split,weights],['coefficient_features.pt','COMPLETE.json'])
    train=execute('train',['--bank',bank,'--split',split,'--features',prepare/'coefficient_features.pt'],[bank,split,prepare/'coefficient_features.pt'],['epoch8.pt','SELECTION.json','COMPLETE.json'])
    deadline=time.monotonic()+7200
    (root/'native_status.json').write_text(json.dumps({'status':'waiting_for_support_evaluations','trained_run_id':ids['train']}))
    while json.loads((root/'support_status.json').read_text())['status']=='running':
        if time.monotonic()>deadline:raise TimeoutError('GPU evaluation wait exceeded two hours; no process terminated.')
        time.sleep(10)
    result=execute('evaluate',['--data',data,'--weights',weights,'--train',train,'--reference',reference],
        [weights,train/'epoch8.pt',train/'SELECTION.json',reference],['RESULTS.json','MATCHED_GT75.json','COMPLETE.json'])
    (root/'native_status.json').write_text(json.dumps({'status':'completed','run_id':ids['evaluate'],'result':str(result)}))
except Exception as exc:
    prev=json.loads((root/'native_status.json').read_text())
    (root/'native_status.json').write_text(json.dumps({**prev,'status':'failed','error':repr(exc)}));raise
