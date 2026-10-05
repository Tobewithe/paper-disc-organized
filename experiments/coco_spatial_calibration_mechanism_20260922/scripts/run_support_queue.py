"""Matched support-removal training, two normal full-COCO evaluations."""
import json
from pathlib import Path
import subprocess
import sys
import uuid

root=Path(__file__).resolve().parent.parent
study=json.loads((root/'study.json').read_text())['study_id']
old=Path('/root/autodl-tmp/coco_mask_spatial_probe_20260916')
template_root=Path('/root/coco_template_source_20260921')
data=Path('/root/autodl-tmp/datasets/coco2017')
bank=old/'runs/RUN_e83cf453bd6e47feb025ef4259bc8499/training_bank.pt'
split=old/'runs/RUN_690c1580277a44679752ff727705f2e6/SPLIT.json'
weights=old/'yolo26m-seg.pt'
template=template_root/'runs/RUN_a5a54b89359f407f905706037bab0728/analytic_center.json'
geometry=template_root/'runs/RUN_206796bc9e844d0c8a779c4203c2b400/geometry.pt'
scalar_reference=old/'runs/RUN_690c1580277a44679752ff727705f2e6'
reference=Path('/root/coco_spatial_layout_20260920/runs/RUN_01ed738ccd4344bbaf428027bc00434d')
assert not (root/'support_ids.json').exists()
names=['prepare']+[mode+'_'+phase for mode in ['gt_support','matched_random'] for phase in ['train','eval']]
ids={name:'RUN_'+uuid.uuid4().hex for name in names}
(root/'support_ids.json').write_text(json.dumps(ids,indent=2))
def execute(name,script,args,inputs,outputs):
    out=root/'runs'/ids[name]
    command=[sys.executable,str(root/'runner.py'),'--study',study,'--run-id',ids[name],'--output',str(out),'--cwd',str(root)]
    for p in inputs:command+=['--input',str(p)]
    for p in [root/'SUPPORT_TRAINING_PROTOCOL.json',*sorted((root/'scripts').glob('*.py'))]:command+=['--snapshot',str(p)]
    for p in outputs:command+=['--expect',str(out/p)]
    command+=['--',sys.executable,'-u',str(root/'scripts'/script),*map(str,args),'--out',str(out)]
    (root/'support_status.json').write_text(json.dumps({'status':'running','stage':name,'run_id':ids[name]}))
    subprocess.run(command,check=True)
    return out
try:
    support=execute('prepare','support_training.py',['prepare','--bank',bank,'--split',split,'--data',data,'--weights',weights],
        [bank,split,weights,data/'annotations/instances_train2017.json'],['supports.pt','COMPLETE.json'])
    results={}
    for mode in ['gt_support','matched_random']:
        train=execute(mode+'_train','support_training.py',['train','--bank',bank,'--split',split,'--template',template,
            '--geometry',geometry,'--supports',support/'supports.pt','--mode',mode],
            [bank,split,template,geometry,support/'supports.pt'],['epoch8.pt','SELECTION.json','COMPLETE.json'])
        result=execute(mode+'_eval','evaluate_support.py',['--train',train,'--template',template,'--data',data,'--weights',weights,
            '--reference',reference,'--scalar-reference',scalar_reference,'--fixed-alpha','0.5'],
            [train/'epoch8.pt',train/'SELECTION.json',template,weights,reference/'RESULTS.json',reference/'MATCHED_GT75.json',
             scalar_reference/'RESULTS.json',scalar_reference/'MATCHED_GT75.json'],['RESULTS.json','MATCHED_GT75.json','COMPLETE.json'])
        results[mode]=str(result)
    (root/'support_status.json').write_text(json.dumps({'status':'completed','evaluations':results}))
except Exception as exc:
    previous=json.loads((root/'support_status.json').read_text())
    (root/'support_status.json').write_text(json.dumps({**previous,'status':'failed','error':repr(exc)}))
    raise
