"""Finite smoke then held-out probe, preserving each recorded execution."""
import json
from pathlib import Path
import subprocess
import sys
import uuid

root=Path(__file__).resolve().parent.parent
study=json.loads((root/'study.json').read_text())['study_id']
old=Path('/root/autodl-tmp/coco_mask_spatial_probe_20260916')
template=Path('/root/coco_template_source_20260921')
options={'data':'/root/autodl-tmp/datasets/coco2017','weights':str(old/'yolo26m-seg.pt'),
         'split':str(old/'runs/RUN_690c1580277a44679752ff727705f2e6/SPLIT.json'),
         'scalar':str(old/'runs/RUN_690c1580277a44679752ff727705f2e6/scalar_plain_epoch8.pt'),
         'shared':str(template/'runs/RUN_64d6c0aeacc44d28a4fc13e1832e6a03/epoch8.pt'),
         'template':str(template/'runs/RUN_a5a54b89359f407f905706037bab0728/fit_residual.json')}
assert not (root/'decomposition_ids.json').exists(),'Queue already launched; inspect existing records.'
ids={name:'RUN_'+uuid.uuid4().hex for name in ['smoke','heldout']}
(root/'decomposition_ids.json').write_text(json.dumps(ids,indent=2))
try:
    for name in ['smoke','heldout']:
        out=root/'runs'/ids[name]
        command=[sys.executable,str(root/'runner.py'),'--study',study,'--run-id',ids[name],
                 '--output',str(out),'--cwd',str(root)]
        for k,v in options.items():
            if k!='data':command+=['--input',v]
        command+=['--input',str(root/'runs/RUN_89e14901533a4493a9617a16414e68e6/SPLIT.json')]
        command+=['--input',options['data']+'/annotations/instances_train2017.json']
        for p in [root/'PROTOCOL.json',root/'DECOMPOSITION_PROTOCOL.json',*sorted((root/'scripts').glob('*.py'))]:command+=['--snapshot',str(p)]
        for p in ['SUMMARY.json','CALIBRATION.json','instances.jsonl','COMPLETE.json']:command+=['--expect',str(out/p)]
        command+=['--',sys.executable,'-u',str(root/'scripts/probe_mechanism.py')]
        for k,v in options.items():command+=['--'+k,v]
        command+=['--out',str(out),'--decompose','--sample-seed','20260923','--exclude-panel',str(root/'runs/RUN_89e14901533a4493a9617a16414e68e6/SPLIT.json')]
        if name=='smoke':command+=['--smoke']
        (root/'decomposition_status.json').write_text(json.dumps({'status':'running','stage':name,'run_id':ids[name]}))
        subprocess.run(command,check=True)
    (root/'decomposition_status.json').write_text(json.dumps({'status':'completed','run_id':ids['heldout']}))
except Exception as exc:
    previous=json.loads((root/'decomposition_status.json').read_text())
    (root/'decomposition_status.json').write_text(json.dumps({**previous,'status':'failed','error':repr(exc)}))
    raise
