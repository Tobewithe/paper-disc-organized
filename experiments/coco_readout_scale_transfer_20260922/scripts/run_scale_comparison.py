"""Same-budget S-model screening after the independently prepared bank is complete."""
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

root=Path(__file__).resolve().parent.parent
assert not (root/'comparison_ids.json').exists()
ids={k:'RUN_'+uuid.uuid4().hex for k in ['native_scalar','native_spatial','native_finetune','smoke','evaluate']}
(root/'comparison_ids.json').write_text(json.dumps(ids,indent=2))
study=json.loads((root/'study.json').read_text())['study_id']
weights=Path('/root/project_assets/yolo26s-seg.pt')
data=Path('/root/autodl-tmp/datasets/coco2017')

def execute(name,script,arguments,inputs,expected):
    out=root/'runs'/ids[name]
    cmd=[sys.executable,str(root/'runner.py'),'--study',study,'--run-id',ids[name],
         '--output',str(out),'--cwd',str(root)]
    for path in inputs:cmd+=['--input',str(path)]
    for path in [root/'COMPARISON_PROTOCOL.json',*sorted((root/'scripts').glob('*.py'))]:cmd+=['--snapshot',str(path)]
    for name_expected in expected:cmd+=['--expect',str(out/name_expected)]
    cmd+=['--',sys.executable,'-u',str(root/'scripts'/script),*map(str,arguments),'--out',str(out)]
    (root/'comparison_status.json').write_text(json.dumps({'status':'running','stage':name,'run_id':ids[name]}))
    subprocess.run(cmd,check=True)
    return out

try:
    (root/'comparison_status.json').write_text(json.dumps({'status':'waiting_for_preparation'}))
    deadline=time.monotonic()+7200
    while True:
        state=json.loads((root/'queue_status.json').read_text())
        if state['status']=='failed':raise RuntimeError('S-model bank preparation failed.')
        if state['status']=='completed':break
        if time.monotonic()>deadline:raise TimeoutError('S-model bank preparation did not finish in two hours.')
        time.sleep(10)
    prep=root/'runs'/state['run_id']
    assert (prep/'COMPLETE.json').exists()
    bank=prep/'training_bank.pt';features=prep/'coefficient_features.pt'
    geometry=prep/'geometry.pt';split=prep/'SPLIT.json'
    common=['train','--bank',bank,'--features',features,'--split',split,'--seed','0']
    inputs=[bank,features,geometry,split]
    models={'heads':{}}
    for mode in ['native_scalar','native_spatial']:
        out=execute(mode,'native_spatial_readout.py',[*common,'--geometry',geometry,'--mode',mode],inputs,
                    ['epoch8.pt','SELECTION.json','COMPLETE.json'])
        models['heads'][mode]=str(out)
    out=execute('native_finetune','native_coefficient_scale_control.py',common,inputs,
                ['epoch8.pt','SELECTION.json','COMPLETE.json'])
    models['native_checkpoint']=str(out/'epoch8.pt')
    (root/'MODELS.json').write_text(json.dumps(models,indent=2))
    arguments=['evaluate','--models',root/'MODELS.json','--data',data,'--weights',weights]
    inputs=[weights,root/'MODELS.json',data/'annotations/instances_val2017.json',out/'epoch8.pt',out/'SELECTION.json']
    inputs += [Path(path)/f for path in models['heads'].values() for f in ['epoch8.pt','SELECTION.json']]
    execute('smoke','evaluate_scale.py',[*arguments,'--smoke'],inputs,['COMPLETE.json'])
    out=execute('evaluate','evaluate_scale.py',arguments,inputs,['COMPLETE.json','RESULTS.json','MATCHED_GT75.json'])
    (root/'comparison_status.json').write_text(json.dumps({'status':'completed','run_id':ids['evaluate'],'result':str(out)}))
except Exception as exc:
    state=json.loads((root/'comparison_status.json').read_text())
    state.update(status='failed',error=repr(exc));(root/'comparison_status.json').write_text(json.dumps(state));raise
