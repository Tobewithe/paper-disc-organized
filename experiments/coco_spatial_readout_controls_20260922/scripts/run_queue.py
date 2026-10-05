"""Run bounded structural comparisons after the existing replication queue finishes."""
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

root=Path(__file__).resolve().parent.parent;study=json.loads((root/'study.json').read_text())['study_id']
old=Path('/root/autodl-tmp/coco_mask_spatial_probe_20260916')
mechanism=Path('/root/coco_spatial_calibration_mechanism_20260922');native=Path('/root/coco_native_spatial_readout_20260922')
bank=old/'runs/RUN_e83cf453bd6e47feb025ef4259bc8499/training_bank.pt';split=old/'runs/RUN_690c1580277a44679752ff727705f2e6/SPLIT.json'
features=mechanism/'runs/RUN_098ba724d6034a1eab6ae058362d2ba0/coefficient_features.pt'
geometry=Path('/root/coco_template_source_20260921/runs/RUN_143f4861823342c8afdca4251ee764d8/geometry.pt')
weights=old/'yolo26m-seg.pt';data=Path('/root/autodl-tmp/datasets/coco2017')
reference=mechanism/'runs/RUN_984bb9ff1cd2432bae2c2c08c3042c66/MATCHED_GT75.json'
arms=['global_scalar','global_spatial','native_coeff_mlp','native_quad_coeff_mlp']
assert not (root/'execution_ids.json').exists()
ids={k:'RUN_'+uuid.uuid4().hex for k in ['checks',*arms,'smoke','evaluate']};(root/'execution_ids.json').write_text(json.dumps(ids,indent=2))


def execute(name,action,args,inputs,expected):
    out=root/'runs'/ids[name]
    cmd=[sys.executable,str(root/'runner.py'),'--study',study,'--run-id',ids[name],'--output',str(out),'--cwd',str(root)]
    for path in inputs:cmd+=['--input',str(path)]
    for path in [root/'PROTOCOL.json',*sorted((root/'scripts').glob('*.py'))]:cmd+=['--snapshot',str(path)]
    for f in expected:cmd+=['--expect',str(out/f)]
    cmd+=['--',sys.executable,'-u',str(root/'scripts/structural_controls.py'),action,*map(str,args),'--out',str(out)]
    (root/'queue_status.json').write_text(json.dumps({'status':'running','stage':name,'run_id':ids[name]}))
    subprocess.run(cmd,check=True);return out


try:
    execute('checks','checks',[],[],['COMPLETE.json'])
    (root/'queue_status.json').write_text(json.dumps({'status':'waiting_for_replication'}));deadline=time.monotonic()+14400
    while True:
        state=json.loads((native/'replication_status.json').read_text())['status']
        if state=='failed':raise RuntimeError('Source replication failed; investigate before additionalGPUexperiments.')
        if state=='completed':break
        if time.monotonic()>deadline:raise TimeoutError('Source replication not complete within four hours.')
        time.sleep(10)
    models={}
    for mode in arms:
        models[mode]=str(execute(mode,'train',['--mode',mode,'--bank',bank,'--features',features,'--geometry',geometry,'--split',split],
            [bank,features,geometry,split],['epoch8.pt','SELECTION.json','COMPLETE.json']))
    (root/'MODELS.json').write_text(json.dumps(models,indent=2))
    common=['--models',root/'MODELS.json','--data',data,'--weights',weights,'--reference',reference]
    inputs=[root/'MODELS.json',weights,reference]+[Path(p)/f for p in models.values() for f in ['epoch8.pt','SELECTION.json']]
    execute('smoke','evaluate',[*common,'--smoke'],inputs,['COMPLETE.json'])
    result=execute('evaluate','evaluate',common,inputs,['COMPLETE.json','RESULTS.json','MATCHED_GT75.json'])
    (root/'queue_status.json').write_text(json.dumps({'status':'completed','run_id':ids['evaluate'],'result':str(result)}))
except Exception as exc:
    s=json.loads((root/'queue_status.json').read_text());s.update(status='failed',error=repr(exc));(root/'queue_status.json').write_text(json.dumps(s));raise
