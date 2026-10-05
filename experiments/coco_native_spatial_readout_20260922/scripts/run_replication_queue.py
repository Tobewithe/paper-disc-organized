"""Finite two-seed replication of the retained head and two mandatory controls."""
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

root=Path(__file__).resolve().parent.parent;study=json.loads((root/'study.json').read_text())['study_id']
old=Path('/root/autodl-tmp/coco_mask_spatial_probe_20260916')
previous=Path('/root/coco_spatial_calibration_mechanism_20260922')
bank=old/'runs/RUN_e83cf453bd6e47feb025ef4259bc8499/training_bank.pt'
split=old/'runs/RUN_690c1580277a44679752ff727705f2e6/SPLIT.json'
features=previous/'runs/RUN_098ba724d6034a1eab6ae058362d2ba0/coefficient_features.pt'
geometry=Path('/root/coco_template_source_20260921/runs/RUN_143f4861823342c8afdca4251ee764d8/geometry.pt')
weights=old/'yolo26m-seg.pt';data=Path('/root/autodl-tmp/datasets/coco2017')
reference=previous/'runs/RUN_984bb9ff1cd2432bae2c2c08c3042c66/MATCHED_GT75.json'
assert not (root/'replication_ids.json').exists()
ids={f'{name}_s{seed}':'RUN_'+uuid.uuid4().hex for seed in [1,2] for name in ['native_scalar','native_spatial','native_finetune','evaluate']}
(root/'replication_ids.json').write_text(json.dumps(ids,indent=2))


def execute(key,script,action,args,inputs,expected):
    out=root/'runs'/ids[key]
    cmd=[sys.executable,str(root/'runner.py'),'--study',study,'--run-id',ids[key],'--output',str(out),'--cwd',str(root)]
    for path in inputs:cmd+=['--input',str(path)]
    for path in [root/'REPLICATION_PROTOCOL.json',*sorted((root/'scripts').glob('*.py'))]:cmd+=['--snapshot',str(path)]
    for file in expected:cmd+=['--expect',str(out/file)]
    cmd+=['--',sys.executable,'-u',str(root/'scripts'/script),action,*map(str,args),'--out',str(out)]
    (root/'replication_status.json').write_text(json.dumps({'status':'running','stage':key,'run_id':ids[key]}))
    subprocess.run(cmd,check=True)
    return out


try:
    # Avoid GPU/CPU contention during the already scheduled timing comparison.
    (root/'replication_status.json').write_text(json.dumps({'status':'waiting_for_benchmark'}))
    deadline=time.monotonic()+3600
    while True:
        state=json.loads((root/'benchmark_status.json').read_text())['status']
        if state=='failed':raise RuntimeError('Timing failed; inspect before training more.')
        if state=='completed':break
        if time.monotonic()>deadline:raise TimeoutError('Timing did not complete within one hour.')
        time.sleep(10)
    evaluations={}
    for seed in [1,2]:
        heads={}
        for name in ['native_scalar','native_spatial']:
            heads[name]=str(execute(f'{name}_s{seed}','native_spatial_readout.py','train',
                ['--mode',name,'--seed',seed,'--bank',bank,'--features',features,'--geometry',geometry,'--split',split],
                [bank,features,geometry,split],['epoch8.pt','SELECTION.json','COMPLETE.json']))
        native=execute(f'native_finetune_s{seed}','native_coefficient_seed_control.py','train',
            ['--seed',seed,'--bank',bank,'--features',features,'--split',split],
            [bank,features,split],['epoch8.pt','SELECTION.json','COMPLETE.json'])
        config=root/f'MODELS_REPLICATION_S{seed}.json'
        config.write_text(json.dumps({'heads':heads,'native_checkpoint':str(native/'epoch8.pt')},indent=2))
        evaluations[str(seed)]=str(execute(f'evaluate_s{seed}','evaluate_replication.py','evaluate',
            ['--models',config,'--data',data,'--weights',weights,'--reference',reference],
            [config,weights,reference,native/'epoch8.pt',native/'SELECTION.json']+
            [Path(p)/f for p in heads.values() for f in ['epoch8.pt','SELECTION.json']],
            ['COMPLETE.json','RESULTS.json','MATCHED_GT75.json']))
    (root/'replication_status.json').write_text(json.dumps({'status':'completed','evaluations':evaluations}))
except Exception as exc:
    state=json.loads((root/'replication_status.json').read_text())
    state.update(status='failed',error=repr(exc));(root/'replication_status.json').write_text(json.dumps(state));raise
