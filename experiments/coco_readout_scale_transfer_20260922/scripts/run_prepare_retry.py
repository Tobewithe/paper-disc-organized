import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

root=Path(__file__).resolve().parent.parent
archive=root/'attempts'/'dependency_failure'
archive.mkdir(parents=True,exist_ok=False)
for name in ['execution_ids.json','queue_status.json']:
    (archive/name).write_text((root/name).read_text())
rid='RUN_'+uuid.uuid4().hex;(root/'execution_ids.json').write_text(json.dumps({'prepare':rid}))
try:
    (root/'queue_status.json').write_text(json.dumps({'status':'waiting_for_structural_controls','run_id':rid}));deadline=time.monotonic()+21600
    dependency=Path('/root/coco_spatial_readout_controls_20260922/queue_status.json')
    while True:
        state=json.loads(dependency.read_text())['status']
        if state=='failed':raise RuntimeError('Structural comparison failed; inspect before scale transfer.')
        if state=='completed':break
        if time.monotonic()>deadline:raise TimeoutError('Structural comparison not complete within six hours.')
        time.sleep(10)
    out=root/'runs'/rid;weights=Path('/root/project_assets/yolo26s-seg.pt')
    split=Path('/root/autodl-tmp/coco_mask_spatial_probe_20260916/runs/RUN_690c1580277a44679752ff727705f2e6/SPLIT.json')
    template=Path('/root/coco_template_source_20260921/runs/RUN_a5a54b89359f407f905706037bab0728/fit_residual.json')
    cmd=[sys.executable,str(root/'runner.py'),'--study',json.loads((root/'study.json').read_text())['study_id'],
         '--run-id',rid,'--output',str(out),'--cwd',str(root),'--input',str(weights),'--input',str(split),'--input',str(template),
         '--input','/root/autodl-tmp/datasets/coco2017/annotations/instances_train2017.json']
    for p in [root/'PROTOCOL.json',*sorted((root/'scripts').glob('*.py'))]:cmd+=['--snapshot',str(p)]
    for name in ['training_bank.pt','coefficient_features.pt','geometry.pt','COMPLETE.json']:cmd+=['--expect',str(out/name)]
    cmd+=['--',sys.executable,'-u',str(root/'scripts/prepare_scale_bank.py'),'--out',str(out),
          '--data','/root/autodl-tmp/datasets/coco2017','--weights',str(weights),'--split',str(split),'--template',str(template)]
    (root/'queue_status.json').write_text(json.dumps({'status':'running','stage':'prepare','run_id':rid}));subprocess.run(cmd,check=True)
    (root/'queue_status.json').write_text(json.dumps({'status':'completed','stage':'preparation_only','run_id':rid,'result':str(out)}))
except Exception as exc:
    (root/'queue_status.json').write_text(json.dumps({'status':'failed','run_id':rid,'error':repr(exc)}));raise
