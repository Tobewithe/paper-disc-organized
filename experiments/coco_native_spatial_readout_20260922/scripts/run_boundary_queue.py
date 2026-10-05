import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

root=Path(__file__).resolve().parent.parent
assert not (root/'boundary_ids.json').exists()
rid='RUN_'+uuid.uuid4().hex;(root/'boundary_ids.json').write_text(json.dumps({'boundary_evaluation':rid}))
try:
    (root/'boundary_status.json').write_text(json.dumps({'status':'waiting_for_benchmark','run_id':rid}))
    deadline=time.monotonic()+10800
    while True:
        state=json.loads((root/'benchmark_status.json').read_text())['status']
        if state=='failed':raise RuntimeError('Timing run failed; inspect before launching next queue.')
        if state=='completed':break
        if time.monotonic()>deadline:raise TimeoutError('Prior queue not complete within three hours.')
        time.sleep(10)
    out=root/'runs'/rid;vendor=root/'vendor/boundary_iou_api';config=root/'BOUNDARY_MODELS.json'
    cmd=[sys.executable,str(root/'runner.py'),'--study',json.loads((root/'study.json').read_text())['study_id'],
         '--run-id',rid,'--output',str(out),'--cwd',str(root),'--input',str(config),
         '--input','/root/autodl-tmp/datasets/coco2017/annotations/instances_val2017.json',
         '--snapshot',str(root/'BOUNDARY_PROTOCOL.json'),'--snapshot',str(root/'scripts/boundary_evaluation.py')]
    for source in json.loads(config.read_text()).values():cmd+=['--input',source]
    for f in vendor.rglob('*'):
        if f.is_file() and '__pycache__' not in str(f):cmd+=['--snapshot',str(f)]
    cmd+=['--expect',str(out/'COMPLETE.json'),'--expect',str(out/'RESULTS.json'),'--',
          sys.executable,'-u',str(root/'scripts/boundary_evaluation.py'),'--out',str(out),
          '--vendor',str(vendor),'--models',str(config),
          '--annotations','/root/autodl-tmp/datasets/coco2017/annotations/instances_val2017.json']
    (root/'boundary_status.json').write_text(json.dumps({'status':'running','run_id':rid}))
    subprocess.run(cmd,check=True)
    (root/'boundary_status.json').write_text(json.dumps({'status':'completed','run_id':rid}))
except Exception as exc:
    (root/'boundary_status.json').write_text(json.dumps({'status':'failed','run_id':rid,'error':repr(exc)}));raise
