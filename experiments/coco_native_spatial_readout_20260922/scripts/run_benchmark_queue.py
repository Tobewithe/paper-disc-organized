import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

root=Path(__file__).resolve().parent.parent;assert not (root/'benchmark_ids.json').exists()
rid='RUN_'+uuid.uuid4().hex;(root/'benchmark_ids.json').write_text(json.dumps({'benchmark':rid}))
try:
    (root/'benchmark_status.json').write_text(json.dumps({'status':'waiting_for_evaluation','run_id':rid}));deadline=time.monotonic()+7200
    while True:
        status=json.loads((root/'queue_status.json').read_text())['status']
        if status=='failed':raise RuntimeError('Source queue failed; benchmark not launched.')
        if status=='completed':break
        if time.monotonic()>deadline:raise TimeoutError('Source queue not complete within two hours.')
        time.sleep(10)
    out=root/'runs'/rid;script=root/'scripts/benchmark_readout.py'
    weights=Path('/root/autodl-tmp/coco_mask_spatial_probe_20260916/yolo26m-seg.pt')
    roi=Path('/root/coco_template_source_20260921/runs/RUN_64d6c0aeacc44d28a4fc13e1832e6a03/epoch8.pt')
    cmd=[sys.executable,str(root/'runner.py'),'--study',json.loads((root/'study.json').read_text())['study_id'],
        '--run-id',rid,'--output',str(out),'--cwd',str(root),'--input',str(weights),'--input',str(roi),'--input',str(root/'MODELS.json')]
    for f in (root/'scripts').glob('*.py'):cmd+=['--snapshot',str(f)]
    cmd+=['--expect',str(out/'COMPLETE.json'),'--expect',str(out/'RESULTS.json'),'--',sys.executable,'-u',str(script),
        '--out',str(out),'--data','/root/autodl-tmp/datasets/coco2017','--weights',str(weights),'--models',str(root/'MODELS.json'),'--roi-checkpoint',str(roi)]
    (root/'benchmark_status.json').write_text(json.dumps({'status':'running','run_id':rid}));subprocess.run(cmd,check=True)
    (root/'benchmark_status.json').write_text(json.dumps({'status':'completed','run_id':rid}))
except Exception as exc:
    (root/'benchmark_status.json').write_text(json.dumps({'status':'failed','run_id':rid,'error':repr(exc)}));raise
