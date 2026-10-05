import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

root=Path(__file__).resolve().parent.parent
assert not (root/'within_instance_ids.json').exists()
rid='RUN_'+uuid.uuid4().hex
(root/'within_instance_ids.json').write_text(json.dumps({'evaluate':rid}))
deadline=time.monotonic()+7200
try:
    (root/'within_instance_status.json').write_text(json.dumps({'status':'waiting_for_native_evaluation','run_id':rid}))
    while json.loads((root/'native_status.json').read_text())['status'] not in ['completed','failed']:
        if time.monotonic()>deadline:raise TimeoutError('Wait exceeded two hours; no process terminated.')
        time.sleep(10)
    out=root/'runs'/rid;script=root/'scripts/within_instance_reliability.py'
    data=Path('/root/autodl-tmp/datasets/coco2017');weights=Path('/root/autodl-tmp/coco_mask_spatial_probe_20260916/yolo26m-seg.pt')
    source=root/'runs/RUN_c1b150dc71c54c3e9abcfe07d6da9303/instances.jsonl'
    command=[sys.executable,str(root/'runner.py'),'--study',json.loads((root/'study.json').read_text())['study_id'],
        '--run-id',rid,'--output',str(out),'--cwd',str(root),'--input',str(source),'--input',str(weights)]
    for p in [root/'WITHIN_INSTANCE_PROTOCOL.json',script,*[root/'scripts'/n for n in ['native_coefficient_control.py','learn_refinement.py','repair_refinement.py','component_seed_probe.py','shared_shape_head.py']]]:
        command+=['--snapshot',str(p)]
    for name in ['RESULTS.json','COMPLETE.json','instances.jsonl']:command+=['--expect',str(out/name)]
    command+=['--',sys.executable,'-u',str(script),'--data',str(data),'--weights',str(weights),'--instances',str(source),'--out',str(out)]
    (root/'within_instance_status.json').write_text(json.dumps({'status':'running','run_id':rid}))
    subprocess.run(command,check=True)
    (root/'within_instance_status.json').write_text(json.dumps({'status':'completed','run_id':rid}))
except Exception as exc:
    (root/'within_instance_status.json').write_text(json.dumps({'status':'failed','run_id':rid,'error':repr(exc)}));raise
