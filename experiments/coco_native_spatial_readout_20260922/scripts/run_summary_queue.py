import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

root=Path(__file__).resolve().parent.parent;assert not(root/'summary_ids.json').exists()
rid='RUN_'+uuid.uuid4().hex;(root/'summary_ids.json').write_text(json.dumps({'summary':rid}))
try:
    (root/'summary_status.json').write_text(json.dumps({'status':'waiting_for_sources','run_id':rid}));deadline=time.monotonic()+14400
    while True:
        states=[json.loads((root/(name+'_status.json')).read_text())['status'] for name in ['replication','boundary']]
        if 'failed' in states:raise RuntimeError('A source queue failed; no summary of incomplete evidence.')
        if all(v=='completed' for v in states):break
        if time.monotonic()>deadline:raise TimeoutError('Source results not complete within four hours.')
        time.sleep(10)
    initial=Path('/root/coco_spatial_calibration_mechanism_20260922/runs/RUN_984bb9ff1cd2432bae2c2c08c3042c66')
    inputs=[Path('/root/autodl-tmp/datasets/coco2017/annotations/instances_val2017.json'),
        Path('/root/coco_template_source_20260921/runs/RUN_9de7e2ec9f5643d3874bbd4422262def/ANALYSIS.json'),
        root/'replication_ids.json',root/'boundary_retry_ids.json',initial/'RESULTS.json',initial/'MATCHED_GT75.json']
    folders=[root/'runs/RUN_1bb359da7b4942b7a7ab3f0be194ff49']
    ids=json.loads((root/'replication_ids.json').read_text())
    folders.extend(root/'runs'/ids[f'evaluate_s{seed}'] for seed in [1,2])
    for folder in folders:inputs.extend([folder/'RESULTS.json',folder/'MATCHED_GT75.json',folder/'run.json'])
    boundary=root/'runs'/json.loads((root/'boundary_retry_ids.json').read_text())['boundary_evaluation']
    inputs.extend([boundary/'RESULTS.json',boundary/'run.json']);out=root/'runs'/rid
    cmd=[sys.executable,str(root/'runner.py'),'--study',json.loads((root/'study.json').read_text())['study_id'],
         '--run-id',rid,'--output',str(out),'--cwd',str(root),'--snapshot',str(root/'scripts/summarize_replicates.py')]
    for p in inputs:cmd+=['--input',str(p)]
    for name in ['RESULTS.json','TABLES.md','REPLICATION_AND_BOUNDARY.png','REPLICATION_AND_BOUNDARY.pdf','COMPLETE.json']:cmd+=['--expect',str(out/name)]
    cmd+=['--',sys.executable,'-u',str(root/'scripts/summarize_replicates.py'),'--root',str(root),'--out',str(out),'--annotations',str(inputs[0])]
    (root/'summary_status.json').write_text(json.dumps({'status':'running','run_id':rid}));subprocess.run(cmd,check=True)
    (root/'summary_status.json').write_text(json.dumps({'status':'completed','run_id':rid}))
except Exception as exc:
    (root/'summary_status.json').write_text(json.dumps({'status':'failed','run_id':rid,'error':repr(exc)}));raise
