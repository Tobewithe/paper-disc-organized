import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

root=Path(__file__).resolve().parent.parent
assert not (root/'summary_ids.json').exists()
rid='RUN_'+uuid.uuid4().hex
(root/'summary_ids.json').write_text(json.dumps({'analysis':rid}))
try:
    deadline=time.monotonic()+7200
    (root/'summary_status.json').write_text(json.dumps({'status':'waiting_for_results','run_id':rid}))
    while True:
        statuses=[json.loads((root/n).read_text())['status'] for n in ['support_status.json','native_status.json','within_instance_status.json']]
        if 'failed' in statuses:raise RuntimeError('A source experiment failed; do not summarize incomplete evidence.')
        if all(v=='completed' for v in statuses):break
        if time.monotonic()>deadline:raise TimeoutError('Source results not ready in two hours.')
        time.sleep(10)
    out=root/'runs'/rid;script=root/'scripts/summarize_mechanism.py'
    template=Path('/root/coco_template_source_20260921/runs/RUN_9de7e2ec9f5643d3874bbd4422262def/ANALYSIS.json')
    command=[sys.executable,str(root/'runner.py'),'--study',json.loads((root/'study.json').read_text())['study_id'],
        '--run-id',rid,'--output',str(out),'--cwd',str(root),'--snapshot',str(script),'--input',str(template)]
    for keyfile in ['support_ids.json','native_ids.json','within_instance_ids.json']:
        command+=['--input',str(root/keyfile)]
        for source in json.loads((root/keyfile).read_text()).values():
            for name in ['RESULTS.json','COMPLETE.json']:
                p=root/'runs'/source/name
                if p.exists():command+=['--input',str(p)]
    for source in ['RUN_89e14901533a4493a9617a16414e68e6','RUN_c1b150dc71c54c3e9abcfe07d6da9303']:
        for name in ['SUMMARY.json','CALIBRATION.json','instances.jsonl']:command+=['--input',str(root/'runs'/source/name)]
    for name in ['RESULTS.json','TABLES.md','MECHANISM.png','COMPLETE.json']:command+=['--expect',str(out/name)]
    command+=['--',sys.executable,'-u',str(script),'--root',str(root),'--template-summary',str(template),'--out',str(out)]
    (root/'summary_status.json').write_text(json.dumps({'status':'running','run_id':rid}))
    subprocess.run(command,check=True)
    (root/'summary_status.json').write_text(json.dumps({'status':'completed','run_id':rid}))
except Exception as exc:
    (root/'summary_status.json').write_text(json.dumps({'status':'failed','run_id':rid,'error':repr(exc)}));raise
