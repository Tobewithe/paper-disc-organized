"""Sequential recorded runs; stops at first execution failure."""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
ids=json.loads((root/'execution_ids.json').read_text())
python=sys.executable

def run(label, entry, args, expect, inputs=()):
    rid=ids[label+'_run']; out=root/'runs'/rid
    command=[python,str(root/'runner.py'),'--study',ids['study_id'],'--run-id',rid,
             '--output',str(out),'--cwd',str(root),'--snapshot',str(root/'scripts'/entry),
             '--snapshot',str(root/'PROTOCOL.json'),'--snapshot',str(root/'scripts/prepare.py'),
             '--expect',str(out/expect)]
    for path in inputs:command.extend(['--input',str(path)])
    command.extend(['--',python,str(root/'scripts'/entry),*map(str,args),'--out',str(out)])
    (root/'queue_status.json').write_text(json.dumps({'stage':label,'run_id':rid,'status':'running'}))
    code=subprocess.call(command)
    if code:
        (root/'queue_status.json').write_text(json.dumps({'stage':label,'run_id':rid,'status':'failed','exit_code':code}))
        raise SystemExit(code)

if __name__ == '__main__':
    run('prep','prepare.py',['--root',root,'--data',root/'assets/datasets/coco'],'SUMMARY.json')
    run('raw','evaluate_raw.py',['--root',root,'--split','train2017','--image-ids',root/'explore_ids.json','--limit',500],
        'COMPLETE.json',[root/'explore_ids.json',root/'assets/models/coco_clean_20260911/yolo26m-seg.pt'])
    run('supervision','supervision_probe.py',['--root',root,'--raw',root/'runs'/ids['raw_run']],
        'SUMMARY.json',[root/'explore_ids.json',root/'LABEL_LEDGER.json',root/'runs'/ids['raw_run']/'COMPLETE.json'])
    (root/'queue_status.json').write_text(json.dumps({'stage':'complete','status':'completed'}))
