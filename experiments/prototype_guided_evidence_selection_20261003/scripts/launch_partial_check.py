"""One disposable six-image server replay check while assets are in transit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
ids=json.loads((root/'RUN_IDS.json').read_text())
splits=json.loads((root/'SPLIT.json').read_text())
image_ids=[i for group in ('fit','dev','val') for i in splits[group][:2]]
rid=ids['runs']['replay_partial']
out=root/'runs'/rid
claim=root/'PARTIAL_CHECK.claim'
fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
os.write(fd,str(os.getpid()).encode());os.close(fd)
deadline=time.time()+6*3600
while not ((root/'SUPPORT_RELAY_RECEIPT.json').is_file()
           and (root/'assets/MIGRATION_IDENTITY.json').is_file()
           and all((root/'assets/images'/f'{i:012d}.meta.json').is_file() for i in image_ids)):
    if time.time()>deadline:raise TimeoutError('Partial data arrival timeout; no model training started')
    time.sleep(20)
if (out/'run.json').exists():raise RuntimeError('Partial replay already registered')
os.environ.update(PYTHONPATH=str(root/'py'),PYTHONUTF8='1')
command=[sys.executable,str(root/'runner.py'),'--study',ids['study_id'],'--run-id',rid,
         '--cwd',str(root),'--output',str(out),'--input',str(root/'RUN_CONFIG.json'),
         '--input',str(root/'assets/MIGRATION_IDENTITY.json')]
for name in ('replay_smoke.py','asset_runtime.py'):
    command += ['--snapshot',str(root/'scripts'/name)]
command += ['--expect',str(out/'COMPLETE.json'),'--',sys.executable,'-u',str(root/'scripts/replay_smoke.py'),
            '--config',str(root/'RUN_CONFIG.json'),'--out',str(out),'--limit-per-split','2']
subprocess.run(command,check=True,cwd=root,env=os.environ.copy(),stdin=subprocess.DEVNULL)
