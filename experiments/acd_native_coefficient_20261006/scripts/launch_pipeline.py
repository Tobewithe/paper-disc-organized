"""Detach only this experiment's queue from the SSH session, without a visible window."""
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path('D:/coco_wire/experiments/acd_native_coefficient_20261006')
out = root / 'runs' / 'RUN_ACD_PIPELINE_S0'
out.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', PYTHONPATH='D:/coco_wire/vendor_8.4.100',
           YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false')
cmd = [sys.executable, str(root / 'scripts' / 'run_pipeline.py'), '--root', str(root)]
with (out / 'stdout.log').open('w', encoding='utf-8') as log:
    process = subprocess.Popen(cmd, cwd=str(root), env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, close_fds=True,
                               creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW)
receipt = {'pid': process.pid, 'command': cmd, 'launch_mode': 'detached, no visible window'}
(out / 'LAUNCH.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt), flush=True)
