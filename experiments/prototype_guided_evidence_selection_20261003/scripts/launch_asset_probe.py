"""Laptop-only CPU packing preflight, wrapped in the research runner."""
from pathlib import Path
import json
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
ids = json.loads((root/'RUN_IDS.json').read_text(encoding='utf-8-sig'))
rid = ids['runs']['asset_probe']
out = root/'runs'/rid
if (out/'run.json').exists():
    raise RuntimeError('Existing Run: refusing duplicate launch')
os.environ.update(PYTHONPATH=r'D:\coco_wire\py', PYTHONUTF8='1', CUDA_VISIBLE_DEVICES='')
python = str(Path(sys.executable).with_name('python.exe'))
command = [python, r'D:\coco_wire\runner.py', '--study', ids['study_id'], '--run-id', rid,
           '--cwd', str(root), '--output', str(out), '--input', str(root/'RUN_CONFIG.json'),
           '--input', str(root/'SPLIT.json')]
for item in [root/'PROTOCOL.md', root/'RUN_CONFIG.json', root/'scripts/prepare_transfer_assets.py']:
    command += ['--snapshot', str(item)]
command += ['--expect', str(out/'COMPLETE.json'), '--', python, '-u',
            str(root/'scripts/prepare_transfer_assets.py'), '--config', str(root/'RUN_CONFIG.json'),
            '--out', str(out), '--assets', str(root/'assets'), '--limit-per-split', '2']
with (root/'asset_probe_launcher.log').open('w', encoding='utf-8') as log:
    result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
raise SystemExit(result.returncode)
