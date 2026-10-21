"""Transfer completed small Run artifacts; never copy or modify regional F caches."""
from pathlib import Path
import datetime
import hashlib
import json
import zipfile

study = Path(r'D:\coco_wire\baseline_native_20261013\experiments\mask_region_f_correspondence_20261017')
target = Path(r'D:\coco_wire\RFC_RECEIPTS_20261017.zip')
files = []
live_runs = []
for path in (study / 'runs').glob('*/run.json'):
    meta = json.loads(path.read_text(encoding='utf-8'))
    if meta['execution_status'] == 'running':
        live_runs.append(dict(run_id=meta['run_id'], scope='partial_live_snapshot',
                              observed_meta=meta))
        names = {'run.json', 'EVAL_CONTRACT.json', 'SOURCE_MANIFEST.json', 'IMAGE_ORDERS.json',
                 'ENVIRONMENT.json', 'ACTUAL_OPTIMIZER.json', 'initial.pt', 'execution.log',
                 'PROTOCOL.md', 'dirty.patch'}
        files.extend(p for p in path.parent.iterdir() if p.is_file() and p.name in names)
        files.extend(p for p in (path.parent / 'source_snapshot').glob('*') if p.is_file())
        continue
    if meta['execution_status'] not in ('completed', 'failed', 'interrupted', 'error'):
        continue
    files.extend(p for p in path.parent.rglob('*') if p.is_file() and 'regions' not in p.relative_to(path.parent).parts)
for pattern in ('*_POINTER.json', 'EXECUTION_STATUS.json', 'EXECUTION_FREEZE.json',
                'REAL_INPUT_REPLAY.json', 'REAL_SMOKE_VERIFICATION.json', 'TRAINING_PAIR_FREEZE.json',
                'DEPLOYMENT_RECEIPT.json', 'launch_*.json', 'launch_*.stdout.log', 'launch_*.stderr.log',
                'evaluation_*.json', 'evaluation_*.stdout.log', 'evaluation_*.stderr.log'):
    files.extend(study.glob(pattern))
manifest = {}
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(set(files)):
        name = p.relative_to(study).as_posix()
        data = p.read_bytes()
        archive.writestr(name, data)
        manifest[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    archive.writestr('TRANSFER_MANIFEST.json', json.dumps(dict(at=datetime.datetime.now().isoformat(),
                     excluded_regional_F=True, live_runs=live_runs, files=manifest), indent=2))
print(json.dumps(dict(files=len(manifest), bytes=target.stat().st_size,
                     archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
