"""Snapshot this Study after execution has ended; never read unrelated tasks."""
from pathlib import Path
import hashlib,json,zipfile,datetime
s=Path(__file__).resolve().parents[1]
if (s/'.RSM.lock').exists():raise RuntimeError('own execution still locked')
runs=[]
for p in (s/'runs').glob('*/run.json'):
    meta=json.loads(p.read_text(encoding='utf8'))
    if meta['execution_status'] not in ['completed','error','interrupted']:raise RuntimeError('unfinished run '+p.parent.name)
    runs.append({'run_id':p.parent.name,'execution_status':meta['execution_status'],'scientific_status':meta['scientific_status']})
excluded={'TRANSFER_FINAL_SOURCE.json','CLOSURE_RECEIPTS.zip'}
files=[p for p in s.rglob('*') if p.is_file() and p.suffix not in ['.zip','.pyc'] and '__pycache__' not in p.parts and p.name not in excluded]
records=[{'path':p.relative_to(s).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size} for p in sorted(files)]
manifest=s/'TRANSFER_FINAL_SOURCE.json'
if manifest.exists():raise FileExistsError('final transfer snapshot exists')
manifest.write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':runs,'files':records},indent=2)+'\n',encoding='utf8')
with zipfile.ZipFile(s/'CLOSURE_RECEIPTS.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files)+[manifest]:z.write(p,p.relative_to(s).as_posix())
print(json.dumps({'files':len(files),'runs':runs,'zip_bytes':(s/'CLOSURE_RECEIPTS.zip').stat().st_size,'zip_sha256':hashlib.sha256((s/'CLOSURE_RECEIPTS.zip').read_bytes()).hexdigest()}))
