"""Post-run transfer only; no model, metric or scientific decision computation."""
from pathlib import Path
import hashlib
import json
import os
import psutil
import zipfile
from datetime import datetime,timezone

STUDY=Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p,d):
    with p.open('x',encoding='utf8') as f: json.dump(d,f,ensure_ascii=False,indent=2)

def main():
    assert read(STUDY/'AUTONOMOUS_STATUS.json')['phase']=='completed'
    assert read(STUDY/'INDEPENDENT_LAPTOP.json')['status']=='passed'
    state=read(STUDY/'study.json')
    for k in ('smoke_run_id','full_run_id'):
        assert read(STUDY/'runs'/state[k]/'run.json')['execution_status']=='completed'
    others=[p.info for p in psutil.process_iter(['pid','name','cmdline']) if p.info['pid']!=os.getpid() and 'python' in (p.info['name'] or '').lower()]
    assert not others,others
    assert not list(STUDY.glob('.*.lock'))
    write(STUDY/'EXECUTION_CLOSURE_LAPTOP.json',dict(at=datetime.now(timezone.utc).isoformat(),status='completed',other_python_processes=others,stage_lock_absent=True,launch_mechanism='Start-Process hidden; disconnect survival was not tested',initial_launch_receipt_independent_of_SSH='intent only; not a separately tested guarantee'))
    files=[];excluded=[]
    for p in sorted(STUDY.rglob('*')):
        if not p.is_file() or p.suffix=='.zip': continue
        row=dict(relative_path=p.relative_to(STUDY).as_posix(),sha256=digest(p),size_bytes=p.stat().st_size)
        if p.name in ('U_predictions.json','G_predictions.json'): excluded.append(row)
        else: files.append(row)
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),files=files,large_files_retained_on_laptop=excluded,scientific_sources_frozen_separately=True)
    write(STUDY/'TRANSFER_MANIFEST.json',manifest)
    archive=STUDY/'evidence.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in files: z.write(STUDY/row['relative_path'],row['relative_path'])
        z.write(STUDY/'TRANSFER_MANIFEST.json','TRANSFER_MANIFEST.json')
    receipt=dict(archive=archive.name,sha256=digest(archive),size_bytes=archive.stat().st_size,files=len(files)+1,omitted_predictions=len(excluded),transfer_manifest_sha256=digest(STUDY/'TRANSFER_MANIFEST.json'))
    write(STUDY/'TRANSFER_PACKAGE.json',receipt)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__': main()
