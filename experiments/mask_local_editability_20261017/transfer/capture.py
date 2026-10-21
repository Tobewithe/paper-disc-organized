"""Closeout-only packaging, after the registered computation has stopped."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, socket, zipfile

study = Path(__file__).resolve().parents[1]
status = study/'AUTONOMOUS_STATUS_V2.json'
if not status.exists():
    status = study/'AUTONOMOUS_STATUS.json'
state = json.loads(status.read_text(encoding='utf8'))
if state['status'] not in ('completed_and_independently_verified', 'failed'):
    raise RuntimeError('Execution is not terminal; do not package mutable artifacts')
if (study/'.AUTONOMOUS_OWNER.guard').exists() or list(study.glob('.*.lock')):
    raise RuntimeError('Execution ownership still active')
files = [p for p in sorted(study.rglob('*')) if p.is_file() and
         'transfer' not in p.relative_to(study).parts and p.name not in ('TRANSFER_MANIFEST.json',)]
def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()
manifest = dict(captured_at=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
                execution_status=state['status'],
                files=[dict(path=p.relative_to(study).as_posix(), size=p.stat().st_size, sha256=digest(p)) for p in files])
mp = study/'TRANSFER_MANIFEST.json'
with mp.open('x',encoding='utf8') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
archive = study/'transfer/LEA_ALL_ARTIFACTS.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for p in [*files, mp]:
        z.write(p,p.relative_to(study).as_posix())
receipt = dict(archive=archive.name, archive_sha256=digest(archive), archive_bytes=archive.stat().st_size,
               files=len(files), bytes=sum(r['size'] for r in manifest['files']),
               manifest_sha256=digest(mp))
(study/'transfer/ARCHIVE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print(json.dumps(receipt))
