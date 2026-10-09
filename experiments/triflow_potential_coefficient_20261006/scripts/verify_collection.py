"""Extract one local collection bundle and verify every source file hash."""
import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--run', type=Path, required=True)
args = ap.parse_args()
run = args.run.resolve()
with zipfile.ZipFile(run / 'collection.zip') as z:
    for entry in z.infolist():
        target = (run / entry.filename).resolve()
        assert target.is_relative_to(run), 'Archive member escapes Run'
    z.extractall(run)
verified = 0
for line in (run / 'manifest.sha256').read_text(encoding='utf-8').splitlines():
    digest, relative = line.split('  ', 1)
    p = run / relative
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    assert h.hexdigest() == digest, relative
    verified += 1
source_manifest = (run / 'manifest.sha256').read_bytes()
(run / '_source_manifest.sha256').write_bytes(source_manifest)
t = json.loads((run / 'transfer.json').read_text(encoding='utf-8'))
t.update(status='verified', verified_files=verified, local_directory=str(run),
         source_manifest_sha256=hashlib.sha256(source_manifest).hexdigest(),
         verified_at=datetime.now(timezone(timedelta(hours=8))).isoformat())
(run / 'transfer.json').write_text(json.dumps(t, indent=2), encoding='utf-8')
meta = run / 'run.json'
r = json.loads(meta.read_text(encoding='utf-8-sig'))
r.update(transfer_status='verified', artifact_status='available', local_directory=str(run))
meta.write_text(json.dumps(r, indent=2), encoding='utf-8')
# Keep the source hashes verbatim and a separate current local manifest after
# recording the verified transfer state in mutable metadata.
current = []
for line in source_manifest.decode('utf-8').splitlines():
    _, relative = line.split('  ', 1)
    p = run / relative
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    current.append(f'{h}  {relative}\n')
current.append(f'{hashlib.sha256(source_manifest).hexdigest()}  _source_manifest.sha256\n')
(run / 'manifest.sha256').write_text(''.join(current), encoding='utf-8')
print(json.dumps({'run_id': r['run_id'], 'verified_files': verified}), flush=True)
