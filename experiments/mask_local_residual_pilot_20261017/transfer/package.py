"""Terminal-state archive including all Run inputs, checkpoints and failures."""
from pathlib import Path
import datetime,hashlib,json,socket,zipfile
S=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
state=read(S/'PROCESS_STATUS.json')
assert state['status'] in ('completed_and_verified','failed')
assert not list(S.glob('.*.lock'))
files=[p for p in sorted(S.rglob('*')) if p.is_file() and 'transfer' not in p.relative_to(S).parts and p.name!='TRANSFER_MANIFEST.json']
m=dict(host=socket.gethostname(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=state['status'],files=[dict(path=p.relative_to(S).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in files])
mp=S/'TRANSFER_MANIFEST.json'
with mp.open('x',encoding='utf8') as f:json.dump(m,f,indent=2)
archive=S/'transfer/LRP_ALL_ARTIFACTS.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
    for p in [*files,mp]:z.write(p,p.relative_to(S).as_posix())
r=dict(files=len(files),bytes=sum(p.stat().st_size for p in files),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,manifest_sha256=sha(mp))
with (S/'transfer/ARCHIVE_RECEIPT.json').open('x',encoding='utf8') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
