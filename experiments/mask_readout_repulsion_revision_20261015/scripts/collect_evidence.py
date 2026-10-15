"""Package immutable E1-R2 evidence after independent verification; never edit Runs."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'experiments/mask_readout_repulsion_revision_20261015'
IDS=['RUN_REVISION_SMOKE_d6d0561a96224035a10ee57bbcf74684','RUN_REVISION_FULL_77f26718c91c4f908cb77610a7803fc5']


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


assert not (STUDY/'.E1_R2.lock').exists()
files=[]
for rid in IDS:
    run=STUDY/'runs'/rid;state=json.loads((run/'run.json').read_text())
    assert state['execution_status']=='completed' and state['exit_code']==0
    for item in json.loads((run/'ARTIFACT_MANIFEST.json').read_text())['artifacts']:
        p=run/item['relative_path'];assert sha(p)==item['sha256'] and p.stat().st_size==item['size_bytes']
    files.extend(p for p in sorted(run.rglob('*')) if p.is_file())
files.extend(STUDY/p for p in ('INDEPENDENT_LAPTOP_FULL.json','INDEPENDENT_LAPTOP_SMOKE.json','LAUNCH_FULL.json','LAUNCH_SMOKE.json'))
files.extend(sorted(STUDY.glob('*launcher*.log')))
manifest=STUDY/'TRANSFER_MANIFEST.json'
with manifest.open('x',encoding='utf8') as stream:
    json.dump(dict(files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),size_bytes=p.stat().st_size) for p in files]),stream,indent=2);stream.write('\n')
archive=ROOT.parent/'repulsion_revision_77f26718.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in files+[manifest]:z.write(p,p.relative_to(ROOT).as_posix())
print(json.dumps(dict(path=str(archive),sha256=sha(archive),size_bytes=archive.stat().st_size,files=len(files))),flush=True)
