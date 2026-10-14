"""Collect completed E1 Runs, independent audit and original GT without modifying Runs."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'experiments/mask_readout_subspace_repulsion_20261015'
IDS=['RUN_REPULSION_SMOKE_561a93db573b4938a571d8056ff007c9','RUN_REPULSION_FULL_8b2e19a348d1476fbf05b64aabb10c7f']


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


assert not (STUDY/'.E1_REPULSION.lock').exists()
files=[]
for rid in IDS:
    run=STUDY/'runs'/rid
    state=json.loads((run/'run.json').read_text());assert state['execution_status']=='completed' and state['exit_code']==0
    for item in json.loads((run/'ARTIFACT_MANIFEST.json').read_text())['artifacts']:
        p=run/item['relative_path'];assert sha(p)==item['sha256'] and p.stat().st_size==item['size_bytes']
    files.extend(p for p in sorted(run.rglob('*')) if p.is_file())
files.extend([STUDY/'INDEPENDENT_LAPTOP_FULL.json',STUDY/'INDEPENDENT_LAPTOP_SMOKE.json',
              ROOT/'assets/datasets/coco_native_s0_20261013/annotations/instances_val2017.json'])
for p in files:assert p.exists(),p
manifest=STUDY/'TRANSFER_MANIFEST.json'
with manifest.open('x',encoding='utf8') as stream:
    json.dump(dict(files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),size_bytes=p.stat().st_size) for p in files]),stream,indent=2);stream.write('\n')
archive=ROOT.parent/'repulsion_results_8b2e19a3.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in files+[manifest]:z.write(p,p.relative_to(ROOT).as_posix())
print(json.dumps(dict(path=str(archive),sha256=sha(archive),size_bytes=archive.stat().st_size,files=len(files))))
