"""Package immutable completed Run artifacts and the pinned corrected source for transfer."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'experiments/mask_readout_large_mask_gate_20261015'
IDS=['RUN_LARGE_MASK_SMOKE_f13cadc4ffdc4c6cb6a6692c917dac2d',
     'RUN_LARGE_MASK_FULL_6a2709ae1cd24429b29a3c6d6f1600b8']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


files=[]
assert not (STUDY/'.LARGE_MASK_GATE.lock').exists()
for rid in IDS:
    run=STUDY/'runs'/rid
    state=json.loads((run/'run.json').read_text())
    assert state['execution_status']=='completed' and state['exit_code']==0
    for record in json.loads((run/'ARTIFACT_MANIFEST.json').read_text())['artifacts']:
        p=run/record['relative_path']
        assert sha(p)==record['sha256'] and p.stat().st_size==record['size_bytes']
    files.extend(p for p in sorted(run.rglob('*')) if p.is_file())
parent=ROOT/'experiments/mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0/B_predictions.json'
assert sha(parent)=='3157da232b5bc7384b8f7f99927b74d40b6f870c6ebadd21ef853f4f1c237de8'
files.append(parent)
records=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),size_bytes=p.stat().st_size) for p in files]
manifest=STUDY/'TRANSFER_MANIFEST.json'
with manifest.open('x',encoding='utf8') as stream:json.dump(dict(files=records),stream,indent=2);stream.write('\n')
archive=ROOT.parent/'large_mask_results_6a2709ae.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in [*files,manifest]:z.write(p,p.relative_to(ROOT).as_posix())
print(json.dumps(dict(path=str(archive),sha256=sha(archive),size_bytes=archive.stat().st_size,
                      files=len(files),uncompressed_bytes=sum(p.stat().st_size for p in files))))
