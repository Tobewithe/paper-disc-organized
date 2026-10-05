"""Fetch from the official asset URL, streaming to a task-scoped partial file."""
import hashlib
import json
from pathlib import Path
import requests

root = Path(__file__).resolve().parent
url = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt'
dest = root/'weights/yolo26m-seg.pt'
dest.parent.mkdir(exist_ok=True)
temp = dest.with_suffix('.pt.download')
with requests.get(url, stream=True, timeout=(15, 30)) as response:
    response.raise_for_status()
    total=0
    with temp.open('wb') as f:
        for block in response.iter_content(1024*1024):
            f.write(block);total+=len(block)
            if total%(10*1024*1024)<1024*1024: print('downloaded',total,flush=True)
assert temp.stat().st_size > 10_000_000
temp.replace(dest)
receipt=dict(source=url,bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
(root/'weight_download.json').write_text(json.dumps(receipt,indent=2))
print('OFFICIAL_WEIGHTS_DOWNLOADED',json.dumps(receipt),flush=True)
