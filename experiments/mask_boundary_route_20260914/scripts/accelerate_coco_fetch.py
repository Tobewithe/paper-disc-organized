"""Acquire remaining public COCO images with unique atomic temporary files.

The original preparation run remains responsible for verifying every selected file.
"""
import argparse,json,os,uuid,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed

p=argparse.ArgumentParser();p.add_argument('--annotations',required=True);p.add_argument('--images',required=True)
args=p.parse_args();root=Path(args.images)
images=json.loads(Path(args.annotations).read_text())['images']
def fetch(im):
    path=root/im['file_name']
    if path.exists():return 'cached'
    try:
        with urllib.request.urlopen(im['coco_url'],timeout=35) as r:data=r.read()
        assert data.startswith(b'\xff\xd8')
        temporary=path.with_suffix('.jpg.'+uuid.uuid4().hex+'.part')
        temporary.write_bytes(data);os.replace(temporary,path)
        return 'downloaded'
    except Exception:return 'deferred_to_preparation'
with ThreadPoolExecutor(max_workers=48) as pool:
    counts={}
    for i,f in enumerate(as_completed([pool.submit(fetch,im) for im in reversed(images)]),1):
        state=f.result();counts[state]=counts.get(state,0)+1
        if i%100==0:print(json.dumps(dict(done=i,counts=counts)),flush=True)
