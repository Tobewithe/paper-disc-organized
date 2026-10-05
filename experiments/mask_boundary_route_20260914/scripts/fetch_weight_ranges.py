"""Bounded ranged download from the official release, checked against publisher digest."""
import argparse,concurrent.futures,hashlib,json,os,time
from pathlib import Path
import requests
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
url='https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt'
size=23467933;chunk=1024*1024;expected='3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4'
def fetch(i):
    start=i*chunk;end=min(size,(i+1)*chunk)-1;path=out/f'part_{i:03d}'
    for attempt in range(3):
        try:
            with requests.get(url+f'?part={i}',headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=(15,35),stream=True) as r:
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise RuntimeError('Server did not honor byte range')
                with path.open('wb') as f:
                    for data in r.iter_content(65536):f.write(data)
            assert path.stat().st_size==end-start+1
            print(json.dumps(dict(part=i,bytes=path.stat().st_size)),flush=True);return path
        except Exception as exc:
            if attempt==2:raise RuntimeError(f'Part {i} failed: {type(exc).__name__}') from None
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:parts=list(pool.map(fetch,range((size+chunk-1)//chunk)))
target=out/'yolo26s-seg.pt'
with target.open('wb') as f:
    for path in parts:f.write(path.read_bytes())
digest=hashlib.sha256(target.read_bytes()).hexdigest();assert digest==expected
result=dict(path=str(target),bytes=target.stat().st_size,sha256=digest,url=url,
    publisher_digest_source='https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26s-seg.pt')
temp=out/'SUMMARY.json.tmp';temp.write_text(json.dumps(result,indent=2));os.replace(temp,out/'SUMMARY.json');print(json.dumps(result),flush=True)
