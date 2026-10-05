"""Laptop-only lossless support-file transport. No model computation."""
import argparse
from pathlib import Path
import hashlib
import json
import time
import zipfile
from server_transport import ROOT, put, ssh

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    files=[]
    source=Path(r'D:\coco_wire\py')
    for f in (source/'ultralytics').rglob('*'):
        if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc':
            files.append((f,'py/'+f.relative_to(source).as_posix()))
    files += [(Path(r'D:\coco_wire\models\yolo26m-seg.pt'),'models/yolo26m-seg.pt'),
       (Path(r'D:\coco_wire\runner.py'),'runner.py')]
    base=Path(r'D:\coco_wire\bgcr_native_20261002')
    for domain in ('train','val'):
        files.append((base/'data/annotations'/f'instances_{domain}2017.json',f'data/instances_{domain}2017.json'))
    for name in ('INDEX.json','CACHE_IDENTITY.json','COMPLETE.json','DATASET_READY.json'):
        files.append((base/'cache'/name,'source_metadata/'+name))
    files.append((base/'runs/RUN_976ee94db0ed49d894567e0fc0151224/PER_CANDIDATE.jsonl','data/BASELINE_PER_CANDIDATE.jsonl'))
    manifest={}
    archive=root/'SUPPORT.zip'
    with zipfile.ZipFile(str(archive)+'.tmp','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for f,name in files:
            if not f.is_file():raise FileNotFoundError(f)
            manifest[name]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size,'source':str(f)}
            z.write(f,name)
        z.writestr('SUPPORT_MANIFEST.json',json.dumps(manifest,indent=2))
    Path(str(archive)+'.tmp').replace(archive)
    put(archive,ROOT+'/SUPPORT.zip',timeout=1800)
    command="/root/miniconda3/bin/python - <<'PY'\nfrom pathlib import Path\nimport zipfile,json,hashlib\nr=Path('"+ROOT+"')\nwith zipfile.ZipFile(r/'SUPPORT.zip') as z:\n for m in z.infolist():\n  if not (r/m.filename).resolve().is_relative_to(r.resolve()):raise ValueError('unsafe archive path')\n z.extractall(r)\nd=json.loads((r/'SUPPORT_MANIFEST.json').read_text())\nfor name,v in d.items():\n assert hashlib.sha256((r/name).read_bytes()).hexdigest()==v['sha256'],name\nprint(json.dumps({'support_verified':len(d)}))\nPY"
    result=ssh(command,timeout=180)
    (root/'SUPPORT_TRANSFER.json').write_text(json.dumps({'status':'complete','files':len(files),'archive_bytes':archive.stat().st_size,'verified':json.loads(result),'completed_unix':time.time()},indent=2),encoding='utf-8')
    print(result,flush=True)

if __name__=='__main__':main()
