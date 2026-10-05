import argparse,hashlib,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--package-root',required=True);p.add_argument('--weights',required=True);p.add_argument('--output',required=True)
a=p.parse_args();sys.path.insert(0,a.package_root)
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics import YOLO,__version__
assert __version__=='8.4.100'
path=Path(a.weights);path.parent.mkdir(parents=True,exist_ok=True)
downloaded=Path(attempt_download_asset(path));assert downloaded.resolve()==path.resolve()
model=YOLO(path);assert model.task=='segment' and len(model.names)==80
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
result=dict(weights=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
    source='https://github.com/ultralytics/assets/releases/download/v8.4.0/'+path.name,ultralytics=__version__,task=model.task,classes=len(model.names))
t=out/'SUMMARY.json.tmp';t.write_text(json.dumps(result,indent=2));os.replace(t,out/'SUMMARY.json');print(json.dumps(result))
