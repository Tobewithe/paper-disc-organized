import os,sys,json,hashlib,subprocess,torch
from pathlib import Path
sys.path.insert(0,'D:/coco_wire/vendor_8.4.100')
import ultralytics
root=Path('D:/coco_wire/data/official_tal_affine_20260930/runs/official_cache')
print('runtime',ultralytics.__version__,torch.__version__,torch.cuda.get_device_name(0))
for n in ['CONVERSION.json','COMPLETE.json','INDEX.json']:
 p=root/n
 x=json.loads(p.read_text()); print(n,str(x)[:5000])
p=Path('D:/coco_wire/models/yolo26m-seg.pt')
print('weights_sha256',hashlib.sha256(p.read_bytes()).hexdigest())
from ultralytics import YOLO
m=YOLO(str(p)); print('train_args',m.ckpt.get('train_args')); print('head',len(m.model.model)-1,type(m.model.model[-1]).__name__)
for mod in ['pycocotools','scipy','polars','psutil']:
 try: __import__(mod);print('dep',mod,'available')
 except ImportError: print('dep',mod,'missing')
