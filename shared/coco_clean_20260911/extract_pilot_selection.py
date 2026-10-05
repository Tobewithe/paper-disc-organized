"""Extract only missing preregistered pilot images from the existing COCO archive."""
import argparse,json,shutil,zipfile
from pathlib import Path
from frozen_mechanism_probe import ROOT,write_json
ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path,required=True);a=ap.parse_args()
ids=set(json.loads(a.selection.read_text())['train']);names={f'train2017/{iid:012d}.jpg' for iid in ids};receipts=[]
with zipfile.ZipFile('/autodl-pub/data/COCO2017/train2017.zip') as z:
    entries=[e for e in z.infolist() if e.filename in names];assert len(entries)==len(ids)
    for e in entries:
        target=ROOT/'data/images'/e.filename
        if target.exists():assert target.stat().st_size==e.file_size;continue
        target.parent.mkdir(parents=True,exist_ok=True)
        with z.open(e) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
        assert target.stat().st_size==e.file_size;receipts.append(dict(file=e.filename,bytes=e.file_size,crc=e.CRC))
write_json(a.selection.parent/'EXTRACTION.json',dict(selected=len(ids),newly_extracted=len(receipts),files=receipts))
print(json.dumps(dict(selected=len(ids),newly_extracted=len(receipts))))
