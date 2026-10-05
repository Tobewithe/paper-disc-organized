import argparse,json
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--official',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();idx=json.loads((a.official/'INDEX.json').read_text());ans={}
for group,images in idx.items():
    records=[]
    for item in images:
        iid=int(item['image_id'])
        x=torch.load(a.official/'images'/f'{iid:012d}.pt',map_location='cpu',weights_only=False,mmap=True)
        for m in x['rows']:
            records.append(dict(image_id=iid,annotation_id=int(m['annotation_id']),branch='one2one',raw_id=int(m['raw_id']),pyramid_level=int(m['level']),box_iou=float(m['box_iou'])))
        del x
    ans[group]=records
    print(group,len(records),flush=True)
a.out.write_text(json.dumps(ans))
