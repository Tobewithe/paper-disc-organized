import sys, json
import torch
from pathlib import Path
p=Path(r'D:\coco_wire\data\official_tal_affine_20260930\runs\official_cache')
idx=json.loads((p/'INDEX.json').read_text(encoding='utf-8'))
iid=int(idx['fit'][0]['image_id'])
x=torch.load(p/'images'/f'{iid:012d}.pt',map_location='cpu',weights_only=False)
for k,v in x.items():
    print(k, (tuple(v.shape),str(v.dtype)) if hasattr(v,'shape') else type(v).__name__, flush=True)
print('image',iid,'rows',len(x['rows']),'example',x['rows'][0],flush=True)
