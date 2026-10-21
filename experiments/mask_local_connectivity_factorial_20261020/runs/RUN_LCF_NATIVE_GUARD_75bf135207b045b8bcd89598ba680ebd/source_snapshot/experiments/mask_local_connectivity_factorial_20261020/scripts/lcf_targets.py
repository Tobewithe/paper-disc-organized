"""Supervision only: native input-grid GT and fixed connectivity anchor pairs."""
import math
import torch
from native_geometry import target_grid
from adapter_data import training_join
from connectivity_loss import make_spec

def prepare_targets(coco,cache,rows,config):
    anns=coco.imgToAnns.get(cache['image_id'],[]);join=training_join(rows,anns)
    normal=[a for a in anns if not a.get('iscrowd',0) and not a.get('ignore',0)]
    projected={a['id']:target_grid(coco.annToMask(a),cache).numpy().astype(bool) for a in normal}
    targets={rank:torch.from_numpy(projected[ann]).to(cache['proto'].device) for rank,ann in join.items()}
    specs={};h,w=cache['input_hw']
    for rank,ann in join.items():
        x0,y0,x1,y1=cache['boxes_input'][rank].cpu().tolist()
        x0=max(0,min(w,math.ceil(x0)));x1=max(0,min(w,math.ceil(x1)))
        y0=max(0,min(h,math.ceil(y0)));y1=max(0,min(h,math.ceil(y1)))
        if x1<=x0 or y1<=y0:continue
        gt=projected[ann][y0:y1,x0:x1]
        others=[m[y0:y1,x0:x1] for k,m in projected.items() if k!=ann]
        spec=make_spec(gt,others,cache['image_id'],rank,config['loss'])
        if spec['positive'] or spec['negative']:specs[rank]=((x0,y0,x1,y1),spec)
    return targets,specs,join
