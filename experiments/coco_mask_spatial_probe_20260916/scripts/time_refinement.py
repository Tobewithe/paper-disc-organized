"""Matched eager FP32 pipeline timing; excludes image I/O and RLE encoding."""
import argparse
import json
from pathlib import Path
import random
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from component_seed_probe import image_correct, setup, original


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for k in ('root','data','weights','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True);setup()
    ref=a.root/'runs/RUN_690c1580277a44679752ff727705f2e6'
    ids=json.loads((ref/'SPLIT.json').read_text())['val_ids'];random.Random(20260916).shuffle(ids)
    warm_ids,ids=ids[:5],ids[5:55]
    coco=json.loads((a.data/'annotations/instances_val2017.json').read_text())
    files={r['id']:r['file_name'] for r in coco['images']}
    sources={'scalar':ref/'scalar_plain_epoch8.pt','local4':a.root/'runs/RUN_8217274af2db49ffa6fb8d6725f8496b/epoch8.pt',
             'coeff_local4':ref/'coeff_local4_plain_epoch8.pt'}
    nets={}
    for mode,source in sources.items():
        net=original.Refiner(mode).cuda().eval();net.load_state_dict(torch.load(source,weights_only=False)['state_dict']);nets[mode]=net
    model=YOLO(str(a.weights)).model.cuda().float().eval();head=model.model[-1];assert head.end2end
    transform=LetterBox((640,640),auto=True,stride=32)
    def prepare(image_id):
        im=cv2.imread(str(a.data/'images/val2017'/files[image_id]));assert im is not None
        params=transform.get_params({'img':im});res=transform.apply_image({'img':im},params)['img']
        inp=torch.from_numpy(np.ascontiguousarray(res[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255
        return inp,im.shape[:2]
    def once(inp,orig,mode):
        torch.cuda.synchronize();start=time.perf_counter()
        _,raw=model(inp);p=raw['one2one'];torch.cuda.synchronize();forward=time.perf_counter()
        shape=tuple(inp.shape[2:]);proto=p['proto'][0];allc=p['mask_coefficient'][0].T
        allboxes=head._get_decode_boxes(p)[0].T
        ts,tc,ti=head.get_topk_index(p['scores'].permute(0,2,1).sigmoid(),300)
        ids=ti[0,ts[0,:,0]>.001,0];boxes=allboxes[ids];coeff=allc[ids]
        low=(allc@proto.flatten(1)).reshape(-1,*proto.shape[-2:])
        scale=proto.square().mean((1,2)).sqrt().clamp(min=.1) if mode else None
        count=0
        for first in range(0,len(boxes),16):
            sl=slice(first,first+16);bb=boxes[sl]
            base=F.interpolate(low[ids[sl]][None],shape,mode='bilinear',align_corners=False)[0]
            z=base
            if mode:
                x,_,_=original.features(proto,coeff[sl],bb,shape)
                z=image_correct(base,proto,scale,bb,shape,nets[mode](x)*.5,mode)
            masks=ops.scale_masks(ops.crop_mask(z.clone(),bb).gt(0).byte()[None],orig)[0].byte().cpu().numpy()
            count+=len(masks)
        torch.cuda.synchronize();end=time.perf_counter()
        return dict(total_ms=1000*(end-start),forward_ms=1000*(forward-start),mask_ms=1000*(end-forward),predictions=count)
    modes=[None,'scalar','local4','coeff_local4']
    for image_id in warm_ids:
        inp,orig=prepare(image_id)
        for mode in modes:once(inp,orig,mode)
    rows=[]
    for repeat in range(3):
        for n,image_id in enumerate(ids):
            inp,orig=prepare(image_id);offset=(n+repeat)%4;order=modes[offset:]+modes[:offset]
            for mode in order:rows.append(dict(image_id=image_id,repeat=repeat,mode=mode or 'baseline',**once(inp,orig,mode)))
        print(json.dumps({'repeat':repeat+1,'images':len(ids)}),flush=True)
    perimage={};summary={}
    for mode in ['baseline',*nets]:
        perimage[mode]={im:np.mean([r['total_ms'] for r in rows if r['mode']==mode and r['image_id']==im]) for im in ids}
        rr=[r for r in rows if r['mode']==mode]
        summary[mode]={metric:dict(mean=float(np.mean([r[metric] for r in rr])),median=float(np.median([r[metric] for r in rr])),p90=float(np.quantile([r[metric] for r in rr],.9))) for metric in ('total_ms','forward_ms','mask_ms')}
        if mode!='baseline':summary[mode]['paired_overhead_ms']=float(np.mean([perimage[mode][im]-perimage['baseline'][im] for im in ids]))
    result=dict(gpu=torch.cuda.get_device_name(0),dtype='float32',batch=1,image_ids=ids,warm_ids=warm_ids,repeats=3,
        summary=summary,prediction_count_range=[min(r['predictions'] for r in rows),max(r['predictions'] for r in rows)],
        note='Unfused eager model plus topk, full-grid logits, correction, mask crop/resize and CPU transfer. Input loading/preprocessing and RLE encoding excluded. Synchronization included equally. Research implementation, not optimized deployed FPS.')
    (a.out/'TIMING.json').write_text(json.dumps(result,indent=2));(a.out/'samples.json').write_text(json.dumps(rows))
    (a.out/'COMPLETE.json').write_text(json.dumps({'images':50,'repeats':3,'conditions':4}));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
