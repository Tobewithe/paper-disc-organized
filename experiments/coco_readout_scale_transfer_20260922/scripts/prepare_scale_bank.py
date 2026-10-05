"""Extract a fresh S-model bank on fixed image splits; do not reuse M raw candidates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou
from pycocotools.coco import COCO

import learn_refinement as original
from component_seed_probe import setup
from shared_shape_head import bank_indices,full_template


@torch.no_grad()
def main(args):
    setup();start=time.monotonic();split=json.loads(args.split.read_text())
    fit_ids=set(split['fit_image_ids']);dev_ids=set(split['selection_image_ids'])
    assert len(fit_ids)==800 and len(dev_ids)==200 and not fit_ids&dev_ids
    ids=sorted(fit_ids|dev_ids);coco=COCO(str(args.data/'annotations/instances_train2017.json'))
    wrapper=YOLO(str(args.weights));model=wrapper.model.cuda().float().eval();head=model.model[-1]
    assert head.end2end and head.nm==32
    dims=[m[-1].weight.shape[1] for m in head.one2one_cv4];assert len(set(dims))==1
    hidden_dim=dims[0];assert hidden_dim==32,'This run is the verified S-model setup.'
    states=[{'weight':m[-1].weight.detach().cpu()[:,:,0,0],'bias':m[-1].bias.detach().cpu()} for m in head.one2one_cv4]
    template=json.loads(args.template.read_text());gridT=torch.tensor(template['field'],device='cuda').reshape(1,16)
    cats=sorted(coco.cats);assert [model.names[i] for i in range(80)]==[coco.cats[c]['name'] for c in cats]
    cat_to_cls={c:i for i,c in enumerate(cats)};transform=LetterBox((640,640),auto=True,stride=32)
    captured={};hooks=[]
    for level,module in enumerate(head.one2one_cv4):
        def hook(m,inputs,level=level):captured[level]=inputs[0].detach()
        hooks.append(module[-1].register_forward_pre_hook(hook))
    values={k:[] for k in ['x','base','target','hidden','level','scale','mean','std','count','boxes','shape']};records=[];max_error=0.
    try:
        for ni,iid in enumerate(ids):
            im=cv2.imread(str(args.data/'images/train2017'/coco.imgs[iid]['file_name']));assert im is not None
            orig=im.shape[:2];res=transform.apply_image({'img':im},transform.get_params({'img':im}))['img']
            inp=torch.from_numpy(np.ascontiguousarray(res[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255
            shape=tuple(inp.shape[2:]);_,raw=model(inp);p=raw['one2one'];proto=p['proto'][0];coef=p['mask_coefficient'][0].T
            hidden=torch.cat([captured[l][0].flatten(1).T for l in range(3)])
            levels=torch.cat([torch.full((captured[l].shape[2]*captured[l].shape[3],),l,device='cuda') for l in range(3)])
            boxes=head._get_decode_boxes(p)[0].T;scores,cls,rids=head.get_topk_index(p['scores'].permute(0,2,1).sigmoid(),300)
            keep=scores[0,:,0]>.001;cls=cls[0,keep,0].long();rids=rids[0,keep,0];bb=boxes[rids]
            anns=[a for a in coco.imgToAnns[iid] if not a.get('iscrowd',0) and not a.get('ignore',0)]
            if anns and len(bb):
                gt=torch.tensor([a['bbox'] for a in anns],device='cuda',dtype=torch.float32);gt[:,2:]+=gt[:,:2]
                quality=box_iou(ops.scale_boxes(shape,bb.clone(),orig),gt)
                gc=torch.tensor([cat_to_cls[a['category_id']] for a in anns],device='cuda');quality[cls[:,None]!=gc[None]]=-1
                best,owner=quality.max(1);ix=torch.where(best>=.5)[0]
                if len(ix):
                    rawids=rids[ix];bb=bb[ix];h=hidden[rawids];lv=levels[rawids]
                    rebuilt=torch.empty_like(coef[rawids])
                    for level in range(3):
                        selected=lv==level
                        if selected.any():rebuilt[selected]=F.linear(h[selected],head.one2one_cv4[level][-1].weight[:,:,0,0],head.one2one_cv4[level][-1].bias)
                    max_error=max(max_error,float((rebuilt-coef[rawids]).abs().max()))
                    x,b,roi=original.features(proto,coef[rawids],bb,shape);owners=owner[ix].cpu().tolist()
                    masks=torch.tensor(np.stack([coco.annToMask(anns[j]) for j in owners]),device='cuda',dtype=torch.float32)
                    oh,ow=orig;ih,iw=shape;gain=min(ih/oh,iw/ow);nh,nw=round(oh*gain),round(ow*gain)
                    top,left=round((ih-nh)/2-.1),round((iw-nw)/2-.1)
                    y=F.pad(F.interpolate(masks[:,None],(nh,nw),mode='bilinear',align_corners=False),(left,iw-nw-left,top,ih-nh-top))
                    y=F.grid_sample(y,roi,align_corners=False)[:,0]
                    for key,t in [('x',x),('base',b),('target',y)]:values[key].append(t.cpu().half())
                    scale=proto.square().mean((1,2)).sqrt().clamp_min(.1)
                    for key,t in [('hidden',h),('level',lv),('scale',scale[None].expand(len(ix),-1)),('boxes',bb),
                                  ('shape',torch.tensor(shape,device='cuda')[None].expand(len(ix),-1))]:values[key].append(t.cpu())
                    for first in range(0,len(bb),16):
                        _,mean,std,count=full_template(gridT,bb[first:first+16],shape)
                        for key,t in [('mean',mean),('std',std),('count',count)]:values[key].append(t.cpu())
                    records.extend({'image_id':iid,'annotation_id':anns[j]['id'],'raw_id':int(rid)} for rid,j in zip(rawids.cpu().tolist(),owners))
            if ni%25==0 or ni+1==len(ids):
                pr={'images':ni+1,'total':len(ids),'records_before_dedup':len(records),'elapsed_s':time.monotonic()-start}
                original.save(args.out/'progress.json',pr);print(json.dumps(pr),flush=True)
    finally:
        for hook in hooks:hook.remove()
    assert max_error<1e-4,max_error
    owner_sets=defaultdict(set)
    for r in records:owner_sets[r['image_id'],r['raw_id']].add(r['annotation_id'])
    keep=[];seen=set()
    for i,r in enumerate(records):
        key=r['image_id'],r['raw_id']
        if len(owner_sets[key])==1 and key not in seen:keep.append(i);seen.add(key)
    audit={'original_rows':len(records),'ambiguous_raw_ids':sum(len(v)>1 for v in owner_sets.values()),'retained_rows':len(keep),'hidden_dim':hidden_dim,'max_reconstruction_error':max_error}
    rows=[records[i] for i in keep];ix=torch.tensor(keep)
    values={k:torch.cat(v)[ix] for k,v in values.items()};bank={k:values[k] for k in ['x','base','target']};bank['records']=rows
    assert all(bool(torch.isfinite(bank[k]).all()) for k in ['x','base','target'])
    torch.save(bank,args.out/'training_bank.pt')
    torch.save({'records':rows,'states':states,**{k:values[k] for k in ['hidden','level','scale']}},args.out/'coefficient_features.pt')
    torch.save({'records':rows,'template':template,**{k:values[k] for k in ['mean','std','count','boxes','shape']}},args.out/'geometry.pt')
    fit,dev=bank_indices(bank,split);audit.update(fit_records=len(fit),selection_records=len(dev),
         fit_effective_images=len({rows[i]['image_id'] for i in fit}),selection_effective_images=len({rows[i]['image_id'] for i in dev}))
    split=dict(split,fit_rows=len(fit),selection_rows=len(dev),source=str(args.out/'training_bank.pt'),
        original_image_split_source=str(args.split),weights=str(args.weights),
        record_scope='Fresh S-model candidates; image IDs reused, M-model candidate counts and identities not reused.')
    original.save(args.out/'SPLIT.json',split);original.save(args.out/'BANK_AUDIT.json',audit)
    original.save(args.out/'COMPLETE.json',{'images':len(ids),'audit':audit,'elapsed_s':time.monotonic()-start,'scope':'S-model preparation only; no validation-performance conclusion'})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['out','data','weights','split','template']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);main(args)
