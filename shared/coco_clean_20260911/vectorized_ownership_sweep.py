"""Frozen-output ownership sweep for same-class overlap pixels.

This is a diagnostic post-processing upper/lower-bound probe, not a proposed
method. It applies the same-class max-logit ownership rule with several fixed
margins and keeps boxes/classes/scores unchanged. No GT is used to choose a
margin or during inference.
"""
from __future__ import annotations
import argparse, contextlib, gzip, io, json, time
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools import mask as mu
from pycocotools.cocoeval import COCOeval
from ultralytics.utils import ops

ROOT=Path(__file__).resolve().parent

def rle(x):
    q=mu.encode(np.asfortranarray(x.astype(np.uint8))); q['counts']=q['counts'].decode('ascii'); return q

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--margins',default='0,0.25,0.5,1.0'); a=ap.parse_args(); out=a.out.resolve(); out.mkdir(parents=False,exist_ok=False); (out/'predictions').mkdir()
    ids=json.loads((a.cache/'selection.json').read_text())['transfer']
    ann_path=ROOT.parent.parent/'datasets/coco/annotations/instances_train2017.json'
    with contextlib.redirect_stdout(io.StringIO()): full=COCO(str(ann_path)); gt=COCO()
    gt.dataset=dict(info=full.dataset.get('info',{}),images=[full.imgs[i] for i in ids],categories=list(full.cats.values()),annotations=[x for i in ids for x in full.imgToAnns[i]])
    with contextlib.redirect_stdout(io.StringIO()): gt.createIndex()
    cats=sorted(gt.cats); margins=[float(x) for x in a.margins.split(',')]; names=['original']+[f'own_m{str(x).replace(".","p")}' for x in margins]
    pred={x:[] for x in names}; stats={x:dict(changed=0,removed=0,images=0) for x in names}; start=time.monotonic()
    for num,iid in enumerate(ids,1):
        with np.load(a.cache/'images'/f'{iid}.npz') as q: item={k:q[k] for k in q.files}
        p=torch.as_tensor(item['proto'],device='cuda',dtype=torch.float32); c=torch.as_tensor(item['coeff'],device='cuda',dtype=torch.float32); b=torch.as_tensor(item['boxes'],device='cuda',dtype=torch.float32); det=item['detections']; ishape=tuple(map(int,item['input_shape'])); shape=tuple(map(int,item['shape']))
        if not len(c): continue
        # Reproduce process_mask logits once, then use the exact cropped binary
        # support as the candidate domain for ownership competition.
        z=(c @ p.flatten(1)).reshape(1,len(c),p.shape[-2],p.shape[-1]); z=torch.nn.functional.interpolate(z,ishape,mode='bilinear',align_corners=False)[0]
        base=ops.crop_mask((z>0).to(torch.uint8),b); base_np=ops.scale_masks(base[:,None],shape)[:,0].cpu().numpy().astype(bool)
        # Set logits outside each candidate's crop to -inf before classwise max.
        domain=ops.crop_mask(torch.ones((len(c),*ishape),device='cuda',dtype=torch.uint8),b).bool(); z=torch.where(domain,z,torch.full_like(z,-1e9))
        out_masks={'original':base}
        for margin,name in zip(margins,names[1:]):
            keep=base.bool().clone()
            for cls in torch.unique(torch.as_tensor(det[:,5],device='cuda',dtype=torch.long)):
                ix=torch.where(torch.as_tensor(det[:,5],device='cuda',dtype=torch.long)==cls)[0]
                if len(ix)<2: continue
                zz=z[ix]; active=base[ix].bool(); mx=zz.max(0).values
                # Remove a positive pixel only if another same-class candidate
                # wins by the prescribed logit margin in common support.
                keep[ix]=active & (zz >= mx[None]-margin)
            out_masks[name]=keep.to(torch.uint8)
        for name,m in out_masks.items():
            restored=ops.scale_masks(m[:,None],shape)[:,0].cpu().numpy().astype(bool)
            if name!='original':
                stats[name]['changed'] += int(np.count_nonzero(restored != base_np)); stats[name]['removed'] += int(np.count_nonzero(base_np & ~restored))
            for j,x in enumerate(restored):
                if not x.any(): continue
                pred[name].append(dict(image_id=iid,category_id=cats[int(det[j,5])],score=float(det[j,4]),segmentation=rle(x)))
        for name in names: stats[name]['images']+=1
        if num%25==0: print(json.dumps({'stage':'decode','images':num,'total':len(ids),'seconds':time.monotonic()-start}),flush=True)
    rows=[]
    for name in names:
        with gzip.open(out/'predictions'/f'{name}.json.gz','wt',encoding='utf8') as f: json.dump(pred[name],f,separators=(',',':'))
        with contextlib.redirect_stdout(io.StringIO()): dt=gt.loadRes(pred[name]); ev=COCOeval(gt,dt,'segm'); ev.params.imgIds=ids; ev.evaluate(); ev.accumulate(); ev.summarize()
        rows.append(dict(arm=name,mask_ap=float(ev.stats[0]),mask_ap50=float(ev.stats[1]),mask_ap75=float(ev.stats[2]),predictions=len(pred[name]),**stats[name]))
    (out/'SUMMARY.json').write_text(json.dumps({'status':'COMPLETE','images':len(ids),'margins':margins,'rows':rows,'GT_used_in_inference':False,'rule':'classwise max logit in candidate crop; suppress only when winner margin exceeds tau'},indent=2),encoding='utf8')
    print(json.dumps(rows,ensure_ascii=False))

if __name__=='__main__': main()
