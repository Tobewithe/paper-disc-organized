"""Phase B local GPU replay from the frozen checkpoint.

The candidate identity (image_id, annotation_id, branch, raw_id, level) comes
from the already completed Phase-A cache.  This script only re-runs the
frozen forward pass to obtain complete proto/coeff/boxes and never performs
assignment, optimization, training, threshold selection, or candidate
filtering.
"""
from __future__ import annotations
import argparse, gc, json, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops

from phase_b_normal_replay import (
    jdump, normalized_affine, load_native_head, padded_gt, decode_original,
    auc_supported, diag_bce, ci_bootstrap, sha256,
)


def aggregate(results, out):
    tables = {}
    for group in ("fit", "dev", "val"):
        rr = [r for r in results if r["group"] == group]
        if not rr: continue
        tables[group] = {"n_candidates": len(rr), "n_images": len({r["image_id"] for r in rr}), "candidate": {}, "image_macro": {}, "by_level": {}}
        for key in ("iou", "coverage", "auc", "bce"):
            aa=np.asarray([r[key+"_A"] for r in rr],float); bb=np.asarray([r[key+"_B"] for r in rr],float)
            tables[group]["candidate"][key]={"A":float(np.nanmean(aa)),"B":float(np.nanmean(bb)),"delta":float(np.nanmean(bb-aa))}
        by_img=defaultdict(list)
        for r in rr: by_img[r["image_id"]].append(r)
        for key in ("iou", "coverage", "auc", "bce"):
            per={str(i):[float(np.nanmean([r[key+"_A"] for r in xs])),float(np.nanmean([r[key+"_B"] for r in xs]))] for i,xs in by_img.items()}
            tables[group]["image_macro"][key]=ci_bootstrap(per)
        for level in (0,1,2):
            q=[r for r in rr if r["pyramid_level"]==level]
            if q: tables[group]["by_level"][str(level)]={"n":len(q),"iou_delta":float(np.mean([r["iou_B"]-r["iou_A"] for r in q])),"bce_delta":float(np.mean([r["bce_B"]-r["bce_A"] for r in q])),"mask75_delta":float(np.mean([r["mask75_B"]-r["mask75_A"] for r in q]))}
        tables[group]["mask75"]={"A":int(sum(r["mask75_A"] for r in rr)),"B":int(sum(r["mask75_B"] for r in rr)),"repair":int(sum(r["mask75_A"]==0 and r["mask75_B"]==1 for r in rr)),"damage":int(sum(r["mask75_A"]==1 and r["mask75_B"]==0 for r in rr))}
        tables[group]["checks"]={"c0_head_maxerr":float(max(r["c0_head_maxerr"] for r in rr)),"merge_direct_coeff_maxerr":float(max(r["merge_direct_coeff_maxerr"] for r in rr)),"merge_direct_logit_maxerr":float(max(r["merge_direct_logit_maxerr"] for r in rr))}
        for success in (0,1):
            q=[r for r in rr if r["orig_success_A"]==success]
            if q: tables[group].setdefault("by_original_success",{})[str(success)]={"n":len(q),"iou_delta":float(np.mean([r["iou_B"]-r["iou_A"] for r in q])),"coverage_delta":float(np.mean([r["coverage_B"]-r["coverage_A"] for r in q])),"auc_delta":float(np.nanmean([r["auc_B"]-r["auc_A"] for r in q]))}
    jdump(out/"SUMMARY.json", {"tables":tables,"protocol":{"decoder":"ultralytics-8.4.100 process_mask upsample=True then scale_masks default bilinear","main_comparison":"B_minus_A","bootstrap":5000,"auc":"continuous full-resolution logit restricted to fixed predicted box support","gt_use":"fit only for shared solve; dev/val only for evaluation","forward":"same frozen YOLO26m-seg checkpoint; candidate raw_id/level loaded from Phase-A identity list"}})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ids',type=Path,required=True); ap.add_argument('--weights',type=Path,required=True); ap.add_argument('--shared',type=Path,required=True); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--groups',nargs='*',default=['fit','dev','val'])
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    ids=json.loads(args.ids.read_text(encoding='utf-8'))
    shared=torch.load(args.shared,map_location='cpu',weights_only=False); A,stats,raw_maps=normalized_affine(shared)
    wrapper=YOLO(str(args.weights)); model=wrapper.model.cuda().float().eval()
    for p in model.parameters(): p.requires_grad_(False)
    layers=load_native_head(args.weights,raw_maps,args.out)
    head=model.model[-1]; captured={}; hooks=[]
    for level,branch in enumerate(head.one2one_cv4):
        hooks.append(branch[-1].register_forward_pre_hook(lambda _,x,level=level: captured.__setitem__(level,x[0].detach())))
    transform=LetterBox((640,640),auto=False,stride=32)
    coco_train=COCO(str(args.data/'annotations'/'instances_train2017.json')); coco_val=COCO(str(args.data/'annotations'/'instances_val2017.json'))
    results=[]; start=time.monotonic()
    for group in args.groups:
        source=coco_val if group=='val' else coco_train; domain='val' if group=='val' else 'train'; root=args.data/'images'/f'{domain}2017'
        by_image=defaultdict(list)
        for m in ids[group]: by_image[int(m['image_id'])].append(m)
        image_ids=sorted(by_image)
        for pos,iid in enumerate(image_ids,1):
            name=source.imgs[iid]['file_name']; im=cv2.imread(str(root/name));
            if im is None: raise FileNotFoundError(root/name)
            shaped=transform(image=im); tensor=torch.from_numpy(np.ascontiguousarray(shaped[:,:,::-1].transpose(2,0,1))).cuda().float()[None]/255.0
            captured.clear()
            with torch.no_grad(): _,raw=model(tensor)
            pred=raw['one2one']; proto=pred['proto'][0].detach(); coeff=pred['mask_coefficient'][0].T.detach(); boxes=head._get_decode_boxes(pred)[0].T.detach(); h=torch.cat([captured[l][0].flatten(1).T for l in range(3)]).detach()
            levels=torch.cat([torch.full((captured[l].shape[-2]*captured[l].shape[-1],),l,dtype=torch.long,device='cuda') for l in range(3)])
            oh,ow=im.shape[:2]; gain=min(640/oh,640/ow); nh,nw=round(oh*gain),round(ow*gain); top,left=round((640-nh)/2-.1),round((640-nw)/2-.1)
            archive={'original_shape':(oh,ow),'gain':gain,'left':left,'top':top}
            for m in by_image[iid]:
                rid=int(m['raw_id']); level=int(m['pyramid_level']); ann=source.anns[int(m['annotation_id'])]
                if rid<0 or rid>=len(coeff): raise RuntimeError(f'raw id out of range: {iid} {rid}')
                if int(levels[rid])!=level: raise RuntimeError(f'level mismatch: {iid} {rid} {level} {int(levels[rid])}')
                c0=coeff[rid]; c0_cpu=c0.cpu(); hh=h[rid].cpu().double(); mu,std=stats[level]; a=A[level]; c1d=c0_cpu.double()+torch.cat(((hh-mu)/std,torch.ones(1,dtype=torch.float64)))@a
                layer=layers[level]; c0h=hh.float()@layer['W0'].T+layer['b0']; c1_cpu=hh.float()@layer['W1'].T+layer['b1']; c1=c1_cpu.cuda()
                gt_pad,gt_orig=padded_gt(ann,archive); box=boxes[rid]
                ma=decode_original(proto,c0,box,archive).cpu(); mb=decode_original(proto,c1,box,archive).cpu()
                ioua=float((ma&gt_orig).sum()/(ma|gt_orig).sum().clamp_min(1)); ioub=float((mb&gt_orig).sum()/(mb|gt_orig).sum().clamp_min(1)); cova=float((ma&gt_orig).sum()/gt_orig.sum().clamp_min(1)); covb=float((mb&gt_orig).sum()/gt_orig.sum().clamp_min(1))
                z0=(c0@proto.flatten(1)).reshape(160,160); z1=(c1@proto.flatten(1)).reshape(160,160); z0u=F.interpolate(z0[None,None],(640,640),mode='bilinear')[0,0].cpu(); z1u=F.interpolate(z1[None,None],(640,640),mode='bilinear')[0,0].cpu()
                aa=auc_supported(z0u,gt_pad,box); ab=auc_supported(z1u,gt_pad,box); ba,_=diag_bce(proto,c0,box,gt_pad); bb,_=diag_bce(proto,c1,box,gt_pad)
                results.append({'group':group,'image_id':iid,'annotation_id':int(m['annotation_id']),'branch':'one2one','raw_id':rid,'pyramid_level':level,'box_iou':float(m.get('box_iou',float('nan'))),'iou_A':ioua,'iou_B':ioub,'coverage_A':cova,'coverage_B':covb,'auc_A':aa,'auc_B':ab,'bce_A':ba,'bce_B':bb,'mask75_A':int(ioua>=.75),'mask75_B':int(ioub>=.75),'orig_success_A':int(ioua>=.75),'c0_head_maxerr':float((c0h-c0_cpu).abs().max()),'merge_direct_coeff_maxerr':float((c1_cpu-c1d.float()).abs().max()),'merge_direct_logit_maxerr':float(((c1_cpu.double()-c1d)@proto.cpu().double().flatten(1)).abs().max())})
            if pos%25==0 or pos==len(image_ids): print(json.dumps({'group':group,'images':pos,'total':len(image_ids),'results':len(results),'elapsed_s':time.monotonic()-start}),flush=True)
    with (args.out/'per_candidate.jsonl').open('w',encoding='utf-8') as f:
        for r in results: f.write(json.dumps(r,allow_nan=True)+'\n')
    aggregate(results,args.out); jdump(args.out/'COMPLETE.json',{'status':'completed','checkpoint_sha256':sha256(args.weights),'candidate_identities':str(args.ids),'groups':args.groups,'n_candidates':len(results)})
    for h in hooks: h.remove()

if __name__=='__main__': main()
