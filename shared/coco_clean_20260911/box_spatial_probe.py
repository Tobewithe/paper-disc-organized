"""GT-free predicted-box spatial coefficient probe; no model training.

For each matched same-class prediction pair, construct differentiable regions from
the two predicted boxes only. Move coefficients 5 degrees along the tangent of a
box-derived spatial surrogate and evaluate against COCO GT after the fact.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4'); os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse, contextlib, csv, hashlib, io, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
from frozen_mechanism_probe import ROOT, decode, original_logits, pixel_metrics, sha, write_csv, write_json

SEED=20260911
VARIANTS={
  'core06_excl': dict(core=.60,pos='core',neg='neighbor_excl',wneg=1.0,wbg=.10),
  'core08_excl': dict(core=.80,pos='core',neg='neighbor_excl',wneg=1.0,wbg=.10),
  'box_excl': dict(core=.70,pos='own_excl',neg='neighbor_excl',wneg=1.0,wbg=.10),
  'box_overlap': dict(core=.70,pos='own',neg='neighbor',wneg=.50,wbg=.10),
}
FIELDS=('coverage','same_neighbor','neighbor','background','mask_iou')

def mask_box(box, shape, frac=1.0):
    h,w=shape; x1,y1,x2,y2=[float(x) for x in box]
    cx=(x1+x2)/2;cy=(y1+y2)/2;bw=(x2-x1)*frac;bh=(y2-y1)*frac
    x1=max(0,min(w,x1 if frac==1 else cx-bw/2));x2=max(0,min(w,x2 if frac==1 else cx+bw/2))
    y1=max(0,min(h,y1 if frac==1 else cy-bh/2));y2=max(0,min(h,y2 if frac==1 else cy+bh/2))
    yy,xx=np.ogrid[:h,:w];return (xx>=x1)&(xx<x2)&(yy>=y1)&(yy<y2)

def direction(c,p,capture,support,variant):
    x=c.detach().clone().requires_grad_(True);z=original_logits(x,p,capture);terms=[]
    cfg=VARIANTS[variant]
    for j in range(2):
        own=support[j];other=support[1-j];
        if cfg['pos']=='core': pos=mask_box(capture['boxes'][j],capture['shape'],cfg['core'])
        elif cfg['pos']=='own_excl': pos=own&~other
        else: pos=own
        if cfg['neg']=='neighbor_excl': neg=other&~own
        elif cfg['neg']=='neighbor': neg=other
        else: neg=~own
        bg=~(own|other)
        for weight,region,sign in [(1.,pos,-1.),(cfg['wneg'],neg,1.),(cfg['wbg'],bg,1.)]:
            if region.any(): terms.append(weight*F.softplus(sign*z[j][torch.as_tensor(region,device=z.device)]).mean())
    objective=sum(terms);grad=torch.autograd.grad(objective,x)[0];u=F.normalize(c,dim=1)
    tangent=-grad+((grad*u).sum(1)[:,None])*u; tangent=F.normalize(tangent,dim=1)
    theta=math.radians(5.);out=(math.cos(theta)*u+math.sin(theta)*tangent)*c.norm(dim=1)[:,None]
    return out.detach(),float(objective.detach())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run.resolve();out=run/'box_spatial_probe';out.mkdir(exist_ok=False)
    gtpath=ROOT/'data/annotations/instances_val2017.json';manifest=ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv'
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(gtpath))
    with manifest.open(encoding='utf-8-sig',newline='') as f:meta={int(r['annotation_id']):r for r in csv.DictReader(f)}
    selected=json.loads((run/'selection.json').read_text());statuses=json.loads((run/'statuses.json').read_text());valid=[s for s in statuses if s['status']=='ok']
    rows=[];start=time.monotonic();torch.set_num_threads(4)
    for k,s in enumerate(valid,1):
        iid=s['image_id'];cache=np.load(run/'tensors'/f'{iid}.npz');p=torch.tensor(cache['proto'],device='cuda');c=torch.tensor(cache['coeff'],device='cuda');boxes=torch.tensor(cache['boxes'],device='cuda');shape=tuple(int(x) for x in cache['shape']);ishape=tuple(int(x) for x in cache['input_shape']);capture=dict(shape=shape,input_shape=ishape)
        aid=cache['annotation_ids'].tolist();anns=gt.imgToAnns[iid];masks={a['id']:gt.annToMask(a).astype(bool) for a in anns};ordinary=[a for a in anns if not a.get('iscrowd',0)];union=np.logical_or.reduce([masks[a['id']] for a in ordinary]);same=np.logical_or.reduce([masks[a['id']] for a in ordinary if a['category_id']==gt.anns[aid[0]]['category_id']]);crowd=[masks[a['id']] for a in anns if a.get('iscrowd',0)];validpix=~np.logical_or.reduce(crowd) if crowd else np.ones(shape,dtype=bool)
        # Exact predicted-box support at original resolution, matching decoder crop.
        support=np.asarray(ops.scale_masks(ops.crop_mask(torch.ones((2,*ishape),dtype=torch.uint8,device='cuda'),boxes)[:,None],shape)[:,0].cpu()>0.5)
        capture['boxes']=boxes
        for name in VARIANTS:
            changed,obj=direction(c,p,capture,support,name);cm=decode(torch.cat([c,changed]),p,boxes.repeat(2,1),capture)[0]['cropped'].cpu().numpy()
            for j,target in enumerate(aid):
                base=pixel_metrics(torch.as_tensor(cm[j]),torch.as_tensor(masks[target]),torch.as_tensor(union),torch.as_tensor(same),torch.as_tensor(validpix));new=pixel_metrics(torch.as_tensor(cm[j+2]),torch.as_tensor(masks[target]),torch.as_tensor(union),torch.as_tensor(same),torch.as_tensor(validpix))
                rows.append(dict(image_id=iid,target_annotation=target,target_ici=float(meta[target]['ici_same']),pair_group=s['pair_group'],variant=name,objective=obj,coefficient_cosine_before=float(F.cosine_similarity(c[0:1],c[1:2]).item()),coefficient_cosine_after=float(F.cosine_similarity(changed[0:1],changed[1:2]).item()),angle_degrees=5.0,**{f'base_{m}':float(base[m]) for m in FIELDS},**{f'new_{m}':float(new[m]) for m in FIELDS},**{f'delta_{m}':float(new[m]-base[m]) for m in FIELDS}))
        if k%50==0:print(json.dumps(dict(completed=k,total=len(valid),seconds=round(time.monotonic()-start,1))),flush=True)
    write_csv(out/'per_target.csv',rows)
    # image-cluster bootstrap, reusing selection image IDs and fixed draws.
    draws=np.load(run/'bootstrap_draws.npz');image_ids=draws['image_ids'];D=draws['draws'];pos={int(i):k for k,i in enumerate(image_ids)};stats=[]
    for variant in VARIANTS:
      part=[r for r in rows if r['variant']==variant]
      for group in ['all','high','low']:
       q=[r for r in part if group=='all' or (float(r['target_ici'])>.5)==(group=='high')]
       for m in FIELDS:
        vals=np.zeros(len(image_ids));cnt=np.zeros(len(image_ids))
        for r in q:vals[pos[int(r['image_id'])]]+=r['delta_'+m];cnt[pos[int(r['image_id'])]]+=1
        boot=(vals[D].sum(1)/cnt[D].sum(1));stats.append(dict(variant=variant,group=group,metric=m,mean=float(vals.sum()/cnt.sum()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),targets=int(cnt.sum()),images=int((cnt>0).sum())))
    write_csv(out/'summary.csv',stats);result=dict(status='COMPLETE',training=False,valid_pairs=len(valid),targets=len(rows),variants=list(VARIANTS),elapsed_seconds=time.monotonic()-start,script_sha256=sha(__file__),limitations='Predicted-box surrogate; GT used only for evaluation. Temporary coefficient gradients; no network update.')
    write_json(out/'COMPLETE.json',result)
    for r in stats:
      if r['group']=='high' and r['metric'] in ['mask_iou','same_neighbor']:print(json.dumps(r),flush=True)

if __name__=='__main__':main()
