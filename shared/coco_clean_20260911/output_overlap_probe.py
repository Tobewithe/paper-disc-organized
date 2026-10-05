"""GT-free output-space overlap probe on cached official COCO predictions."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,csv,json,math,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
from frozen_mechanism_probe import ROOT,decode,original_logits,pixel_metrics,write_csv,write_json,sha

FIELDS=('coverage','same_neighbor','neighbor','background','mask_iou')
VARIANTS={
 'q_overlap':dict(wq=1.0,wex=0.,wbg=0.),
 'q_overlap_excl':dict(wq=1.,wex=.5,wbg=.05),
 'q_overlap_preserve':dict(wq=1.,wex=1.,wbg=.05),
 'q_overlap_strong':dict(wq=2.,wex=1.,wbg=.10),
}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run.resolve();out=run/'output_overlap_probe';out.mkdir(exist_ok=False)
 with contextlib.redirect_stdout(open(Path(os.devnull),'w')):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
 with (ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv').open(encoding='utf-8-sig',newline='') as f:meta={int(r['annotation_id']):r for r in csv.DictReader(f)}
 statuses=json.loads((run/'statuses.json').read_text());valid=[s for s in statuses if s['status']=='ok'];rows=[];start=time.monotonic();torch.set_num_threads(4)
 for k,s in enumerate(valid,1):
  iid=s['image_id'];cache=np.load(run/'tensors'/f'{iid}.npz');p=torch.tensor(cache['proto'],device='cuda');c=torch.tensor(cache['coeff'],device='cuda');boxes=torch.tensor(cache['boxes'],device='cuda');shape=tuple(int(x) for x in cache['shape']);ishape=tuple(int(x) for x in cache['input_shape']);capture=dict(shape=shape,input_shape=ishape)
  aids=cache['annotation_ids'].tolist();anns=gt.imgToAnns[iid];masks={a['id']:gt.annToMask(a).astype(bool) for a in anns};ordinary=[a for a in anns if not a.get('iscrowd',0)];union=np.logical_or.reduce([masks[a['id']] for a in ordinary]);same=np.logical_or.reduce([masks[a['id']] for a in ordinary if a['category_id']==gt.anns[aids[0]]['category_id']]);crowd=[masks[a['id']] for a in anns if a.get('iscrowd',0)];validpix=~np.logical_or.reduce(crowd) if crowd else np.ones(shape,dtype=bool)
  support=np.asarray(ops.scale_masks(ops.crop_mask(torch.ones((2,*ishape),dtype=torch.uint8,device='cuda'),boxes)[:,None],shape)[:,0].cpu()>0.5); overlap=support[0]&support[1]; ex=[support[0]&~support[1],support[1]&~support[0]]; 
  x=c.detach().clone().requires_grad_(True);z=original_logits(x,p,capture);z0=z.detach();terms=[]
  if overlap.any(): terms.append(torch.stack([F.softplus(z[0][torch.as_tensor(overlap,device='cuda')]),F.softplus(z[1][torch.as_tensor(overlap,device='cuda')])]).prod(0).mean())
  for j in range(2):
   if ex[j].any(): terms.append(.5*F.softplus(-z[j][torch.as_tensor(ex[j],device='cuda')]).mean())
   bg=~(support[0]|support[1]);
   if bg.any(): terms.append(.05*F.softplus(z[j][torch.as_tensor(bg,device='cuda')]).mean())
  grad=torch.autograd.grad(sum(terms),x)[0];u=F.normalize(c,dim=1);t=grad-((grad*u).sum(1)[:,None])*u;t=F.normalize(t,dim=1);baseang=math.radians(5.)
  # One base gradient; weights alter objective only through current terms. Variants are
  # recomputed below so each has an independent output-space surrogate.
  for name,cfg in VARIANTS.items():
   x=c.detach().clone().requires_grad_(True);z=original_logits(x,p,capture);terms=[]
   if overlap.any():
    zz=[F.softplus(z[j][torch.as_tensor(overlap,device='cuda')]) for j in range(2)];terms.append(cfg['wq']*(zz[0]*zz[1]).mean())
   for j in range(2):
    if ex[j].any():terms.append(cfg['wex']*F.softplus(-z[j][torch.as_tensor(ex[j],device='cuda')]).mean())
    bg=~(support[0]|support[1])
    if bg.any():terms.append(cfg['wbg']*F.softplus(z[j][torch.as_tensor(bg,device='cuda')]).mean())
   grad=torch.autograd.grad(sum(terms),x)[0];u=F.normalize(c,dim=1);t=grad-((grad*u).sum(1)[:,None])*u;t=F.normalize(t,dim=1);changed=(math.cos(baseang)*u-math.sin(baseang)*t)*c.norm(dim=1)[:,None]
   cm=decode(torch.cat([c,changed]),p,boxes.repeat(2,1),capture)[0]['cropped'].cpu().numpy()
   for j,target in enumerate(aids):
    b=pixel_metrics(torch.as_tensor(cm[j]),torch.as_tensor(masks[target]),torch.as_tensor(union),torch.as_tensor(same),torch.as_tensor(validpix));n=pixel_metrics(torch.as_tensor(cm[j+2]),torch.as_tensor(masks[target]),torch.as_tensor(union),torch.as_tensor(same),torch.as_tensor(validpix))
    rows.append(dict(image_id=iid,target_annotation=target,target_ici=float(meta[target]['ici_same']),pair_group=s['pair_group'],variant=name,objective=float(sum(terms).detach()),coefficient_cosine_before=float(F.cosine_similarity(c[0:1],c[1:2]).item()),coefficient_cosine_after=float(F.cosine_similarity(changed[0:1],changed[1:2]).item()),**{f'base_{m}':float(b[m]) for m in FIELDS},**{f'new_{m}':float(n[m]) for m in FIELDS},**{f'delta_{m}':float(n[m]-b[m]) for m in FIELDS}))
  if k%50==0:print(json.dumps(dict(completed=k,total=len(valid),seconds=round(time.monotonic()-start,1))),flush=True)
 write_csv(out/'per_target.csv',rows);draws=np.load(run/'bootstrap_draws.npz');ids=draws['image_ids'];D=draws['draws'];pos={int(i):q for q,i in enumerate(ids)};stats=[]
 for v in VARIANTS:
  for g in ['all','high','low']:
   part=[r for r in rows if r['variant']==v and (g=='all' or (float(r['target_ici'])>.5)==(g=='high'))]
   for m in FIELDS:
    sm=np.zeros(len(ids));ct=sm.copy()
    for r in part:sm[pos[int(r['image_id'])]]+=r['delta_'+m];ct[pos[int(r['image_id'])]]+=1
    boot=sm[D].sum(1)/ct[D].sum(1);stats.append(dict(variant=v,group=g,metric=m,mean=float(sm.sum()/ct.sum()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),targets=int(ct.sum()),images=int((ct>0).sum())))
 write_csv(out/'summary.csv',stats);result=dict(status='COMPLETE',training=False,valid_pairs=len(valid),targets=len(rows),variants=list(VARIANTS),elapsed_seconds=time.monotonic()-start,script_sha256=sha(__file__),limitations='Predicted-box output-space surrogate; GT only evaluation; temporary coefficients and no network update.');write_json(out/'COMPLETE.json',result)
 for r in stats:
  if r['group']=='high' and r['metric'] in ['mask_iou','same_neighbor']:print(json.dumps(r),flush=True)
if __name__=='__main__':main()
