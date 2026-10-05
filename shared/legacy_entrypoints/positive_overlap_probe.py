"""GT-free positive-overlap suppression with one-sided pseudo-mask preservation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,csv,json,math,time
import numpy as np,torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
from frozen_mechanism_probe import ROOT,decode,original_logits,pixel_metrics,write_csv,write_json,sha
FIELDS=('coverage','same_neighbor','neighbor','background','mask_iou')
VARIANTS={'pos_anchor03':.3,'pos_anchor1':1.,'pos_anchor3':3.,'pos_anchor10':10.,'pos_anchor30':30.}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=__import__('pathlib').Path,required=True);a=ap.parse_args();run=a.run.resolve();out=run/'positive_overlap_probe';out.mkdir(exist_ok=False)
 with contextlib.redirect_stdout(open(os.devnull,'w')):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
 with (ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv').open(encoding='utf-8-sig',newline='') as f:meta={int(r['annotation_id']):r for r in csv.DictReader(f)}
 valid=[s for s in json.loads((run/'statuses.json').read_text()) if s['status']=='ok'];rows=[];start=time.monotonic();torch.set_num_threads(4)
 for k,s in enumerate(valid,1):
  iid=s['image_id'];cache=np.load(run/'tensors'/f'{iid}.npz');p=torch.tensor(cache['proto'],device='cuda');c0=torch.tensor(cache['coeff'],device='cuda');boxes=torch.tensor(cache['boxes'],device='cuda');shape=tuple(int(x) for x in cache['shape']);ishape=tuple(int(x) for x in cache['input_shape']);capture=dict(shape=shape,input_shape=ishape)
  aids=cache['annotation_ids'].tolist();anns=gt.imgToAnns[iid];masks={a['id']:gt.annToMask(a).astype(bool) for a in anns};ordinary=[a for a in anns if not a.get('iscrowd',0)];union=np.logical_or.reduce([masks[a['id']] for a in ordinary]);same=np.logical_or.reduce([masks[a['id']] for a in ordinary if a['category_id']==gt.anns[aids[0]]['category_id']]);crowd=[masks[a['id']] for a in anns if a.get('iscrowd',0)];validpix=~np.logical_or.reduce(crowd) if crowd else np.ones(shape,dtype=bool)
  support=np.asarray(ops.scale_masks(ops.crop_mask(torch.ones((2,*ishape),dtype=torch.uint8,device='cuda'),boxes)[:,None],shape)[:,0].cpu()>0.5);overlap=torch.as_tensor(support[0]&support[1],device='cuda');bg=torch.as_tensor(~(support[0]|support[1]),device='cuda');own=[torch.as_tensor(support[j],device='cuda') for j in range(2)];z0=original_logits(c0,p,capture).detach();positive=(z0>0)&overlap;norm=c0.norm(dim=1)
  for name,anchor in VARIANTS.items():
   c=c0.detach().clone()
   for step in range(5):
    x=c.detach().clone().requires_grad_(True);z=original_logits(x,p,capture);terms=[]
    if positive.any():
     # Only jointly positive overlap pixels are penalized; soft weights emphasize stable positives.
     w=torch.sigmoid(z0[0][positive[0]])*torch.sigmoid(z0[1][positive[0]])
     terms.append((w*F.softplus(z[0][positive[0]])*F.softplus(z[1][positive[0]])).sum()/(w.sum()+1e-6))
    for j in range(2):
     # One-sided anchor: do not penalize harmless increases, strongly resist loss of baseline positives.
     keep=own[j]&(z0[j]>0);terms.append(anchor*F.relu(z0[j][keep]-z[j][keep]).square().mean())
     if bg.any():terms.append(.02*F.softplus(z[j][bg]).mean())
    grad=torch.autograd.grad(sum(terms),x)[0];u=F.normalize(c,dim=1);t=grad-((grad*u).sum(1)[:,None])*u;t=F.normalize(t,dim=1);c=norm[:,None]*(math.cos(math.radians(1))*u-math.sin(math.radians(1))*t)
   cm=decode(torch.cat([c0,c]),p,boxes.repeat(2,1),capture)[0]['cropped'].cpu().numpy()
   for j,target in enumerate(aids):
    b=pixel_metrics(torch.as_tensor(cm[j]),torch.as_tensor(masks[target]),torch.as_tensor(union),torch.as_tensor(same),torch.as_tensor(validpix));n=pixel_metrics(torch.as_tensor(cm[j+2]),torch.as_tensor(masks[target]),torch.as_tensor(union),torch.as_tensor(same),torch.as_tensor(validpix))
    rows.append(dict(image_id=iid,target_annotation=target,target_ici=float(meta[target]['ici_same']),pair_group=s['pair_group'],variant=name,anchor_weight=anchor,angle_degrees=5.,positive_overlap_pixels=int(positive[j].sum()),**{f'base_{m}':float(b[m]) for m in FIELDS},**{f'new_{m}':float(n[m]) for m in FIELDS},**{f'delta_{m}':float(n[m]-b[m]) for m in FIELDS}))
  if k%50==0:print(json.dumps(dict(completed=k,total=len(valid),seconds=round(time.monotonic()-start,1))),flush=True)
 write_csv(out/'per_target.csv',rows);draws=np.load(run/'bootstrap_draws.npz');ids=draws['image_ids'];D=draws['draws'];pos={int(i):q for q,i in enumerate(ids)};stats=[]
 for v in VARIANTS:
  for g in ['all','high','low']:
   part=[r for r in rows if r['variant']==v and (g=='all' or (float(r['target_ici'])>.5)==(g=='high'))]
   for m in FIELDS:
    sm=np.zeros(len(ids));ct=sm.copy()
    for r in part:sm[pos[int(r['image_id'])]]+=r['delta_'+m];ct[pos[int(r['image_id'])]]+=1
    boot=sm[D].sum(1)/ct[D].sum(1);stats.append(dict(variant=v,group=g,metric=m,mean=float(sm.sum()/ct.sum()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),targets=int(ct.sum()),images=int((ct>0).sum())))
 write_csv(out/'summary.csv',stats);result=dict(status='COMPLETE',training=False,valid_pairs=len(valid),targets=len(rows),variants=list(VARIANTS),steps=5,step_degrees=1,elapsed_seconds=time.monotonic()-start,script_sha256=sha(__file__),limitations='Predicted-box positive-overlap surrogate; GT only evaluation; temporary coefficients, no network update.');write_json(out/'COMPLETE.json',result)
 for r in stats:
  if r['group']=='high' and r['metric'] in ['mask_iou','same_neighbor']:print(json.dumps(r),flush=True)
if __name__=='__main__':main()
