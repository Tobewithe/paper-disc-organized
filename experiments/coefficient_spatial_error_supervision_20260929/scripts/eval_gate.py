import argparse,json,sys,math
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).parent))
from train_full import load
from train_capacity import BigGlobal

def bd(box):
    x1=max(0,min(159,math.floor(float(box[0])*160/640))); y1=max(0,min(159,math.floor(float(box[1])*160/640)))
    x2=max(x1+1,min(160,math.ceil(float(box[2])*160/640))); y2=max(y1+1,min(160,math.ceil(float(box[3])*160/640)))
    return x1,y1,x2,y2

def auc(y,s):
    y=np.asarray(y).astype(np.uint8); s=np.asarray(s); pos=y==1; neg=y==0; np1=int(pos.sum()); nn=int(neg.sum())
    if np1==0 or nn==0:return float('nan')
    o=np.argsort(s,kind='mergesort'); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    return float((r[pos].sum()-np1*(np1+1)/2)/(np1*nn))

def calc(z,gt):
    pr=z>0; g=gt>0; inter=(pr&g).sum(); uni=(pr|g).sum()
    return {'auc':auc(gt,z),'iou':float(inter/uni) if uni else 1.,'mask75':float(inter/uni>=.75) if uni else 1.}

def main(a):
    dev=load(a.dev); raw=Path(a.raw); coco=__import__('pycocotools.coco',fromlist=['COCO']).COCO(str(a.coco))
    h=torch.stack([r['h'].float() for r in dev]); p=torch.stack([r['p'].float() for r in dev])
    ck=torch.load(a.model,map_location='cpu',weights_only=True); norm=ck['norm']; model=BigGlobal(h.shape[1]).cuda(); model.load_state_dict(ck['state']); model.eval()
    hh=((h-norm['hm'])/norm['hs']).cuda(); pp=((p-norm['pm'])/norm['ps']).cuda()
    with torch.no_grad(): dz=torch.cat([model(hh[j:j+512],pp[j:j+512]).cpu() for j in range(0,len(dev),512)])
    scores={k:[] for k in ['abs_mean','l2','positive_mean','negative_mean']}; base=[]; corr=[]
    cache={}
    for idx,r in enumerate(dev):
        iid=int(r['image_id']); f=cache.get(iid)
        if f is None: f=torch.load(raw/f'{iid:012d}.pt',map_location='cpu',weights_only=True); cache[iid]=f
        rid=int(r['raw_id']); proto=f['proto'].float(); c0=f['coeff'][rid].float(); box=f['boxes'][rid].float(); x1,y1,x2,y2=bd(box)
        z0=(proto*c0[:,None,None]).sum(0)[y1:y2,x1:x2]
        gt=torch.from_numpy(coco.annToMask(coco.anns[int(r['annotation_id'])]).astype('float32'))[None,None]
        gt=F.interpolate(gt,(160,160),mode='nearest')[0,0][y1:y2,x1:x2].numpy().astype(np.uint8)
        dzc=F.interpolate(dz[idx].reshape(1,1,16,16),size=z0.shape,mode='bilinear',align_corners=False)[0,0]
        b=calc(z0.numpy().ravel(),gt.ravel()); c=calc((z0+dzc).numpy().ravel(),gt.ravel()); base.append(b); corr.append(c)
        v=dz[idx].numpy(); scores['abs_mean'].append(float(np.abs(v).mean())); scores['l2'].append(float(np.sqrt((v*v).mean()))); scores['positive_mean'].append(float(np.maximum(v,0).mean())); scores['negative_mean'].append(float(np.maximum(-v,0).mean()))
    base_i=np.array([x['iou'] for x in base]); corr_i=np.array([x['iou'] for x in corr]); benefit=corr_i-base_i
    result={'n':len(dev),'raw_benefit_mean':float(benefit.mean()),'benefit_auc':{}}
    for k,s in scores.items():
        s=np.asarray(s); result['benefit_auc'][k]=auc((benefit>0).astype(np.uint8),s)
        tab={}
        for q in [.05,.10,.20,.30,.50]:
            n=max(1,int(round(len(s)*q))); ix=np.argsort(-s)[:n]; sel=np.zeros(len(s),bool); sel[ix]=1
            applied_i=np.where(sel,corr_i,base_i); tab[str(q)]={'coverage':float(q),'selected_mean_benefit':float(benefit[ix].mean()),'net_mean_iou_gain':float((applied_i-base_i).mean()),'selected_failure_gain':float(benefit[ix][base_i[ix]<.75].mean()) if np.any(base_i[ix]<.75) else None,'selected_success_damage':float(benefit[ix][base_i[ix]>=.75].mean()) if np.any(base_i[ix]>=.75) else None}
        result[k]=tab
    Path(a.out).write_text(json.dumps(result,indent=2))
if __name__=='__main__':
    q=argparse.ArgumentParser(); q.add_argument('--dev',required=True); q.add_argument('--raw',required=True); q.add_argument('--coco',required=True); q.add_argument('--model',required=True); q.add_argument('--out',required=True); main(q.parse_args())

