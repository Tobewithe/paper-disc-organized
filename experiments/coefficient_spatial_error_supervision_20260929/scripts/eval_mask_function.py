import argparse,json,sys,math
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).parent))
from train_full import load
from train_representation import Probe
from train_fusion import Fusion
from train_capacity import BigGlobal

def bounds(box):
    x1=max(0,min(159,math.floor(float(box[0])*160/640))); y1=max(0,min(159,math.floor(float(box[1])*160/640)))
    x2=max(x1+1,min(160,math.ceil(float(box[2])*160/640))); y2=max(y1+1,min(160,math.ceil(float(box[3])*160/640)))
    return x1,y1,x2,y2

def auc_binary(y,s):
    y=np.asarray(y).astype(np.uint8); s=np.asarray(s); pos=(y==1); neg=(y==0)
    np1=int(pos.sum()); nn=int(neg.sum())
    if np1==0 or nn==0: return float('nan')
    order=np.argsort(s,kind='mergesort'); ranks=np.empty(len(s),dtype=float); ranks[order]=np.arange(1,len(s)+1)
    return float((ranks[pos].sum()-np1*(np1+1)/2)/(np1*nn))

def main(a):
    dev=load(a.dev); w=load(a.weights)
    raw=Path(a.raw); coco=__import__('pycocotools.coco',fromlist=['COCO']).COCO(str(a.coco))
    h=torch.stack([r['h'].float() for r in dev]); p=torch.stack([r['p'].float() for r in dev]); d=torch.stack([r['delta'].float() for r in dev])
    # p in the data is the same 16x16 ROI used by all predictors
    z0=(p*torch.stack([r['c0'].float() for r in dev])[:,:,None,None]).sum(1).flatten(1)
    specs=[('h','h',Path(a.rep)/'h.pt'),('global','global',Path(a.rep)/'global.pt'),('local','local',Path(a.rep)/'local.pt'),('fusion','fusion',Path(a.fusion)/'fusion.pt'),('big_global','big',Path(a.big)/'big_global.pt')]
    preds={}
    for name,mode,path in specs:
        ck=torch.load(path,map_location='cpu',weights_only=True); norm=ck['norm']
        model=Fusion(h.shape[1]).cuda() if mode=='fusion' else BigGlobal(h.shape[1]).cuda() if mode=='big' else Probe(h.shape[1],mode).cuda()
        model.load_state_dict(ck['state']); model.eval(); hh=((h-norm['hm'])/norm['hs']).cuda(); pp=((p-norm['pm'])/norm['ps']).cuda()
        with torch.no_grad(): preds[name]=torch.cat([model(hh[i:i+512],pp[i:i+512]).cpu() for i in range(0,len(dev),512)])
    per={k:[] for k in ['baseline','h','global','local','fusion','big_global']}
    for idx,r in enumerate(dev):
        ann=coco.anns[int(r['annotation_id'])]; gt=torch.from_numpy(coco.annToMask(ann).astype('float32'))[None,None]
        box=r['box']; x1,y1,x2,y2=bounds(box); gt=F.interpolate(gt,(160,160),mode='nearest')[0,0][y1:y2,x1:x2]
        gt=F.interpolate(gt[None,None],(16,16),mode='nearest')[0,0].flatten().numpy().astype(np.uint8)
        base=z0[idx].numpy()
        for name,delta in [('baseline',np.zeros(256)),*[(k,preds[k][idx].numpy()) for k in ['h','global','local','fusion','big_global']]]:
            z=base+delta; auc=auc_binary(gt,z); pr=z>0; inter=(pr & (gt>0)).sum(); union=(pr | (gt>0)).sum(); iou=float(inter/union) if union else 1.; cov=float(inter/max((gt>0).sum(),1)); fp=float((pr & (gt==0)).sum()/max((gt==0).sum(),1))
            per[name].append({'auc':auc,'iou':iou,'coverage':cov,'fpr':fp})
    summary={}
    for name,rows in per.items():
        summary[name]={m:float(np.nanmean([x[m] for x in rows])) for m in ['auc','iou','coverage','fpr']}
        summary[name]['mask75_rate']=float(np.mean([x['iou']>=.75 for x in rows]))
    Path(a.out).write_text(json.dumps({'n':len(dev),'summary':summary},indent=2))
if __name__=='__main__':
    q=argparse.ArgumentParser(); q.add_argument('--dev',required=True); q.add_argument('--weights',required=True); q.add_argument('--raw',required=True); q.add_argument('--coco',required=True); q.add_argument('--rep',required=True); q.add_argument('--fusion',required=True); q.add_argument('--big',required=True); q.add_argument('--out',required=True); main(q.parse_args())

