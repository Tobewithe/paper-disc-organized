import argparse,json,sys,math
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).parent))
from train_full import load
from train_representation import Probe
from train_fusion import Fusion
from train_capacity import BigGlobal

def auc(y,s):
    y=np.asarray(y).astype(np.uint8); s=np.asarray(s); pos=y==1; neg=y==0; np1=int(pos.sum()); nn=int(neg.sum())
    if np1==0 or nn==0:return float('nan')
    o=np.argsort(s,kind='mergesort'); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    return float((r[pos].sum()-np1*(np1+1)/2)/(np1*nn))

def bd(box):
    x1=max(0,min(159,math.floor(float(box[0])*160/640))); y1=max(0,min(159,math.floor(float(box[1])*160/640)))
    x2=max(x1+1,min(160,math.ceil(float(box[2])*160/640))); y2=max(y1+1,min(160,math.ceil(float(box[3])*160/640)))
    return x1,y1,x2,y2

def calc(z,gt):
    aucv=auc(gt,z); pr=z>0; g=gt>0; inter=(pr&g).sum(); uni=(pr|g).sum()
    return {'auc':aucv,'iou':float(inter/uni) if uni else 1.,'coverage':float(inter/max(g.sum(),1)),'fpr':float((pr&(~g)).sum()/max((~g).sum(),1)),'mask75':float(inter/uni>=.75) if uni else 1.}

def main(a):
    dev=load(a.dev); raw=Path(a.raw); coco=__import__('pycocotools.coco',fromlist=['COCO']).COCO(str(a.coco))
    h=torch.stack([r['h'].float() for r in dev]); p=torch.stack([r['p'].float() for r in dev])
    specs=[('h','h',Path(a.rep)/'h.pt'),('global','global',Path(a.rep)/'global.pt'),('local','local',Path(a.rep)/'local.pt'),('fusion','fusion',Path(a.fusion)/'fusion.pt'),('big_global','big',Path(a.big)/'big_global.pt')]
    preds={}
    for name,mode,path in specs:
        ck=torch.load(path,map_location='cpu',weights_only=True); norm=ck['norm']; model=Fusion(h.shape[1]).cuda() if mode=='fusion' else BigGlobal(h.shape[1]).cuda() if mode=='big' else Probe(h.shape[1],mode).cuda(); model.load_state_dict(ck['state']); model.eval()
        hh=((h-norm['hm'])/norm['hs']).cuda(); pp=((p-norm['pm'])/norm['ps']).cuda()
        with torch.no_grad(): preds[name]=torch.cat([model(hh[j:j+512],pp[j:j+512]).cpu() for j in range(0,len(dev),512)])
    per={k:[] for k in ['baseline','h','global','local','fusion','big_global']}
    cache={}
    for idx,r in enumerate(dev):
        iid=int(r['image_id']); f=cache.get(iid)
        if f is None: f=torch.load(raw/f'{iid:012d}.pt',map_location='cpu',weights_only=True); cache[iid]=f
        rid=int(r['raw_id']); proto=f['proto'].float(); c0=f['coeff'][rid].float(); box=f['boxes'][rid].float(); x1,y1,x2,y2=bd(box)
        z0=(proto*c0[:,None,None]).sum(0)[y1:y2,x1:x2]
        gt=torch.from_numpy(coco.annToMask(coco.anns[int(r['annotation_id'])]).astype('float32'))[None,None]
        gt=F.interpolate(gt,(160,160),mode='nearest')[0,0][y1:y2,x1:x2].numpy().astype(np.uint8)
        for name,delta in [('baseline',None),*[(k,preds[k][idx].reshape(16,16)) for k in ['h','global','local','fusion','big_global']]]:
            dz=torch.zeros_like(z0) if delta is None else F.interpolate(delta[None,None],size=z0.shape,mode='bilinear',align_corners=False)[0,0]
            per[name].append(calc((z0+dz).numpy().ravel(),gt.ravel()))
    summary={k:{m:float(np.nanmean([x[m] for x in v])) for m in ['auc','iou','coverage','fpr','mask75']} for k,v in per.items()}
    rng=np.random.default_rng(0); diffs={}
    for name in ['h','global','local','fusion','big_global']:
        diffs[name+'_minus_baseline']={}
        for m in ['auc','iou','coverage','fpr','mask75']:
            d=np.asarray([x[m] for x in per[name]],dtype=float)-np.asarray([x[m] for x in per['baseline']],dtype=float)
            bs=[float(np.nanmean(d[rng.integers(0,len(d),len(d))])) for _ in range(2000)]
            diffs[name+'_minus_baseline'][m]={'mean':float(np.nanmean(d)),'ci95':[float(np.nanquantile(bs,.025)),float(np.nanquantile(bs,.975))]}
    fail=np.asarray([x['iou']<.75 for x in per['baseline']],dtype=bool)
    subgroup={}
    for subset,mask in [('baseline_mask75_failure',fail),('baseline_mask75_success',~fail)]:
        subgroup[subset]={}
        for name in ['baseline','h','global','local','fusion','big_global']:
            subgroup[subset][name]={m:float(np.nanmean([per[name][j][m] for j in np.where(mask)[0]])) for m in ['auc','iou','coverage','fpr','mask75']}
        subgroup[subset+'_n']=int(mask.sum())
    subdiff={}
    for subset,mask in [('baseline_mask75_failure',fail),('baseline_mask75_success',~fail)]:
        ix=np.where(mask)[0]; subdiff[subset+'_n']=int(len(ix)); subdiff[subset]={}
        for name in ['global','big_global']:
            subdiff[subset][name]={}
            for m in ['auc','iou','coverage','fpr','mask75']:
                d=np.asarray([per[name][j][m]-per['baseline'][j][m] for j in ix],dtype=float)
                bs=[float(np.nanmean(d[rng.integers(0,len(d),len(d))])) for _ in range(2000)]
                subdiff[subset][name][m]={'mean':float(np.nanmean(d)),'ci95':[float(np.nanquantile(bs,.025)),float(np.nanquantile(bs,.975))]}
    Path(a.out).write_text(json.dumps({'n':len(dev),'summary':summary,'paired_bootstrap':diffs,'subgroup':subgroup,'subgroup_bootstrap':subdiff},indent=2))
if __name__=='__main__':
    q=argparse.ArgumentParser(); q.add_argument('--dev',required=True); q.add_argument('--raw',required=True); q.add_argument('--coco',required=True); q.add_argument('--rep',required=True); q.add_argument('--fusion',required=True); q.add_argument('--big',required=True); q.add_argument('--out',required=True); main(q.parse_args())

