import argparse, json, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO

class Probe(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(32,16,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(16,16,3,stride=2,padding=1),nn.SiLU(),
            nn.Flatten())
        self.head=nn.Sequential(nn.Linear(hdim+256,256),nn.SiLU(),
                                nn.Linear(256,256),nn.SiLU(),
                                nn.Linear(256,256))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
    def forward(self,h,p): return self.head(torch.cat([h,self.encoder(p)],1))

def load(path):
    return torch.load(path,map_location='cpu',weights_only=True)

def bounds(box):
    x1=max(0,min(639,math.floor(float(box[0])))); y1=max(0,min(639,math.floor(float(box[1]))))
    x2=max(x1+1,min(640,math.ceil(float(box[2])))); y2=max(y1+1,min(640,math.ceil(float(box[3]))))
    return x1,y1,x2,y2

def make_weights(rows,coco_path,cache):
    if cache.exists(): return torch.load(cache,map_location='cpu',weights_only=True)
    coco=COCO(str(coco_path)); out=[]
    for r in rows:
        aid=int(r['annotation_id']); ann=coco.anns[aid]
        m=torch.from_numpy(coco.annToMask(ann).astype('float32'))[None,None]
        m=F.interpolate(m,(640,640),mode='bilinear',align_corners=False)[0,0]
        dil=F.max_pool2d(m[None,None],5,1,2)[0,0]
        ero=-F.max_pool2d((-m)[None,None],5,1,2)[0,0]
        bd=(dil-ero).clamp(0,1)
        x1,y1,x2,y2=bounds(r['box']); crop=bd[y1:y2,x1:x2][None,None]
        w=F.interpolate(crop,(16,16),mode='bilinear',align_corners=False)[0,0]
        w=(1.0+2.0*w); w=w/w.mean().clamp_min(1e-6)
        out.append(w)
    w=torch.stack(out).float()
    cache.parent.mkdir(parents=True,exist_ok=True); torch.save(w,cache); return w

def pack(rows,norm,weights):
    h=torch.stack([r['h'].float() for r in rows])
    p=torch.stack([r['p'].float() for r in rows])
    d=torch.stack([r['delta'].float() for r in rows])
    target=(p*d[:,:,None,None]).sum(1).flatten(1)
    h=((h-norm['hm'])/norm['hs']).cuda(non_blocking=True)
    p=((p-norm['pm'])/norm['ps']).cuda(non_blocking=True)
    target=target.cuda(non_blocking=True); weights=weights.flatten(1).cuda(non_blocking=True)
    return h,p,target,weights

def metrics(model,data,weighted):
    h,p,t,w=data; model.eval(); vals=[]
    with torch.no_grad():
        for lo in range(0,len(h),512):
            sl=slice(lo,min(lo+512,len(h)))
            pred=model(h[sl],p[sl]); ww=w[sl] if weighted else torch.ones_like(w[sl])
            mse=((pred-t[sl]).square()*ww).mean(1)/(t[sl].square().mean(1).clamp_min(.01))
            cos=(pred*t[sl]).sum(1)/((pred.square().sum(1)+.01)*(t[sl].square().sum(1)+.01)).sqrt()
            # Use a binary region mask for diagnostics. The normalized loss
            # weights have zero-mean offset, so w-1 is not a valid region mask.
            bw=(w[sl]>1.05).float(); bn=bw.sum(1).clamp_min(1.0)
            bc=(pred*t[sl]*bw).sum(1)/(((pred.square()*bw).sum(1)+.01*bn)*((t[sl].square()*bw).sum(1)+.01*bn)).sqrt()
            iw=1.0-bw; inn=(pred*t[sl]*iw).sum(1)/(((pred.square()*iw).sum(1)+.01*iw.sum(1))*((t[sl].square()*iw).sum(1)+.01*iw.sum(1))).sqrt()
            vals.append(torch.stack([mse,cos,bc,inn],1).cpu())
    x=torch.cat(vals)
    return {'mse':float(x[:,0].mean()),'cos':float(x[:,1].mean()),
            'boundary_cos':float(x[:,2].mean()),'interior_cos':float(x[:,3].mean())}

def main(a):
    torch.set_num_threads(8); torch.manual_seed(a.seed)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    fit=load(a.fit); dev=load(a.dev)
    if not fit or not dev: raise RuntimeError('empty full subset')
    h=torch.stack([r['h'].float() for r in fit]); p=torch.stack([r['p'].float() for r in fit])
    norm={'hm':h.mean(0),'hs':h.std(0).clamp_min(.01),
          'pm':p.mean((0,2,3))[None,:,None,None],
          'ps':p.std((0,2,3)).clamp_min(.01)[None,:,None,None]}
    wf=make_weights(fit,Path(a.coco),out/'FIT_WEIGHTS.pt')
    wd=make_weights(dev,Path(a.coco),out/'DEV_WEIGHTS.pt')
    fitpack=pack(fit,norm,wf); devpack=pack(dev,norm,wd)
    init=Probe(h.shape[1]).state_dict(); records={}
    for arm,weighted in [('uniform',False),('boundary_weighted',True)]:
        model=Probe(h.shape[1]).cuda(); model.load_state_dict(init)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        best=1e9; hist=[]
        for epoch in range(1,a.epochs+1):
            model.train(); order=torch.randperm(len(fitpack[0]),device='cuda')
            for lo in range(0,len(order),512):
                idx=order[lo:lo+512]; pred=model(fitpack[0][idx],fitpack[1][idx])
                t=fitpack[2][idx]; w=fitpack[3][idx] if weighted else torch.ones_like(fitpack[3][idx])
                mse=((pred-t).square()*w).mean(1)/(t.square().mean(1).clamp_min(.01))
                cos=(pred*t).sum(1)/((pred.square().sum(1)+.01)*(t.square().sum(1)+.01)).sqrt()
                loss=(mse+.25*(1-cos)).mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),10); opt.step()
            fs=metrics(model,fitpack,weighted); ds=metrics(model,devpack,weighted)
            rec={'epoch':epoch,'fit':fs,'dev':ds}; hist.append(rec)
            print(json.dumps({'arm':arm,'epoch':epoch,'dev':ds}),flush=True)
            if ds['mse']<best:
                best=ds['mse']; torch.save({'state':model.state_dict(),'norm':norm,'arm':arm,'epoch':epoch},out/(arm+'.pt'))
        records[arm]={'best_dev_mse':best,'history':hist}
    (out/'RESULTS.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (out/'COMPLETE.json').write_text(json.dumps({'status':'completed','fit':len(fit),'dev':len(dev),
                                                   'epochs':a.epochs,'seed':a.seed},indent=2),encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--fit',required=True); p.add_argument('--dev',required=True)
    p.add_argument('--coco',required=True); p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=12); p.add_argument('--seed',type=int,default=0)
    main(p.parse_args())

