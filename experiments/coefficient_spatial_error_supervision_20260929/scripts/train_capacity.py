import argparse,json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_full import load

class BigGlobal(nn.Module):
    def __init__(self,hdim,w=560):
        super().__init__()
        self.enc=nn.Sequential(nn.Conv2d(32,16,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(16,16,3,stride=2,padding=1),nn.SiLU(),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(hdim+256,w),nn.SiLU(),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,256))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
    def forward(self,h,p):
        p=F.interpolate(F.adaptive_avg_pool2d(p,(4,4)),size=(16,16),mode='nearest')
        return self.head(torch.cat([h,self.enc(p)],1))

def metric(model,h,p,t,bw):
    model.eval()
    with torch.no_grad(): pred=model(h,p).cpu()
    t=t.cpu(); m=(bw.cpu().flatten(1)>1.05).float(); i=1-m
    def c(q):
        n=q.sum(1).clamp_min(1.); den=(((pred.square()*q).sum(1)+.01*n)*((t.square()*q).sum(1)+.01*n)).sqrt()
        return float(((pred*t*q).sum(1)/den).mean())
    return {'mse':float((pred-t).square().mean()),'cos':c(torch.ones_like(m)),'boundary_cos':c(m),'interior_cos':c(i)}

def main(a):
    torch.set_num_threads(8); torch.manual_seed(a.seed); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    fit=load(a.fit); dev=load(a.dev); wf=load(a.fit_weights); wd=load(a.dev_weights)
    h=torch.stack([r['h'].float() for r in fit]); p=torch.stack([r['p'].float() for r in fit]); d=torch.stack([r['delta'].float() for r in fit]); t=(p*d[:,:,None,None]).sum(1).flatten(1)
    hd=torch.stack([r['h'].float() for r in dev]); pd=torch.stack([r['p'].float() for r in dev]); dd=torch.stack([r['delta'].float() for r in dev]); td=(pd*dd[:,:,None,None]).sum(1).flatten(1)
    norm={'hm':h.mean(0),'hs':h.std(0).clamp_min(.01),'pm':p.mean((0,2,3))[None,:,None,None],'ps':p.std((0,2,3)).clamp_min(.01)[None,:,None,None]}
    h=((h-norm['hm'])/norm['hs']).cuda(); p=((p-norm['pm'])/norm['ps']).cuda(); t=t.cuda(); hd=((hd-norm['hm'])/norm['hs']).cuda(); pd=((pd-norm['pm'])/norm['ps']).cuda(); td=td.cuda(); wd=wd.cuda()
    model=BigGlobal(h.shape[1]).cuda(); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); hist=[]; best=1e9
    for ep in range(1,a.epochs+1):
        model.train(); order=torch.randperm(len(h),device='cuda')
        for lo in range(0,len(order),512):
            idx=order[lo:lo+512]; pred=model(h[idx],p[idx]); tt=t[idx]
            mse=(pred-tt).square().mean(1)/(tt.square().mean(1).clamp_min(.01)); cs=(pred*tt).sum(1)/((pred.square().sum(1)+.01)*(tt.square().sum(1)+.01)).sqrt()
            loss=(mse+.25*(1-cs)).mean(); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),10); opt.step()
        ds=metric(model,hd,pd,td,wd); hist.append({'epoch':ep,'dev':ds}); print(json.dumps({'epoch':ep,'dev':ds}),flush=True)
        if ds['mse']<best: best=ds['mse']; torch.save({'state':model.state_dict(),'norm':norm,'epoch':ep},out/'big_global.pt')
    (out/'RESULTS.json').write_text(json.dumps({'big_global':{'params':sum(x.numel() for x in model.parameters()),'best_dev_mse':best,'history':hist}},indent=2)); (out/'COMPLETE.json').write_text(json.dumps({'status':'completed','fit':len(fit),'dev':len(dev),'epochs':a.epochs,'seed':a.seed},indent=2))
if __name__=='__main__':
    q=argparse.ArgumentParser(); q.add_argument('--fit',required=True); q.add_argument('--dev',required=True); q.add_argument('--fit-weights',required=True); q.add_argument('--dev-weights',required=True); q.add_argument('--out',required=True); q.add_argument('--epochs',type=int,default=12); q.add_argument('--seed',type=int,default=0); main(q.parse_args())

