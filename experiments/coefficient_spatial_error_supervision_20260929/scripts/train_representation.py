import argparse,json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_full import load, make_weights

class Probe(nn.Module):
    def __init__(self, hdim, mode):
        super().__init__(); self.mode=mode
        if mode != 'h':
            self.encoder=nn.Sequential(nn.Conv2d(32,16,3,stride=2,padding=1),nn.SiLU(),
                                        nn.Conv2d(16,16,3,stride=2,padding=1),nn.SiLU(),nn.Flatten())
            inp=hdim+256
        else:
            inp=hdim
        self.head=nn.Sequential(nn.Linear(inp,256),nn.SiLU(),
                                nn.Linear(256,256),nn.SiLU(),nn.Linear(256,256))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
    def forward(self,h,p):
        if self.mode=='h': x=h
        else:
            if self.mode=='global': p=F.interpolate(F.adaptive_avg_pool2d(p,(4,4)),size=(16,16),mode='nearest')
            x=torch.cat([h,self.encoder(p)],1)
        return self.head(x)

def metrics(model,data,bw):
    h,p,t=data; model.eval(); vals=[]
    with torch.no_grad():
        for lo in range(0,len(h),512): vals.append(model(h[lo:lo+512],p[lo:lo+512]).cpu())
    pred=torch.cat(vals); t=t.cpu(); bw=(bw.cpu().flatten(1)>1.05).float(); iw=1-bw
    def cos(m):
        n=m.sum(1).clamp_min(1.0)
        return ((pred*t*m).sum(1)/(((pred.square()*m).sum(1)+.01*n)*((t.square()*m).sum(1)+.01*n)).sqrt()).mean().item()
    return {'mse':float((pred-t).square().mean()),'cos':cos(torch.ones_like(bw)),
            'boundary_cos':cos(bw),'interior_cos':cos(iw)}

def main(a):
    torch.set_num_threads(8); torch.manual_seed(a.seed); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    fit=load(a.fit); dev=load(a.dev); wf=load(a.fit_weights); wd=load(a.dev_weights)
    h=torch.stack([r['h'].float() for r in fit]); p=torch.stack([r['p'].float() for r in fit]); d=torch.stack([r['delta'].float() for r in fit]); tf=(p*d[:,:,None,None]).sum(1).flatten(1)
    hd=torch.stack([r['h'].float() for r in dev]); pd=torch.stack([r['p'].float() for r in dev]); dd=torch.stack([r['delta'].float() for r in dev]); td=(pd*dd[:,:,None,None]).sum(1).flatten(1)
    norm={'hm':h.mean(0),'hs':h.std(0).clamp_min(.01),'pm':p.mean((0,2,3))[None,:,None,None],'ps':p.std((0,2,3)).clamp_min(.01)[None,:,None,None]}
    h=((h-norm['hm'])/norm['hs']).cuda(); p=((p-norm['pm'])/norm['ps']).cuda(); tf=tf.cuda()
    hd=((hd-norm['hm'])/norm['hs']).cuda(); pd=((pd-norm['pm'])/norm['ps']).cuda(); td=td.cuda()
    train=(h,p,tf); devpack=(hd,pd,td); bwf=wf.cuda(); bwd=wd.cuda(); records={}
    init={m:Probe(h.shape[1],m).state_dict() for m in ['h','global','local']}
    for mode in ['h','global','local']:
        model=Probe(h.shape[1],mode).cuda(); model.load_state_dict(init[mode]); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); best=1e9; hist=[]
        for epoch in range(1,a.epochs+1):
            model.train(); order=torch.randperm(len(h),device='cuda')
            for lo in range(0,len(order),512):
                idx=order[lo:lo+512]; pred=model(h[idx],p[idx]); t=tf[idx]
                mse=(pred-t).square().mean(1)/(t.square().mean(1).clamp_min(.01)); cos=(pred*t).sum(1)/((pred.square().sum(1)+.01)*(t.square().sum(1)+.01)).sqrt()
                loss=(mse+.25*(1-cos)).mean(); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),10); opt.step()
            ds=metrics(model,devpack,bwd); hist.append({'epoch':epoch,'dev':ds}); print(json.dumps({'mode':mode,'epoch':epoch,'dev':ds}),flush=True)
            if ds['mse']<best: best=ds['mse']; torch.save({'state':model.state_dict(),'norm':norm,'mode':mode,'epoch':epoch},out/(mode+'.pt'))
        records[mode]={'best_dev_mse':best,'history':hist}
    (out/'RESULTS.json').write_text(json.dumps(records,indent=2)); (out/'COMPLETE.json').write_text(json.dumps({'status':'completed','fit':len(fit),'dev':len(dev),'epochs':a.epochs,'seed':a.seed},indent=2))
if __name__=='__main__':
    q=argparse.ArgumentParser(); q.add_argument('--fit',required=True); q.add_argument('--dev',required=True); q.add_argument('--fit-weights',required=True); q.add_argument('--dev-weights',required=True); q.add_argument('--out',required=True); q.add_argument('--epochs',type=int,default=12); q.add_argument('--seed',type=int,default=0); main(q.parse_args())

