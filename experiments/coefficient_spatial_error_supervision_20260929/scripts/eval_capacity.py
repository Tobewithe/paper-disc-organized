import argparse,json,sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0,str(Path(__file__).parent))
from train_full import load
from train_fusion import Fusion
from train_capacity import BigGlobal

def rows(pred,t,m):
    n=m.sum(1).clamp_min(1.); den=(((pred.square()*m).sum(1)+.01*n)*((t.square()*m).sum(1)+.01*n)).sqrt()
    return (pred*t*m).sum(1)/den

def main(a):
    dev=load(a.dev); w=load(a.weights).flatten(1); b=(w>1.05).float(); i=1-b
    h=torch.stack([r['h'].float() for r in dev]); p=torch.stack([r['p'].float() for r in dev]); d=torch.stack([r['delta'].float() for r in dev]); t=(p*d[:,:,None,None]).sum(1).flatten(1)
    specs=[('fusion',Fusion(h.shape[1]),Path(a.fusion)/'fusion.pt'),('big_global',BigGlobal(h.shape[1]),Path(a.big)/'big_global.pt')]
    out={}; arr={}
    for name,model,path in specs:
        ck=torch.load(path,map_location='cpu',weights_only=True); norm=ck['norm']; model=model.cuda(); model.load_state_dict(ck['state']); model.eval()
        hh=((h-norm['hm'])/norm['hs']).cuda(); pp=((p-norm['pm'])/norm['ps']).cuda()
        with torch.no_grad(): pred=model(hh,pp).cpu()
        arr[name]={m:rows(pred,t,mm).numpy() for m,mm in [('cos',torch.ones_like(b)),('boundary',b),('interior',i)]}
        out[name]={k:float(np.nanmean(v)) for k,v in arr[name].items()}; out[name]['mse']=float((pred-t).square().mean())
    rng=np.random.default_rng(0); dif={}
    for m in ['cos','boundary','interior']:
        dlt=arr['fusion'][m]-arr['big_global'][m]; bs=[float(np.nanmean(dlt[rng.integers(0,len(dlt),len(dlt))])) for _ in range(2000)]
        dif[m]={'mean':float(np.nanmean(dlt)),'ci95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))]}
    out['fusion_minus_big_global']=dif
    Path(a.out).write_text(json.dumps({'n':len(t),'summary':out},indent=2))
if __name__=='__main__':
    q=argparse.ArgumentParser(); q.add_argument('--dev',required=True); q.add_argument('--weights',required=True); q.add_argument('--fusion',required=True); q.add_argument('--big',required=True); q.add_argument('--out',required=True); main(q.parse_args())

