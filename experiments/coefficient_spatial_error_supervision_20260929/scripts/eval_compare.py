import argparse,json,sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0,str(Path(__file__).parent))
from train_full import load
from train_representation import Probe
from train_fusion import Fusion

def cos_rows(pred,t,m):
    n=m.sum(1).clamp_min(1.0)
    den=(((pred.square()*m).sum(1)+.01*n)*((t.square()*m).sum(1)+.01*n)).sqrt()
    return (pred*t*m).sum(1)/den

def main(a):
    torch.set_num_threads(8); dev=load(a.dev); w=load(a.weights).flatten(1); b=(w>1.05).float(); i=1-b
    h=torch.stack([r['h'].float() for r in dev]); p=torch.stack([r['p'].float() for r in dev]); d=torch.stack([r['delta'].float() for r in dev]); t=(p*d[:,:,None,None]).sum(1).flatten(1)
    specs=[('h','h',Path(a.representation)/'h.pt'),('global','global',Path(a.representation)/'global.pt'),('local','local',Path(a.representation)/'local.pt'),('fusion','fusion',Path(a.fusion)/'fusion.pt')]
    rows={}; preds={}
    for name,mode,path in specs:
        ck=torch.load(path,map_location='cpu',weights_only=True); norm=ck['norm']
        model=Fusion(h.shape[1]).cuda() if mode=='fusion' else Probe(h.shape[1],mode).cuda()
        model.load_state_dict(ck['state']); model.eval()
        hh=((h-norm['hm'])/norm['hs']).cuda(); pp=((p-norm['pm'])/norm['ps']).cuda(); 
        out=[]
        with torch.no_grad():
            for lo in range(0,len(hh),512): out.append(model(hh[lo:lo+512],pp[lo:lo+512]).cpu())
        pred=torch.cat(out); preds[name]=pred
        rows[name]={'mse':float((pred-t).square().mean()),'cos':cos_rows(pred,t,torch.ones_like(b)).tolist(),
                    'boundary_cos':cos_rows(pred,t,b).tolist(),'interior_cos':cos_rows(pred,t,i).tolist()}
    summary={}
    rng=np.random.default_rng(0); n=len(t)
    for name in rows:
        summary[name]={k:float(np.nanmean(v)) for k,v in rows[name].items() if k!='mse'}
        summary[name]['mse']=rows[name]['mse']
    for name in ['global','local','fusion']:
        diffs={}
        for metric in ['cos','boundary_cos','interior_cos']:
            x=np.asarray(rows[name][metric]); y=np.asarray(rows['global'][metric]); dlt=x-y; 
            bs=[]
            for _ in range(2000): bs.append(float(np.nanmean(dlt[rng.integers(0,n,n)])))
            diffs[metric]={'mean':float(np.nanmean(dlt)),'ci95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))]}
        summary[name+'_vs_global']=diffs
    Path(a.out).write_text(json.dumps({'n':n,'summary':summary},indent=2))
if __name__=='__main__':
    q=argparse.ArgumentParser(); q.add_argument('--dev',required=True); q.add_argument('--weights',required=True); q.add_argument('--representation',required=True); q.add_argument('--fusion',required=True); q.add_argument('--out',required=True); main(q.parse_args())

