import argparse,json
from pathlib import Path
import torch
from train_full import Probe, load

def main(a):
    dev=load(a.dev); weights=load(a.weights)
    h=torch.stack([r['h'].float() for r in dev]); p=torch.stack([r['p'].float() for r in dev]); d=torch.stack([r['delta'].float() for r in dev])
    target=(p*d[:,:,None,None]).sum(1).flatten(1)
    out={}
    for arm in ['uniform','boundary_weighted']:
        ck=torch.load(Path(a.modeldir)/(arm+'.pt'),map_location='cpu',weights_only=True)
        norm=ck['norm']; hh=((h-norm['hm'])/norm['hs']).cuda(); pp=((p-norm['pm'])/norm['ps']).cuda(); tt=target.cuda()
        model=Probe(h.shape[1]).cuda(); model.load_state_dict(ck['state']); model.eval()
        preds=[]
        with torch.no_grad():
            for lo in range(0,len(hh),512): preds.append(model(hh[lo:lo+512],pp[lo:lo+512]).cpu())
        pred=torch.cat(preds); t=target
        bw=(weights>1.05).float().flatten(1); iw=1-bw; 
        def cos(a,b,m=None):
            if m is None: m=torch.ones_like(a)
            n=m.sum(1).clamp_min(1.0)
            return ((a*b*m).sum(1)/(((a.square()*m).sum(1)+.01*n)*((b.square()*m).sum(1)+.01*n)).sqrt()).mean().item()
        out[arm]={'unweighted_mse':float((pred-t).square().mean()),'cos':cos(pred,t),
                  'boundary_cos':cos(pred,t,bw),'interior_cos':cos(pred,t,iw)}
    Path(a.out).write_text(json.dumps(out,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dev',required=True); p.add_argument('--weights',required=True); p.add_argument('--modeldir',required=True); p.add_argument('--out',required=True); main(p.parse_args())

