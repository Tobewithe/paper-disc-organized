"""Paired coefficient-vs-pixel residual prediction; no test labels used."""
import argparse,json
from copy import deepcopy
import torch
from common import *

def read_fit(data):
    ds=[load(p) for p in sorted((data/"fit").glob("PART_*.pt"))]
    assert ds
    return {k:sum([d[k] for d in ds],[]) if k=="keys" else torch.cat([d[k] for d in ds]) for k in ds[0]}

@torch.no_grad()
def evaluate(model,d,mode,batch=512):
    total=torch.zeros(3,device="cuda")
    for lo in range(0,len(d["h"]),batch):
        idx=torch.arange(lo,min(lo+batch,len(d["h"])),device="cuda")
        pred=effect(model(*inputs(d,idx,mode)),d["p"][idx],model.kind)
        loss,cos=normalized_loss(pred,d["target"][idx])
        total+=torch.stack([loss.sum(),cos.sum(),loss.new_tensor(len(loss))])
    return dict(loss=float(total[0]/total[2]),cos=float(total[1]/total[2]))

def main(a):
    setup();torch.manual_seed(a.seed)
    a.out.mkdir(parents=True,exist_ok=True)
    fit=read_fit(a.data);dev=load(a.data/"DEV.pt")
    assert set(k[0] for k in fit["keys"]).isdisjoint(k[0] for k in dev["keys"])
    p=fit["p"]
    z=(p*fit["c0"][:,:,None,None]).sum(1)
    norm=dict(hmean=fit["h"].mean(0),hstd=fit["h"].std(0).clamp_min(.01),
        pmean=p.mean((0,2,3))[None,:,None,None],pstd=p.std((0,2,3)).clamp_min(.01)[None,:,None,None],
        zscale=z.square().mean().sqrt().clamp_min(.1))
    fit=state_pack(fit,norm);dev=state_pack(dev,norm)
    model=Probe(a.kind).cuda()
    parameters=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best=float("inf");stale=0;history=[]
    for epoch in range(41):
        if epoch:
            model.train()
            order=torch.randperm(len(fit["h"]),generator=torch.Generator().manual_seed(10000*a.seed+epoch)).cuda()
            for lo in range(0,len(order),256):
                idx=order[lo:lo+256]
                optimizer.zero_grad(set_to_none=True)
                result=model(*inputs(fit,idx,a.mode))
                pred=effect(result,fit["p"][idx],a.kind)
                loss=normalized_loss(pred,fit["target"][idx])[0].mean()
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
                optimizer.step()
        model.eval()
        ds=evaluate(model,dev,a.mode);ts=evaluate(model,fit,a.mode)
        history.append(dict(epoch=epoch,fit=ts,dev=ds))
        if ds["loss"]<best-1e-5:
            best=ds["loss"];selected=epoch;stale=0
            torch.save(dict(state=deepcopy(model.state_dict()),norm=norm,kind=a.kind,mode=a.mode,
                seed=a.seed,selected_epoch=epoch,dev=ds,fit=ts,parameters=parameters),a.out/"BEST.pt")
        else:stale+=1
        row=dict(epoch=epoch,fit=ts,dev=ds,selected_epoch=selected,mode=a.mode,kind=a.kind,seed=a.seed)
        print(json.dumps(row),flush=True)
        write(a.out/"TRAINING.json",dict(history=history,selected_epoch=selected,parameters=parameters))
        if stale>=8:break
    write(a.out/"COMPLETE.json",dict(kind=a.kind,mode=a.mode,seed=a.seed,parameters=parameters,
        selected_epoch=selected,best_dev_loss=best,fit_candidates=len(fit["h"]),dev_candidates=len(dev["h"]),
        fit_wrong_instance_fallback=fit["fallback"],dev_wrong_instance_fallback=dev["fallback"]))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--mode",choices=MODES,required=True)
    p.add_argument("--kind",choices=KINDS,required=True);p.add_argument("--seed",type=int,required=True)
    main(p.parse_args())

