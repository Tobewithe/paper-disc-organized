"""Matched h-only and prototype-conditioned probes of a frozen coefficient head."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
from pycocotools.coco import COCO
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ultralytics.utils import ops


class Probe(nn.Module):
    def __init__(self, arm, h_mean, h_std, p_mean, p_std, c_scale):
        super().__init__()
        self.arm = arm
        for name, tensor in (("h_mean",h_mean),("h_std",h_std),("p_mean",p_mean),("p_std",p_std),("c_scale",c_scale)):
            self.register_buffer(name,tensor)
        self.phi = nn.ModuleList([nn.Sequential(nn.Linear(512,128),nn.SiLU()) for _ in range(3)]) if arm.startswith("condition") else None
        width = 192 if self.phi is not None else 64
        hidden = 512 if arm=="mlp_large" else 256
        self.heads = nn.ModuleList([nn.Linear(64,32) if arm=="linear" else
            nn.Sequential(nn.Linear(width,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,32)) for _ in range(3)])
        for head in self.heads:
            last = head if arm=="linear" else head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self,h,phi,levels):
        h=(h-self.h_mean)/self.h_std
        phi=(phi-self.p_mean)/self.p_std
        if self.arm=="condition_mean":
            phi=torch.zeros_like(phi)
        output=torch.zeros(len(h),32,device=h.device)
        for level in range(3):
            indices=torch.where(levels==level)[0]
            if not len(indices):
                continue
            x=h[indices]
            if self.phi is not None:
                x=torch.cat([x,self.phi[level](phi[indices])],dim=1)
            output[indices]=self.heads[level](x)
        return output*self.c_scale


def collate(rows):
    longest=max(len(r["y"]) for r in rows)
    p=torch.zeros(len(rows),longest,32)
    y=torch.zeros(len(rows),longest)
    valid=torch.zeros_like(y)
    for k,r in enumerate(rows):
        n=len(r["y"])
        p[k,:n]=r["p"]
        y[k,:n]=r["y"]
        valid[k,:n]=r["factor"]/n
    return dict(p=p,y=y,weight=valid,h=torch.stack([r["h"] for r in rows]),
                phi=torch.stack([r["phi"] for r in rows]),c=torch.stack([r["c"] for r in rows]),
                levels=torch.tensor([r["meta"]["level"] for r in rows]))


def objective(net,batch):
    b={k:v.cuda() for k,v in batch.items()}
    c=b["c"]+net(b["h"],b["phi"],b["levels"])
    z=torch.bmm(b["p"],c[:,:,None])[:,:,0]
    return (F.binary_cross_entropy_with_logits(z,b["y"],reduction="none")*b["weight"]).sum(1)


def mismatched(rows):
    image_phi=defaultdict(list)
    for r in rows:
        image_phi[r["meta"]["image_id"]].append(r["phi"])
    ids=sorted(image_phi)
    random.Random(20260924).shuffle(ids)
    donor={ids[i]:ids[(i+1)%len(ids)] for i in range(len(ids))}
    assert len(ids)>1
    # Global conditioning maps are identical within an image. ROI maps are not:
    # cycle over donor instances instead of giving every recipient its last ROI.
    counts=defaultdict(int)
    output=[]
    for r in rows:
        iid=r["meta"]["image_id"]
        options=image_phi[donor[iid]]
        output.append({**r,"phi":options[counts[iid]%len(options)]})
        counts[iid]+=1
    return output


def decoded_iou(coeff,proto,boxes,shape,targets):
    z=(coeff@proto.flatten(1)).reshape(-1,*proto.shape[-2:])
    z=F.interpolate(z[None],(640,640),mode="bilinear",align_corners=False)[0]
    masks=ops.scale_masks(ops.crop_mask(z,boxes).gt(0).byte()[None],shape)[0].cpu().numpy()>0
    return [float(np.logical_and(m,y).sum()/max(np.logical_or(m,y).sum(),1)) for m,y in zip(masks,targets)]


def evaluate_models(a,bank,models):
    coco=COCO(str(a.annotations))
    groups=defaultdict(list)
    for r in bank:
        groups[r["meta"]["image_id"]].append(r)
    shuffled={r["meta"]["annotation_id"]:r["phi"] for r in mismatched(bank)}
    rows=[]
    for number,(iid,group) in enumerate(groups.items(),1):
        image=torch.load(a.bank/"images"/f"{iid:012d}.pt",weights_only=False,map_location="cpu")
        ids=torch.tensor([r["meta"]["raw_id"] for r in group])
        proto=image["proto"].cuda()
        c=image["coeff"][ids].cuda()
        boxes=image["boxes"][ids].cuda()
        h=torch.stack([r["h"] for r in group]).cuda()
        phi=torch.stack([r["phi"] for r in group]).cuda()
        levels=torch.tensor([r["meta"]["level"] for r in group],device="cuda")
        targets=[coco.annToMask(coco.anns[r["meta"]["annotation_id"]]).astype(bool) for r in group]
        values={}
        with torch.no_grad():
            coefficients={"original":c}
            for arm,net in models.items():
                this_phi=phi
                if net.arm=="condition_shuffle":
                    this_phi=torch.stack([shuffled[r["meta"]["annotation_id"]] for r in group]).cuda()
                coefficients[arm]=c+net(h,this_phi,levels)
            for arm,coeff in coefficients.items():
                output=[]
                for first in range(0,len(c),16):
                    output.extend(decoded_iou(coeff[first:first+16],proto,boxes[first:first+16],image["original_shape"],targets[first:first+16]))
                values[arm]=output
        for i,r in enumerate(group):
            rows.append({**r["meta"],"iou":{arm:v[i] for arm,v in values.items()}})
        if number%25==0:
            print(json.dumps(dict(stage="eval",images=number,total=len(groups))),flush=True)
    result={}
    for arm in ("original",*models):
        result[arm]=dict(n=len(rows),mean_iou=float(np.mean([r["iou"][arm] for r in rows])),
            mask75=sum(r["iou"][arm]>=.75 for r in rows),
            repairs=sum(r["iou"]["original"]<.75<=r["iou"][arm] for r in rows),
            damages=sum(r["iou"][arm]<.75<=r["iou"]["original"] for r in rows))
    (a.out/"VAL_ROWS.json").write_text(json.dumps(rows,indent=2))
    (a.out/"VAL_SUMMARY.json").write_text(json.dumps(result,indent=2))
    return result


def main(a):
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    fit=torch.load(a.bank/"fit.pt",weights_only=False,map_location="cpu")
    dev=torch.load(a.bank/"dev.pt",weights_only=False,map_location="cpu")
    val=torch.load(a.bank/"val.pt",weights_only=False,map_location="cpu")
    h=torch.stack([r["h"] for r in fit])
    phi=torch.stack([r["phi"] for r in fit])
    cs=torch.stack([r["c"] for r in fit]).std(0).clamp_min(.1)
    stats=(h.mean(0),h.std(0).clamp_min(.01),phi.mean(0),phi.std(0).clamp_min(.01),cs)
    arms=tuple(a.arms.split(","))
    assert all(arm in ("linear","mlp","mlp_large","condition_mean","condition_true","condition_shuffle") for arm in arms)
    models={}
    histories={}
    start=time.monotonic()
    for arm in arms:
        torch.manual_seed(0)
        net=Probe(arm,*stats).cuda()
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        fit_rows=mismatched(fit) if arm=="condition_shuffle" else fit
        dev_rows=mismatched(dev) if arm=="condition_shuffle" else dev
        loader=DataLoader(fit_rows,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(0),collate_fn=collate)
        dev_loader=DataLoader(dev_rows,batch_size=32,collate_fn=collate)
        best=float("inf")
        patience=0
        history=[]
        for epoch in range(1,a.epochs+1):
            net.train()
            train_losses=[]
            for batch in loader:
                value=objective(net,batch).mean()
                assert torch.isfinite(value)
                optimizer.zero_grad()
                value.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),10,error_if_nonfinite=True)
                optimizer.step()
                train_losses.append(float(value.detach()))
            net.eval()
            with torch.no_grad():
                dev_loss=torch.cat([objective(net,b).cpu() for b in dev_loader]).mean().item()
            state=dict(arm=arm,epoch=epoch,train_loss=float(np.mean(train_losses)),dev_loss=dev_loss,elapsed=time.monotonic()-start)
            history.append(state)
            print(json.dumps(state),flush=True)
            (a.out/"PROGRESS.json").write_text(json.dumps(state))
            torch.save(dict(arm=arm,state_dict=net.state_dict(),epoch=epoch,dev_loss=dev_loss),a.out/f"{arm}_epoch{epoch}.pt")
            if dev_loss<best:
                best=dev_loss
                patience=0
                torch.save(dict(arm=arm,state_dict=net.state_dict(),epoch=epoch,dev_loss=dev_loss),a.out/f"{arm}_best.pt")
            else:
                patience+=1
            if patience>=5:
                break
        checkpoint=torch.load(a.out/f"{arm}_best.pt",weights_only=False)
        net.load_state_dict(checkpoint["state_dict"])
        models[arm]=net.eval()
        histories[arm]=dict(history=history,chosen_epoch=checkpoint["epoch"],parameters=sum(p.numel() for p in net.parameters()))
        (a.out/"TRAINING.json").write_text(json.dumps(histories,indent=2))
    result=evaluate_models(a,val,{a.label_prefix+k:v for k,v in models.items()})
    (a.out/"COMPLETE.json").write_text(json.dumps(dict(arms=list(arms),val_instances=len(val),elapsed=time.monotonic()-start)))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("bank","annotations","out"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--arms",default="linear,mlp,condition_mean,condition_true,condition_shuffle")
    p.add_argument("--label-prefix",default="")
    main(p.parse_args())
