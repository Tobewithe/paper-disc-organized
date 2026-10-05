"""7Q shared paths, ROI operator, records and paired probe."""
import json, math, uuid
from pathlib import Path
from datetime import datetime, timezone
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path("/root/coefficient_pixel_direction_20260927")
PRIOR=Path("/root/coefficient_expected_benefit_20260927")
BASE=Path("/root/autodl-tmp")
BANK=BASE/"coefficient_data_scaling_20260924/runs/RUN_a24f6c759f714fb9bfcd5a59bb04e6d3"
OLD_BANK=BASE/"coefficient_predictability_20260924/runs/RUN_d2e3941a644c4cdbbf43405b3b54035f"
OFFICIAL=BASE/"coefficient_predictability_20260924"
ORACLE=BASE/"coefficient_finite_oracle_20260925"
WEIGHTS=OFFICIAL/"yolo26m-seg.pt"
TARGETS=PRIOR/"runs/RUN_9ad188b832fd9bc49c0aafc7a61de989"
DEV_TARGETS=BASE/"coefficient_direction_predictability_20260925/runs/RUN_3b1968a2fb1b41bfb427c6e1d08ab93e/DATA.pt"
TEST_LABELS=PRIOR/"runs/RUN_8aa5feb68a444c66a2b2ba65e0de12c3"
MODES=("h","z","p","wrong_instance","wrong_image")
KINDS=("coefficient","spatial")

def load(p):
    return torch.load(p,map_location="cpu",weights_only=True)

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=True),encoding="utf-8")

def record(out,stage,status,**extra):
    old=json.loads((out/"run.json").read_text()) if (out/"run.json").exists() else {}
    write(out/"run.json",dict(old,run_id=out.name,stage=stage,status=status,**extra))

def setup():
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

def bounds(box,height=160,width=160):
    x1=max(0,min(width-1,math.floor(float(box[0])*width/640)))
    y1=max(0,min(height-1,math.floor(float(box[1])*height/640)))
    x2=max(x1+1,min(width,math.ceil(float(box[2])*width/640)))
    y2=max(y1+1,min(height,math.ceil(float(box[3])*height/640)))
    return x1,y1,x2,y2

def roi(feature,box):
    x1,y1,x2,y2=bounds(box,*feature.shape[-2:])
    return F.adaptive_avg_pool2d(feature[:,y1:y2,x1:x2],(16,16))

def assemble(rows):
    keys=["h","c0","delta","p","box"]
    packed={k:torch.stack([r[k] for r in rows]).float() for k in keys}
    packed["keys"]=[(int(r["image_id"]),int(r["annotation_id"]),int(r["raw_id"])) for r in rows]
    packed["level"]=torch.tensor([r["level"] for r in rows])
    assert len(set(packed["keys"]))==len(rows)
    return packed

def donors(d):
    """Nearest predicted area among same-level donors; no target or quality labels."""
    from collections import defaultdict
    import numpy as np
    ids=np.array([k[0] for k in d["keys"]])
    levels=d["level"].numpy()
    b=d["box"].numpy(); area=np.log(np.maximum((b[:,2]-b[:,0])*(b[:,3]-b[:,1]),1e-6))
    same=defaultdict(list)
    for i,iid in enumerate(ids): same[int(iid)].append(i)
    groups={l:np.where(levels==l)[0][np.argsort(area[levels==l],kind="stable")] for l in set(levels)}
    wrong_image=np.empty(len(ids),dtype=np.int64)
    wrong_instance=np.empty(len(ids),dtype=np.int64)
    fallback=0
    for i in range(len(ids)):
        group=groups[levels[i]]
        pos=int(np.searchsorted(area[group],area[i]))
        candidates=group[max(0,pos-32):pos+33]
        candidates=candidates[ids[candidates]!=ids[i]]
        if not len(candidates): candidates=np.where(ids!=ids[i])[0]
        assert len(candidates)
        other=int(candidates[np.argmin(np.abs(area[candidates]-area[i]))])
        wrong_image[i]=other
        options=[j for j in same[int(ids[i])] if j!=i]
        if options:
            wrong_instance[i]=min(options,key=lambda j:(levels[j]!=levels[i],abs(area[j]-area[i]),j))
        else:
            wrong_instance[i]=other;fallback+=1
    assert np.all(ids[wrong_image]!=ids)
    return torch.from_numpy(wrong_instance),torch.from_numpy(wrong_image),fallback

class Probe(nn.Module):
    def __init__(self,kind):
        super().__init__()
        width=317 if kind=="coefficient" else 256
        size=32 if kind=="coefficient" else 256
        self.kind=kind
        self.encoder=nn.Sequential(nn.Conv2d(32,16,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(16,16,3,stride=2,padding=1),nn.SiLU(),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(64+256,width),nn.SiLU(),
            nn.Linear(width,width),nn.SiLU(),nn.Linear(width,size))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,h,spatial):
        return self.head(torch.cat([h,self.encoder(spatial)],1))

def normalized_loss(pred,target):
    power=target.square().mean(1).clamp_min(.01)
    error=(pred-target).square().mean(1)/power
    # Smooth denominator gives finite zero-correction gradients.
    cosine=(pred*target).sum(1)/((pred.square().sum(1)+.01)*(target.square().sum(1)+.01)).sqrt()
    return error+.25*(1-cosine),cosine

def effect(output,p,kind):
    return torch.bmm(p.flatten(2).transpose(1,2),output[:,:,None]).squeeze(-1) if kind=="coefficient" else output

def state_pack(d,norm):
    wi,wj,fallback=donors(d)
    p=d["p"].cuda()
    return dict(h=((d["h"]-norm["hmean"])/norm["hstd"]).cuda(),p=p,
        z=(p*d["c0"].cuda()[:,:,None,None]).sum(1,keepdim=True),
        target=(p*d["delta"].cuda()[:,:,None,None]).sum(1).flatten(1) if "delta" in d else None,
        wrong_instance=wi.cuda(),wrong_image=wj.cuda(),fallback=fallback,
        pmean=norm["pmean"].cuda(),pstd=norm["pstd"].cuda(),zscale=norm["zscale"].cuda())

def inputs(data,idx,mode):
    h=data["h"][idx]
    if mode=="h": spatial=torch.zeros(len(idx),32,16,16,device="cuda")
    elif mode=="z":
        spatial=torch.zeros(len(idx),32,16,16,device="cuda")
        spatial[:,0:1]=data["z"][idx]/data["zscale"]
    else:
        choose=idx if mode=="p" else data[mode][idx]
        spatial=(data["p"][choose]-data["pmean"])/data["pstd"]
    return h,spatial

