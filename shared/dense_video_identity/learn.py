from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from .common import Run, iou_matrix, load_gt, read_json, setup_paths, sha256, unit, write_json
from .tracking import quality_vector, run_sequence


class IdentityHead(nn.Module):
    def __init__(self, dim, hidden=128, embedding=64):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Linear(hidden,embedding))
    def forward(self,x):
        return F.normalize(self.net(x),dim=-1)


class UtilityHead(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(8,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,x):
        return self.net(x).squeeze(-1)


def match_gt(records, gt):
    mapping={}
    for rec in records:
        g=gt[rec["frame"]]
        boxes=np.asarray(rec["boxes"],np.float32).reshape(-1,4)
        sim=iou_matrix(g[:,2:6],boxes)
        if sim.size:
            rr,cc=linear_sum_assignment(-sim)
            mapping[rec["frame"]]={int(c):int(g[r,1]) for r,c in zip(rr,cc) if sim[r,c]>=.5}
        else:
            mapping[rec["frame"]]={}
    return mapping


def load_cache(root, seq):
    pointer=read_json(root/("demo_features_pointer.json" if seq["split"]=="demo" and (root/"demo_features_pointer.json").exists() else "features_pointer.json"))
    path=Path(seq["cache_path"]) if seq.get("cache_path") else root/pointer["cache"]/seq["name"]
    return read_json(path/"detections.json"),np.load(path/"features.npz")["x"]


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False


def load_identity(root, device="cpu"):
    pointer=read_json(root/"identity_pointer.json")
    ck=torch.load(root/pointer["checkpoint"],map_location="cpu",weights_only=False)
    model=IdentityHead(ck["dim"],ck["config"]["hidden"],ck["config"]["embedding"])
    model.load_state_dict(ck["state"]);model.eval().to(device)
    return model,ck


def embed_array(model, ck, raw, device="cpu"):
    x=np.clip((raw-ck["mean"])/ck["std"],-8,8).astype(np.float32)
    if len(x)==0:
        return np.empty((0,64),np.float32)
    with torch.inference_mode():
        return np.concatenate([model(torch.from_numpy(x[i:i+2048]).to(device)).cpu().numpy() for i in range(0,len(x),2048)])


def train_identity(root: Path, config: dict):
    setup_paths(root);set_seed(config["seed"])
    sequences=[s for s in read_json(root/"input_manifest.json")["sequences"] if s["split"]=="train"]
    ptr=read_json(root/"features_pointer.json")
    contract=dict(config=config["identity"],seed=config["seed"],sequences=[s["name"] for s in sequences],
                  parent_feature_run=ptr["run_id"],trainable="identity.net only; detector fully frozen",
                  labels="one-to-one bbox IoU >= .5 to official train GT", checkpoint_selection="last_step_300")
    with Run(root,"identity_train",contract,[ptr["run_id"]]) as run:
        xs,ys,frame_labels=[],[],[]
        identity_lookup={}
        for seq in sequences:
            records,raw=load_cache(root,seq)
            mapping=match_gt(records,load_gt(root,seq))
            for rec in records:
                for det,gid in mapping[rec["frame"]].items():
                    key=(seq["name"],gid)
                    label=identity_lookup.setdefault(key,len(identity_lookup))
                    xs.append(raw[rec["feature_start"]+det]);ys.append(label);frame_labels.append(rec["frame"])
        x=np.stack(xs);y=np.asarray(ys)
        mean=x.mean(0);std=np.maximum(x.std(0),.05)
        norm=np.clip((x-mean)/std,-8,8).astype(np.float32)
        groups={i:np.flatnonzero(y==i) for i in sorted(set(y)) if np.sum(y==i)>=4}
        if len(groups)<4:
            raise RuntimeError("Insufficient train identities for contrastive learning")
        cfg=config["identity"]
        model=IdentityHead(x.shape[1],cfg["hidden"],cfg["embedding"]).cuda()
        optim=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
        tx=torch.from_numpy(norm).cuda()
        rng=np.random.default_rng(config["seed"])
        curve=[]
        for step in range(cfg["steps"]):
            ids=rng.choice(list(groups),min(len(groups),cfg["identities_per_batch"]),replace=False)
            inds=np.concatenate([rng.choice(groups[i],cfg["samples_per_identity"],replace=False) for i in ids])
            labels=torch.from_numpy(y[inds]).cuda()
            z=model(tx[torch.from_numpy(inds).cuda()])
            logits=z@z.T/cfg["temperature"]
            eye=torch.eye(len(z),dtype=torch.bool,device=z.device)
            logits=logits.masked_fill(eye,-1e4)
            positive=(labels[:,None]==labels[None,:]) & ~eye
            logprob=logits-torch.logsumexp(logits,dim=1,keepdim=True)
            loss=-(logprob*positive).sum()/positive.sum()
            optim.zero_grad(set_to_none=True);loss.backward();optim.step()
            curve.append(dict(step=step+1,loss=float(loss.detach())))
            if (step+1)%50==0:
                print("IDENTITY",curve[-1],flush=True)
        ck=dict(state={k:v.cpu() for k,v in model.state_dict().items()},dim=x.shape[1],mean=mean,std=std,
                config=cfg,training_sequences=[s["name"] for s in sequences],labels=len(identity_lookup))
        torch.save(ck,run.path/"identity.pt")
        write_json(run.path/"loss_curve.json",curve)
        write_json(run.path/"SUMMARY.json",dict(samples=len(x),identities=len(groups),steps=cfg["steps"],
                   trainable_parameters=sum(p.numel() for p in model.parameters()),last_loss=curve[-1]["loss"],claim="training completed; tracking utility not yet evaluated"))
        write_json(root/"identity_pointer.json",dict(run_id=run.id,checkpoint=(run.path/"identity.pt").relative_to(root).as_posix(),sha256=sha256(run.path/"identity.pt")))
    return run.id


def train_utility(root: Path, config: dict):
    setup_paths(root);set_seed(config["seed"])
    sequences=[s for s in read_json(root/"input_manifest.json")["sequences"] if s["split"]=="train"]
    identity,ck=load_identity(root)
    contract=dict(config=config["utility"],seed=config["seed"],sequences=[s["name"] for s in sequences],
                  identity_sha256=read_json(root/"identity_pointer.json")["sha256"],
                  label="future same-ID vs competitor margin change from a hypothetical normal confidence-EMA write",
                  future_permissions="train label only; never gate input",checkpoint_selection="last_step_200",
                  actual_baseline_associations=True)
    with Run(root,"utility_train",contract,[read_json(root/"identity_pointer.json")["run_id"]]) as run:
        samples,targets=[],[]
        sample_meta=[]
        for seq in sequences:
            records,raw=load_cache(root,seq);z=embed_array(identity,ck,raw)
            mapping=match_gt(records,load_gt(root,seq))
            future={}
            for rec in records:
                future[rec["frame"]]={gid:z[rec["feature_start"]+d] for d,gid in mapping[rec["frame"]].items()}
            anchors={};snap={}
            def before(frame,tracker):
                snap.clear()
                snap.update({t.track_id:dict(memory=t.smooth_feat.copy(),length=frame-t.start_frame,gap=frame-t.end_frame)
                             for t in tracker.tracked_stracks+tracker.lost_stracks if t.smooth_feat is not None})
            def after(frame,tracker,obs,rec):
                for o in obs:
                    bid=o["base_id"];gid=mapping[frame].get(o["det_id"])
                    if bid not in anchors and gid is not None:
                        anchors[bid]=gid
                    if bid not in snap or bid not in anchors:
                        continue
                    old=snap[bid]["memory"]
                    quality=rec["quality"][o["det_id"]]
                    xx=quality_vector(old,o["feature"],quality,[s["memory"] for i,s in snap.items() if i!=bid],snap[bid]["length"],snap[bid]["gap"])
                    trust=max(0.,(o["score"]-config["tracker"]["track_high_thresh"])/(1-config["tracker"]["track_high_thresh"]))
                    beta=(1-config["tracker"]["alpha_fixed_emb"])*trust
                    new=unit((1-beta)*old+beta*o["feature"])
                    margins=[]
                    for dt in (3,6,config["utility"]["future_horizon"]):
                        fs=future.get(frame+dt,{})
                        anchor=anchors[bid]
                        if anchor not in fs or len(fs)<2:
                            continue
                        competitors=[f for g,f in fs.items() if g!=anchor]
                        uold=float(old@fs[anchor])-max(float(old@f) for f in competitors)
                        unew=float(new@fs[anchor])-max(float(new@f) for f in competitors)
                        margins.append(unew-uold)
                    if margins:
                        samples.append(xx);targets.append(float(np.mean(margins)))
                        sample_meta.append(dict(sequence=seq["name"],frame=frame,track=bid,anchor_gt=anchors[bid],current_gt=gid,wrong_association=gid is not None and gid!=anchors[bid]))
            run_sequence("deepocsort",records,z,config,on_before=before,on_after=after)
        x=np.stack(samples);y=np.asarray(targets,np.float32)
        mean=x.mean(0);std=np.maximum(x.std(0),.03)
        tx=torch.from_numpy((x-mean)/std).cuda();ty=torch.from_numpy(y*100).cuda()
        model=UtilityHead(config["utility"]["hidden"]).cuda()
        optim=torch.optim.AdamW(model.parameters(),lr=config["utility"]["lr"],weight_decay=.0001)
        rng=np.random.default_rng(config["seed"]);curve=[]
        for step in range(config["utility"]["steps"]):
            idx=rng.integers(0,len(x),size=min(256,len(x)));ii=torch.from_numpy(idx).cuda()
            pred=model(tx[ii]);loss=F.smooth_l1_loss(pred,ty[ii])
            optim.zero_grad(set_to_none=True);loss.backward();optim.step()
            curve.append(dict(step=step+1,loss=float(loss.detach())))
            if (step+1)%50==0:
                print("UTILITY",curve[-1],flush=True)
        torch.save(dict(state={k:v.cpu() for k,v in model.state_dict().items()},mean=mean,std=std,hidden=config["utility"]["hidden"],scale=100,training_sequences=[s["name"] for s in sequences]),run.path/"utility.pt")
        np.savez_compressed(run.path/"training_examples.npz",x=x,y=y)
        write_json(run.path/"training_examples_metadata.json",sample_meta)
        write_json(run.path/"loss_curve.json",curve)
        write_json(run.path/"SUMMARY.json",dict(samples=len(x),positive_labels=int(np.sum(y>0)),negative_labels=int(np.sum(y<0)),
                    wrong_association_examples=sum(s["wrong_association"] for s in sample_meta),steps=config["utility"]["steps"],
                    trainable_parameters=sum(p.numel() for p in model.parameters()),last_loss=curve[-1]["loss"],
                    claim="utility predictor trained; actual benefit requires held-out tracking comparison"))
        write_json(root/"utility_pointer.json",dict(run_id=run.id,checkpoint=(run.path/"utility.pt").relative_to(root).as_posix(),sha256=sha256(run.path/"utility.pt")))
    return run.id


def load_utility(root):
    ptr=read_json(root/"utility_pointer.json")
    ck=torch.load(root/ptr["checkpoint"],map_location="cpu",weights_only=False)
    model=UtilityHead(ck["hidden"]);model.load_state_dict(ck["state"]);model.eval()
    def predict(x):
        with torch.inference_mode():
            return model(torch.from_numpy((x-ck["mean"])/ck["std"])).item()/ck["scale"]
    return predict
