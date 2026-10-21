"""Learned set evidence + temporal accumulation. No GT in model or inference."""
from pathlib import Path
from collections import defaultdict
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import Run, read_json, write_json, sha256
from .learn import set_seed

ARMS=['full','self','now']


def intervention(own,rivals,mask,mode):
    if mode=='self':
        r=torch.stack([own[...,0],own[...,7],own[...,7]-own[...,0],own[...,1],own[...,3],own[...,2],own[...,6]],dim=-1)
        # Distinct causal self-history tokens keep set-attention parameters effective.
        # Identical copies would make attention gradients mathematically zero.
        rivals=torch.stack([r[:,torch.arange(5,device=r.device).sub(lag).clamp(min=0)] for lag in range(6)],dim=-2);mask=torch.ones_like(mask)
    elif mode=='now':
        own=own[:,-1:].expand(-1,5,-1);rivals=rivals[:,-1:].expand(-1,5,-1,-1);mask=mask[:,-1:].expand(-1,5,-1)
    return own,rivals,mask


class EvidenceDecision(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_encoder=nn.Sequential(nn.Linear(8,24),nn.SiLU())
        self.rival_encoder=nn.Sequential(nn.Linear(7,24),nn.SiLU(),nn.Linear(24,24),nn.SiLU())
        self.attention=nn.Sequential(nn.Linear(72,24),nn.SiLU(),nn.Linear(24,1,bias=False))
        self.temporal=nn.GRU(72,32,batch_first=True)
        self.dropout=nn.Dropout(.1);self.decision=nn.Linear(32,1)
    def forward(self,own,rivals,mask,mode='full'):
        own,rivals,mask=intervention(own,rivals,mask,mode)
        h=self.self_encoder(own);r=self.rival_encoder(rivals);q=h.unsqueeze(-2).expand_as(r)
        score=self.attention(torch.cat([q,r,r-q],dim=-1)).squeeze(-1)
        weight=torch.softmax(score.masked_fill(mask==0,-1e4),dim=-1)*mask
        weight=weight/weight.sum(-1,keepdim=True).clamp(min=1e-6)
        c=(weight.unsqueeze(-1)*r).sum(-2)
        seq=torch.cat([h,c,c-h],dim=-1);_,hidden=self.temporal(seq)
        return self.decision(self.dropout(hidden[-1])).squeeze(-1)


def get_data(root,split):
    ptr=read_json(root/'decision_dataset_pointer.json');base=Path(ptr['path']);manifest=read_json(root/'input_manifest.json');arrays=defaultdict(list);rows=[]
    for s in manifest['sequences']:
        if s['decision_split']!=split:continue
        rr=read_json(base/s['name']/'rows.json');a=np.load(base/s['name']/'windows.npz');keep=[i for i,r in enumerate(rr) if r['y']>=0]
        rows += [rr[i] for i in keep]
        for k in ['own','rivals','mask']:arrays[k].append(a[k][keep])
    return {k:np.concatenate(v) for k,v in arrays.items()},rows


def predict(model,arrays,mode='full',device='cpu'):
    model.eval();values=[]
    with torch.inference_mode():
        for i in range(0,len(arrays['own']),512):
            tensors=[torch.from_numpy(arrays[k][i:i+512]).to(device) for k in ['own','rivals','mask']]
            values.append(torch.sigmoid(model(*tensors,mode)).cpu().numpy())
    return np.concatenate(values) if values else np.empty(0,np.float32)


def train(root,smoke=False):
    arrays,rows=get_data(root,'fit');y=np.array([r['y'] for r in rows],np.float32);groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['sequence']+':'+r['event']].append(i)
    names=sorted(groups);pos_weight=min(float(np.sum(y==0)/max(np.sum(y==1),1)),10.)
    steps=20 if smoke else 800;paths={};device='cuda';torch.set_num_threads(4)
    contract=dict(arms=ARMS,steps=steps,batch=128,seed=20261019,lr=.0003,weight_decay=.01,loss='event-balanced sampling BCEWithLogits',positive_weight=pos_weight,whitelist='EvidenceDecision only',model_protocol_sha=sha256(root/'METHOD_PROTOCOL.md'),data=read_json(root/'decision_dataset_pointer.json'),checkpoint='last',new_parameters='recorded after construction')
    with Run(root,'decision_smoke' if smoke else 'decision_train',contract) as run:
        tx={k:torch.from_numpy(v).to(device) for k,v in arrays.items()};ty=torch.from_numpy(y).to(device);summary=[];reference_init=None;reference_batch=None
        for arm in ARMS:
            set_seed(20261019);rng=np.random.default_rng(20261019);model=EvidenceDecision().to(device)
            init=torch.cat([p.detach().flatten().cpu() for p in model.parameters()]).numpy()
            if reference_init is None:reference_init=init.copy()
            else:assert np.array_equal(init,reference_init)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);curve=[];batch_trace=[];start=time.perf_counter();model.train();gradcheck={}
            for step in range(steps):
                ids=[int(rng.choice(groups[names[j]])) for j in rng.integers(0,len(names),128)];batch_trace.append(ids)
                index=torch.tensor(ids,device=device);pred=model(tx['own'][index],tx['rivals'][index],tx['mask'][index],arm)
                loss=F.binary_cross_entropy_with_logits(pred,ty[index],pos_weight=torch.tensor(pos_weight,device=device));assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if step==0:
                    gradcheck={n:bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0) for n,p in model.named_parameters()};assert all(gradcheck.values()),gradcheck
                nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                if step%100==0 or step==steps-1:curve.append(dict(step=step+1,loss=float(loss.detach())));print('TRAIN',arm,curve[-1],flush=True)
                if time.perf_counter()-start>1800:raise RuntimeError('Frozen 30 minute budget exceeded')
                if torch.cuda.max_memory_allocated()>6*(1<<30):raise RuntimeError('Frozen GPU allocation cap exceeded')
            if reference_batch is None:reference_batch=batch_trace
            else:assert batch_trace==reference_batch
            model.eval()
            with torch.inference_mode():
                inputs=[tx[k][:16] for k in ['own','rivals','mask']];a=model(*inputs,arm);perm=torch.tensor([5,3,1,4,2,0],device=device);b=model(inputs[0],inputs[1][:,:,perm],inputs[2][:,:,perm],arm);assert torch.allclose(a,b,atol=1e-5),arm
            dest=run.path/arm;dest.mkdir();checkpoint=dest/'model.pt';torch.save(dict(state={k:v.cpu() for k,v in model.state_dict().items()},arm=arm,steps=steps,seed=20261019,architecture='EvidenceDecision-v1',train_sequences=sorted({r['sequence'] for r in rows})),checkpoint)
            write_json(dest/'loss_curve.json',curve);np.save(dest/'initial_parameters.npy',init);write_json(dest/'batch_indices.json',batch_trace)
            paths[arm]=dict(path=str(checkpoint),sha256=sha256(checkpoint),mode=arm)
            summary.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),samples=len(rows),events=len(groups),steps=steps,seconds=time.perf_counter()-start,final_loss=curve[-1]['loss'],gradient_checks=gradcheck,set_permutation_invariance=True))
            del model,optimizer;torch.cuda.empty_cache()
        write_json(run.path/'SUMMARY.json',dict(arms=summary,matched_initialization=True,matched_batch_indices=True,positive_weight=pos_weight));write_json(root/('decision_smoke_pointer.json' if smoke else 'decision_model_pointer.json'),dict(run_id=run.id,path=str(run.path),models=paths));print('TRAIN_COMPLETE',summary,flush=True)


def load_model(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);m=EvidenceDecision();m.load_state_dict(ck['state']);m.eval();return m

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();train(a.root,a.smoke)
