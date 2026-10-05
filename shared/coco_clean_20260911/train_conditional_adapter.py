"""Matched 3-seed supervised training of plain/diagonal/conditional mask residuals."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from conditional_coefficient_adapter import transform,predict
from frozen_mechanism_probe import sha,write_csv,write_json

MODES=['plain','diagonal','conditional'];EPOCHS=15;LR=.003;BATCH=128

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    cache=a.cache.resolve();out=a.out.resolve();out.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    chunks=[];ids=[]
    for path in sorted((cache/'train').glob('*.npz')):
        q=np.load(path);idx=q['sample_prediction']
        if len(idx):chunks.append((q['features'][idx],q['sample_proto'],q['sample_gt'],q['level'][idx]));ids.extend((int(path.stem),int(t)) for t in q['sample_annotation'])
    X,P,Y,L=[torch.tensor(np.concatenate([c[k] for c in chunks]),device='cuda',dtype=torch.long if k==3 else torch.float32) for k in range(4)];del chunks
    init=np.load(cache/'initial_head.npz');initial={k:torch.tensor(init[k],device='cuda') for k in ['weight','bias']};N=len(X)
    transforms={}
    for mode in MODES:transforms[mode]=torch.cat([transform(P[k:k+256],mode) for k in range(0,N,256)])
    # Null residual is exactly the original predictor for every transform.
    zero={k:torch.zeros_like(v) for k,v in initial.items()}
    ref=predict(X[:16],L[:16],initial,zero,transforms['plain'][:16])
    for mode in MODES:assert torch.equal(ref,predict(X[:16],L[:16],initial,zero,transforms[mode][:16]))
    # Analytic chain-rule gradient, checked against autograd in the actual dynamic adapter.
    wr=torch.zeros_like(initial['weight'],requires_grad=True);br=torch.zeros_like(initial['bias'],requires_grad=True)
    t=transforms['conditional'][:8];x=X[:8];lev=L[:8];p=P[:8];y=Y[:8];c=predict(x,lev,initial,dict(weight=wr,bias=br),t);z=torch.einsum('nsk,nk->ns',p,c)
    aw,ab=torch.autograd.grad(F.binary_cross_entropy_with_logits(z,y),[wr,br]);g=torch.einsum('nsk,ns->nk',p,z.sigmoid()-y)/p.shape[1];gr=torch.einsum('nki,nk->ni',t,g)
    ew=torch.zeros_like(wr);eb=torch.zeros_like(br);ew.index_add_(0,lev,gr[:,:,None]*x[:,None,:]/8);eb.index_add_(0,lev,gr/8)
    torch.testing.assert_close(aw,ew,atol=1e-6,rtol=1e-5);torch.testing.assert_close(ab,eb,atol=1e-6,rtol=1e-5)
    write_json(out/'protocol.json',dict(type='conditional coefficient residual learning pilot',training=True,modes=MODES,seeds=[0,1,2],epochs=EPOCHS,batch=BATCH,lr=LR,
        cache=str(cache),cache_protocol_sha256=sha(cache/'protocol.json'),script_sha256=sha(__file__),adapter_sha256=sha(Path(__file__).with_name('conditional_coefficient_adapter.py')),
        train_targets=N,train_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),GT_used_in_inference=False,
        objective='identical ordinary BCE; dynamic inverse-root prototype Gram transform in forward residual, 0.01 ridge, matched average output energy',
        validation='15th epoch only, no validation selection; current 300-image development-validation subset, independent final confirmation required',
        verification=dict(zero_initialization='exact parity',analytic_gradient='matches autograd'),limits='Frozen features, boxes, scores, prototypes and matched training assignments. Head-local supervised pilot.'))
    start=time.monotonic()
    for seed in [0,1,2]:
        for mode in MODES:
            run=out/f'{mode}_s{seed}';run.mkdir();(run/'checkpoints').mkdir();T=transforms[mode]
            W=torch.zeros_like(initial['weight']);b=torch.zeros_like(initial['bias']);VW=torch.zeros_like(W);VB=torch.zeros_like(b);rng=np.random.default_rng(seed);history=[]
            for epoch in range(EPOCHS):
                perm=rng.permutation(N);lr=LR*(.1+.9*(1+math.cos(math.pi*epoch/EPOCHS))/2);loss_sum=0
                for pos in range(0,N,BATCH):
                    idx=torch.tensor(perm[pos:pos+BATCH],device='cuda');bs=len(idx);x=X[idx];p=P[idx];y=Y[idx];lev=L[idx];t=T[idx]
                    c=predict(x,lev,initial,dict(weight=W,bias=b),t);z=torch.einsum('nsk,nk->ns',p,c);loss=F.binary_cross_entropy_with_logits(z,y)
                    assert torch.isfinite(loss);g=torch.einsum('nsk,ns->nk',p,z.sigmoid()-y)/p.shape[1];g=torch.einsum('nki,nk->ni',t,g)
                    gw=torch.zeros_like(W);gb=torch.zeros_like(b);gw.index_add_(0,lev,g[:,:,None]*x[:,None,:]/bs);gb.index_add_(0,lev,g/bs)
                    norm=torch.sqrt(gw.double().square().sum()+gb.double().square().sum());assert torch.isfinite(norm);clip=min(1,10/max(float(norm),1e-10));gw*=clip;gb*=clip
                    VW.mul_(.9).add_(gw);VB.mul_(.9).add_(gb);W.add_(VW,alpha=-lr);b.add_(VB,alpha=-lr)
                    assert torch.isfinite(W).all() and torch.isfinite(b).all();loss_sum+=float(loss)*bs
                record=dict(seed=seed,mode=mode,epoch=epoch+1,train_bce=loss_sum/N,updates=(epoch+1)*math.ceil(N/BATCH),lr=lr);history.append(record)
                np.savez(run/'checkpoints'/f'epoch{epoch+1:02d}.npz',weight=W.cpu().numpy(),bias=b.cpu().numpy());write_csv(run/'history.csv',history)
                if (epoch+1)%5==0:print(json.dumps(record),flush=True)
            write_json(run/'COMPLETE.json',dict(status='COMPLETE',checkpoint='checkpoints/epoch15.npz',sha256=sha(run/'checkpoints/epoch15.npz')))
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',runs=9,epochs=EPOCHS,seconds=time.monotonic()-start))

if __name__=='__main__':main()
