"""Paired 3-seed supervised learning of the official last coefficient layer only.

Compare ordinary BCE, diagonal and full Gram preconditioning of coefficient
gradients. Input activations, prototypes, predictions and GT assignments are frozen.
Training uses train2017 only. Evaluation GT is never read by fit().
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,csv,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from frozen_mechanism_probe import write_json,write_csv,sha

SEEDS=[0,1,2];MODES=['bce','diagonal','gram'];EPOCHS=15;BATCH=128;LR=.003

def fit(cache,out,seed,mode,data,initial):
    run=out/f'{mode}_s{seed}';run.mkdir(exist_ok=False);(run/'checkpoints').mkdir()
    X,P,Y,levels=data;N=len(X);W=initial[0].clone();bias=initial[1].clone();VW=torch.zeros_like(W);VB=torch.zeros_like(bias)
    rng=np.random.default_rng(seed);records=[];start=time.monotonic();updates=0
    for epoch in range(EPOCHS):
        order=rng.permutation(N);lr=LR*(.1+.9*(1+math.cos(math.pi*epoch/EPOCHS))/2)
        loss_sum=0.;count=0
        for pos in range(0,N,BATCH):
            idx=torch.tensor(order[pos:pos+BATCH],device='cuda');x=X[idx];p=P[idx];y=Y[idx];lev=levels[idx];bs=len(idx)
            c=torch.einsum('nki,ni->nk',W[lev],x)+bias[lev];logits=torch.einsum('nsk,nk->ns',p,c)
            loss=F.binary_cross_entropy_with_logits(logits,y);assert torch.isfinite(loss)
            gradient=torch.einsum('nsk,ns->nk',p,logits.sigmoid()-y)/p.shape[1]
            if mode!='bce':
                G=torch.einsum('nsk,nsl->nkl',p,p)/p.shape[1];ridge=.01*G.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-10)
                if mode=='diagonal':d=gradient/(G.diagonal(dim1=-2,dim2=-1)+ridge[:,None])
                else:
                    M=G+ridge[:,None,None]*torch.eye(32,device='cuda');d=torch.linalg.solve(M,gradient[:,:,None])[:,:,0]
                # Same sample logit-change RMS as the plain BCE coefficient gradient.
                rms_g=torch.einsum('nsk,nk->ns',p,gradient).square().mean(1).sqrt()
                rms_d=torch.einsum('nsk,nk->ns',p,d).square().mean(1).sqrt()
                gradient=d*(rms_g/rms_d.clamp_min(1e-12))[:,None]
            gw=torch.zeros_like(W);gb=torch.zeros_like(bias)
            gw.index_add_(0,lev,gradient[:,:,None]*x[:,None,:]/bs);gb.index_add_(0,lev,gradient/bs)
            norm=torch.sqrt(gw.double().square().sum()+gb.double().square().sum());assert torch.isfinite(norm)
            factor=min(1.,10./max(float(norm),1e-10));gw*=factor;gb*=factor
            VW.mul_(.9).add_(gw);VB.mul_(.9).add_(gb);W.add_(VW,alpha=-lr);bias.add_(VB,alpha=-lr)
            assert torch.isfinite(W).all() and torch.isfinite(bias).all()
            loss_sum+=float(loss)*bs;count+=bs;updates+=1
        # Training-only exact objective checkpoint; never select an epoch by val GT.
        record=dict(seed=seed,mode=mode,epoch=epoch+1,lr=lr,train_bce=loss_sum/count,updates=updates,seconds=time.monotonic()-start)
        records.append(record);np.savez(run/'checkpoints'/f'epoch{epoch+1:02d}.npz',weight=W.cpu().numpy(),bias=bias.cpu().numpy())
        write_csv(run/'history.csv',records)
        if (epoch+1)%5==0:print(json.dumps(record),flush=True)
    write_json(run/'COMPLETE.json',dict(status='COMPLETE',seed=seed,mode=mode,epochs=EPOCHS,updates=updates,final_checkpoint='checkpoints/epoch15.npz',sha256=sha(run/'checkpoints/epoch15.npz')))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    cache=a.cache.resolve();out=a.out.resolve();assert (cache/'COMPLETE.json').exists();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    chunks=[];targetids=[]
    for path in sorted((cache/'train').glob('*.npz')):
        item=np.load(path);idx=item['sample_prediction']
        if len(idx):chunks.append((item['features'][idx],item['sample_proto'],item['sample_gt'],item['level'][idx]));targetids.extend([(int(path.stem),int(x)) for x in item['sample_annotation']])
    assert len(set(targetids))==len(targetids)
    data=[torch.tensor(np.concatenate([x[k] for x in chunks]),device='cuda',dtype=torch.long if k==3 else torch.float32) for k in range(4)]
    del chunks
    init=np.load(cache/'initial_head.npz');initial=[torch.tensor(init[k],device='cuda') for k in ['weight','bias']]
    # Analytic chain-rule witness for ordinary BCE, including averaging.
    X,P,Y,levels=data;w=initial[0].detach().clone().requires_grad_(True);b=initial[1].detach().clone().requires_grad_(True);n=8
    x=X[:n];p=P[:n];y=Y[:n];lev=levels[:n];c=torch.einsum('nki,ni->nk',w[lev],x)+b[lev];z=torch.einsum('nsk,nk->ns',p,c)
    aw,ab=torch.autograd.grad(F.binary_cross_entropy_with_logits(z,y),[w,b]);g=torch.einsum('nsk,ns->nk',p,z.sigmoid()-y)/p.shape[1];ew=torch.zeros_like(w);eb=torch.zeros_like(b);ew.index_add_(0,lev,g[:,:,None]*x[:,None,:]/n);eb.index_add_(0,lev,g/n)
    torch.testing.assert_close(aw,ew,atol=1e-6,rtol=1e-5);torch.testing.assert_close(ab,eb,atol=1e-6,rtol=1e-5)
    write_json(out/'protocol.json',dict(training=True,type='supervised frozen-coefficient-head pilot',seeds=SEEDS,modes=MODES,epochs=EPOCHS,batch=BATCH,lr=LR,momentum=.9,
       script_sha256=sha(__file__),cache=str(cache),cache_protocol_sha256=sha(cache/'protocol.json'),initial_head_sha256=sha(cache/'initial_head.npz'),train_targets=len(targetids),
       train_target_ids_sha256=hashlib.sha256(json.dumps(targetids).encode()).hexdigest(),matching='fixed official bbox ownership',sampling='256 uniform pixels per matched train GT within predicted box',
       geometry='ridge 0.01 trace(G)/32; normalize transformed gradient to the plain BCE per-sample output-logit RMS',
       schedule='same 15 epochs and permutations for all modes of each seed; all epoch checkpoints; evaluate final epoch only',
       validation='GT not read by fit; no val selection of weights or hyperparameters',analytic_bce_gradient_check='PASS',limitations='Frozen features/detections/prototypes, last-layer-only pilot. Not end-to-end YOLO training, and validation subset previously used for mechanism analysis.'))
    start=time.monotonic()
    for seed in SEEDS:
        for mode in MODES:fit(cache,out,seed,mode,data,initial)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',runs=9,seeds=SEEDS,epochs=EPOCHS,seconds=time.monotonic()-start,script_sha256=sha(__file__)))

if __name__=='__main__':main()
