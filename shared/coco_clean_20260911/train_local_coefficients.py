"""Same-capacity global versus spatially varying coefficient residual, train only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from local_coefficient_head import LocalCoefficientHead,gates
from frozen_mechanism_probe import sha,write_json,write_csv

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--pixels',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parts=[];identities=[]
    for path in sorted((a.cache/'train').glob('*.npz')):
        q=np.load(path);idx=q['sample_prediction']
        if len(idx):parts.append((q['sample_proto'],q['sample_gt'],q['coeff'][idx]));identities.extend((int(path.stem),int(aid)) for aid in q['sample_annotation'])
    q=np.load(a.features/'features.npz');r=np.load(a.pixels/'pixels.npz');assert np.array_equal(np.array(identities),q['identities']) and np.array_equal(q['identities'],r['identities'])
    X=torch.tensor(q['features'][:,:73],device='cuda');P,Y,C=[torch.tensor(np.concatenate([q[k] for q in parts]),device='cuda',dtype=torch.float32) for k in range(3)];N=len(X)
    uv=torch.tensor(r['features'][:,2:4].reshape(N,256,2),device='cuda')/2+.5;G=gates(uv);torch.testing.assert_close(G.sum(-1),torch.ones_like(Y));assert torch.all(G>=0)
    mean=X.mean(0);std=X.std(0).clamp_min(.01);X=(X-mean)/std;torch.save(dict(mean=mean.cpu(),std=std.cpu()),out/'normalizer.pt')
    write_json(out/'protocol.json',dict(script_sha256=sha(__file__),feature_source_sha256=sha(a.features/'features.npz'),pixel_source_sha256=sha(a.pixels/'pixels.npz'),seeds=[0,1,2],modes=['global','local'],epochs=15,batch=128,optimizer='Adam',lr=.0001,GT_inference=False,
        learning='Zero-initialized MLP 73->128->128->128 outputs 4x32 coefficients. Local bilinear interpolation over normalized predicted-box xy; global averages same 4 coefficient vectors. Equal architecture and optimizer.',
        objective='Ordinary BCE at same 256 original GT sample pixels; frozen detector/prototypes. Position in training reconstructed from verified pixel feature cache.',
        schedule='Same paired initialization/permutation; fixed epoch15, all epoch checkpoints.',limitations='Expressivity pilot, bilinear spatial coefficients have existing related work and are not yet a novelty claim. Frozen candidates and reused development validation.'))
    start=time.monotonic()
    for seed in [0,1,2]:
        for mode in ['global','local']:
            torch.manual_seed(seed);model=LocalCoefficientHead().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.0001);rng=np.random.default_rng(seed);run=out/f'{mode}_s{seed}';run.mkdir();(run/'checkpoints').mkdir();history=[]
            assert torch.equal(model(X[:8]),torch.zeros((8,4,32),device='cuda'))
            for epoch in range(15):
                order=rng.permutation(N);loss_sum=0.
                for pos in range(0,N,128):
                    idx=torch.tensor(order[pos:pos+128],device='cuda');dc=model(X[idx]);base=torch.einsum('nsk,nk->ns',P[idx],C[idx])
                    residual=torch.einsum('nsk,nqk,nsq->ns',P[idx],dc,G[idx]) if mode=='local' else torch.einsum('nsk,nk->ns',P[idx],dc.mean(1))
                    loss=F.binary_cross_entropy_with_logits(base+residual,Y[idx]);assert torch.isfinite(loss)
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);optimizer.step();loss_sum+=float(loss.detach())*len(idx)
                r=dict(seed=seed,mode=mode,epoch=epoch+1,bce=loss_sum/N);history.append(r);write_csv(run/'history.csv',history);torch.save(model.state_dict(),run/'checkpoints'/f'epoch{epoch+1:02d}.pt')
                if epoch==14:print(json.dumps(r),flush=True)
            write_json(run/'COMPLETE.json',dict(status='COMPLETE',sha256=sha(run/'checkpoints/epoch15.pt')))
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',runs=6,seconds=time.monotonic()-start))

if __name__=='__main__':main()
