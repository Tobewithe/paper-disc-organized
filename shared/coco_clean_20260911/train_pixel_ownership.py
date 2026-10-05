"""3-seed paired own-only versus neighbor-aware pixel residual learning."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pixel_ownership_refiner import PixelRefiner
from frozen_mechanism_probe import sha,write_json,write_csv

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;q=np.load(a.data/'pixels.npz')
    X=torch.tensor(q['features'],device='cuda');Y=torch.tensor(q['labels'],device='cuda');Z=torch.tensor(q['base_logits'],device='cuda');N=len(Y);epochs=10;batch=8192
    write_json(out/'protocol.json',dict(training=True,seeds=[0,1,2],modes=['own','neighbor'],epochs=epochs,batch=batch,optimizer='Adam',lr=.001,script_sha256=sha(__file__),feature_script_sha256=sha(Path(__file__).with_name('pixel_ownership_refiner.py')),data_sha256=sha(a.data/'pixels.npz'),
        initial='zero residual; same seed network initialization for paired variants',inference_GT=False,objective='ordinary per-pixel BCE, equal sampled pixels per training instance',final_epoch=10,
        limits='Pixel residual pilot, frozen detections and prototypes, development validation only. Neighbor arm adds no parameters; zeros six neighbor features in own-only control.'))
    start=time.monotonic()
    for seed in [0,1,2]:
        for mode in ['own','neighbor']:
            torch.manual_seed(seed);model=PixelRefiner().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.default_rng(seed)
            run=out/f'{mode}_s{seed}';run.mkdir();(run/'checkpoints').mkdir();history=[]
            assert torch.equal(model(X[:8]),torch.zeros(8,device='cuda'))
            for epoch in range(epochs):
                permutation=rng.permutation(N);loss_sum=0
                for pos in range(0,N,batch):
                    idx=torch.tensor(permutation[pos:pos+batch],device='cuda');x=X[idx].clone()
                    if mode=='own':x[:,8:]=0
                    loss=F.binary_cross_entropy_with_logits(Z[idx]+model(x),Y[idx]);assert torch.isfinite(loss)
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);optimizer.step();loss_sum+=float(loss)*len(idx)
                record=dict(seed=seed,mode=mode,epoch=epoch+1,bce=loss_sum/N);history.append(record);write_csv(run/'history.csv',history)
                torch.save(model.state_dict(),run/'checkpoints'/f'epoch{epoch+1:02d}.pt')
                if (epoch+1)%5==0:print(json.dumps(record),flush=True)
            write_json(run/'COMPLETE.json',dict(status='COMPLETE',sha256=sha(run/'checkpoints/epoch10.pt')))
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',runs=6,epochs=10,seconds=time.monotonic()-start))

if __name__=='__main__':main()
