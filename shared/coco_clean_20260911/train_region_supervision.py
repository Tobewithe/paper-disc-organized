"""Paired region-supervision pilot: BCE vs neighbor vs budget-matched background.

Region labels come from original train2017 annotations. Validation GT is never read.
Same frozen 1,200 images, three seeds, 15 epochs, final checkpoint, last head only.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,io,json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
from frozen_mechanism_probe import ROOT,write_json,write_csv,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();cache=a.cache.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_train2017.json'))
    chunks=[];regions=[];ids=[];start=time.monotonic()
    for path in sorted((cache/'train').glob('*.npz')):
        iid=int(path.stem);q=np.load(path);shape=tuple(map(int,q['shape']));ishape=tuple(map(int,q['input_shape']));boxes=torch.tensor(q['boxes'],device='cuda');idx=q['sample_prediction']
        anns=gt.imgToAnns[iid];raster={g['id']:torch.tensor(gt.annToMask(g).astype(bool),device='cuda') for g in anns};ordinary=[g for g in anns if not g.get('iscrowd',0)]
        union=torch.stack([raster[g['id']] for g in ordinary]).any(0) if ordinary else torch.zeros(shape,dtype=torch.bool,device='cuda')
        same={cat:torch.stack([raster[g['id']] for g in ordinary if g['category_id']==cat]).any(0) for cat in {g['category_id'] for g in ordinary}}
        crowd=[raster[g['id']] for g in anns if g.get('iscrowd',0)];valid=~torch.stack(crowd).any(0) if crowd else torch.ones(shape,dtype=torch.bool,device='cuda')
        for k,(j,aid) in enumerate(zip(idx,q['sample_annotation'])):
            j=int(j);aid=int(aid);support=ops.crop_mask(torch.ones((1,*ishape),device='cuda',dtype=torch.uint8),boxes[j:j+1]);support=(ops.scale_masks(support[:,None],shape)[0,0]>.5)&valid
            coords=support.flatten().nonzero().flatten();rng=np.random.default_rng(20260911+iid+aid);pick=coords[torch.tensor(rng.integers(len(coords),size=256),device='cuda')]
            own=raster[aid].flatten()[pick];assert np.array_equal(own.cpu().numpy(),q['sample_gt'][k])
            neighbor=same[gt.anns[aid]['category_id']].flatten()[pick]&~own;background=~union.flatten()[pick]
            assert not (neighbor&background).any() and not (neighbor&own).any()
            regions.append(torch.stack([neighbor,background]).cpu().numpy());ids.append((iid,aid))
        if len(idx):chunks.append((q['features'][idx],q['sample_proto'],q['sample_gt'],q['level'][idx]))
    X,P,Y,levels=[torch.tensor(np.concatenate([q[k] for q in chunks]),device='cuda',dtype=torch.long if k==3 else torch.float32) for k in range(4)]
    regions=torch.tensor(np.stack(regions),device='cuda');neighbor=regions[:,0].float();background=regions[:,1].float();N=len(X)
    # Each active target receives the same extra total negative weight in both arms.
    # Require sampled background in both so the background control is exactly matched.
    active=(neighbor.sum(1)>0)&(background.sum(1)>0);extra=neighbor.sum(1)*active
    wn=1+neighbor*active[:,None];wb=1+background*(extra/background.sum(1).clamp_min(1))[:,None]
    torch.testing.assert_close(wn.sum(1),wb.sum(1),atol=1e-4,rtol=1e-6)
    weights={'bce':torch.ones_like(Y),'neighbor':wn/wn.mean(1,keepdim=True),'background':wb/wb.mean(1,keepdim=True)}
    for w in weights.values():assert torch.isfinite(w).all();torch.testing.assert_close(w.mean(1),torch.ones(N,device='cuda'))
    init=np.load(cache/'initial_head.npz');initial=[torch.tensor(init[k],device='cuda') for k in ['weight','bias']]
    write_json(out/'protocol.json',dict(script_sha256=sha(__file__),cache_protocol_sha256=sha(cache/'protocol.json'),original_train_gt_sha256=sha(ROOT/'data/annotations/instances_train2017.json'),
        seeds=[0,1,2],modes=list(weights),epochs=15,batch=128,lr=.003,momentum=.9,train_targets=N,active_targets=int(active.sum()),
        weighting='Neighbor: negative same-class-other-GT pixels weight 2. Background: exactly matched per-target total extra weight placed on background negatives. Normalize all weights to mean 1 per target. Both arms active only where sampled neighbor and background exist; otherwise ordinary BCE.',
        training='Frozen official features/prototypes/detections; update last coefficient conv weights/bias only. Original GT supervision. Same sample order and final epoch15 for each arm/seed.',GT_at_inference=False,
        limitations='Exploratory 1,200 train images/300 reused validation subset. Region weighting is a mechanism test and not claimed novel. Uniform 256-pixel samples may miss small neighbor regions.'))
    np.savez_compressed(out/'region_sampling.npz',ids=np.array(ids),neighbor=neighbor.cpu().numpy().astype(bool),background=background.cpu().numpy().astype(bool),active=active.cpu().numpy())
    for seed in [0,1,2]:
        for mode,allw in weights.items():
            W=initial[0].clone().requires_grad_(True);b=initial[1].clone().requires_grad_(True);optimizer=torch.optim.SGD([W,b],lr=.003,momentum=.9);rng=np.random.default_rng(seed)
            run=out/f'{mode}_s{seed}';run.mkdir();(run/'checkpoints').mkdir();history=[]
            for epoch in range(15):
                order=rng.permutation(N);lr=.003*(.1+.9*(1+math.cos(math.pi*epoch/15))/2);optimizer.param_groups[0]['lr']=lr;loss_sum=0.;bce_sum=0.
                for pos in range(0,N,128):
                    idx=torch.tensor(order[pos:pos+128],device='cuda');c=torch.einsum('nki,ni->nk',W[levels[idx]],X[idx])+b[levels[idx]];z=torch.einsum('nsk,nk->ns',P[idx],c)
                    losses=F.binary_cross_entropy_with_logits(z,Y[idx],reduction='none');loss=(losses*allw[idx]).mean();assert torch.isfinite(loss)
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([W,b],10,error_if_nonfinite=True);optimizer.step();loss_sum+=float(loss.detach())*len(idx);bce_sum+=float(losses.detach().mean())*len(idx)
                r=dict(seed=seed,mode=mode,epoch=epoch+1,weighted_bce=loss_sum/N,ordinary_bce=bce_sum/N);history.append(r);write_csv(run/'history.csv',history)
                np.savez(run/'checkpoints'/f'epoch{epoch+1:02d}.npz',weight=W.detach().cpu().numpy(),bias=b.detach().cpu().numpy())
                if epoch==14:print(json.dumps(r),flush=True)
            write_json(run/'COMPLETE.json',dict(status='COMPLETE',sha256=sha(run/'checkpoints/epoch15.npz')))
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',runs=9,seconds=time.monotonic()-start,script_sha256=sha(__file__)))

if __name__=='__main__':main()
