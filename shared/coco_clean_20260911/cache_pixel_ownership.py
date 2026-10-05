"""Build train2017 pixel-refiner data from frozen official prediction cache."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,io,json,time
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics.utils import ops
from frozen_mechanism_probe import ROOT,sha,write_json
from pixel_ownership_refiner import build_features

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_train2017.json'))
    features=[];labels=[];base=[];identities=[];start=time.monotonic()
    paths=sorted((a.cache/'train').glob('*.npz'))
    for n,path in enumerate(paths,1):
        iid=int(path.stem);q=np.load(path);shape=tuple(map(int,q['shape']));ishape=tuple(map(int,q['input_shape']));boxes=torch.tensor(q['boxes'],device='cuda');c=torch.tensor(q['coeff'],device='cuda');det=torch.tensor(q['detections'],device='cuda')
        anns=gt.imgToAnns[iid];crowd=[torch.tensor(gt.annToMask(g).astype(bool),device='cuda') for g in anns if g.get('iscrowd',0)];valid=~torch.stack(crowd).any(0) if crowd else torch.ones(shape,dtype=torch.bool,device='cuda')
        gain=min(ishape[0]/shape[0],ishape[1]/shape[1]);pad_x=(ishape[1]-shape[1]*gain)/2;pad_y=(ishape[0]-shape[0]*gain)/2
        for k,(j,aid) in enumerate(zip(q['sample_prediction'],q['sample_annotation'])):
            j=int(j);aid=int(aid);support=ops.crop_mask(torch.ones((1,*ishape),device='cuda',dtype=torch.uint8),boxes[j:j+1]);support=(ops.scale_masks(support[:,None],shape)[0,0]>.5)&valid
            coords=support.flatten().nonzero().flatten();rng=np.random.default_rng(20260911+iid+aid);pick=coords[torch.tensor(rng.integers(len(coords),size=256),device='cuda')]
            # Same pixel sample as the earlier cache; verify official GT bit identity.
            own=torch.tensor(gt.annToMask(gt.anns[aid]).astype(bool),device='cuda').flatten()[pick]
            assert np.array_equal(own.cpu().numpy(),q['sample_gt'][k])
            pts=torch.stack([(pick%shape[1]+.5)*gain+pad_x,(pick//shape[1]+.5)*gain+pad_y],1)
            p=torch.tensor(q['sample_proto'][k],device='cuda');z=c@p.T
            f=build_features(z,boxes,det,j,pts,'neighbor')
            features.append(f.cpu().numpy());labels.append(q['sample_gt'][k]);base.append(z[j].cpu().numpy());identities.append((iid,aid))
        if n%200==0:print(json.dumps(dict(completed=n,total=len(paths),seconds=time.monotonic()-start)),flush=True)
    X=np.concatenate(features).astype(np.float32);Y=np.concatenate(labels).astype(np.float32);Z=np.concatenate(base).astype(np.float32)
    np.savez(out/'pixels.npz',features=X,labels=Y,base_logits=Z,identities=np.array(identities))
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',training=False,images=len(paths),targets=len(identities),pixels=len(Y),script_sha256=sha(__file__),source_cache=str(a.cache),source_protocol_sha256=sha(a.cache/'protocol.json'),data_sha256=sha(out/'pixels.npz'),
        GT_usage='Supervised labels and crowd-excluded sampling only; feature construction uses frozen predictions, boxes, coefficients and pixel coordinates',seconds=time.monotonic()-start))

if __name__=='__main__':main()
