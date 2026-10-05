"""Cache official last coefficient-layer inputs and original-COCO pixel targets.

Training is standard instance-supervised BCE on matched detections in train2017.
No GT is used to construct validation predictions; GT ownership is evaluation only.
All backbone, prototype, box, score and assignment outputs remain frozen.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,csv,hashlib,io,json,time,types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.utils import ops
from frozen_mechanism_probe import ROOT,Capture,ownership,sha,write_json,write_csv

SEED=20260911
def rank(x):return hashlib.sha256(f'pilot:{SEED}:{x}'.encode()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--train-images',type=int,default=1200);ap.add_argument('--val-selection',type=Path,required=True);a=ap.parse_args()
    out=a.out.resolve();out.mkdir(exist_ok=False,parents=True);(out/'train').mkdir();(out/'val').mkdir()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with contextlib.redirect_stdout(io.StringIO()):train=COCO(str(ROOT/'data/annotations/instances_train2017.json'));val=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    with (ROOT/'census/train2017_images.csv').open(encoding='utf-8-sig') as f:census=list(csv.DictReader(f))
    pools={'high':[],'low':[]}
    for r in census:
        if int(r['n_instances'])==0:continue
        pools['high' if int(r['n_high_same']) else 'low'].append(int(r['image_id']))
    chosen=[]
    for group,n in [('high',2*a.train_images//3),('low',a.train_images-2*a.train_images//3)]:chosen.extend(sorted(pools[group],key=rank)[:n])
    chosen.sort(key=rank);vselection=json.loads(a.val_selection.read_text());vids=[r['image_id'] for r in vselection];assert not set(chosen)&set(vids)
    write_json(out/'selection.json',dict(train=chosen,val=vids,val_source_sha256=sha(a.val_selection)))
    write_json(out/'protocol.json',dict(script_sha256=sha(__file__),official_weight_sha256=sha(ROOT/'weights/yolo26m-seg.pt'),train_images=a.train_images,val_images=len(vids),
       train_sampling='2/3 train2017 images containing high-ICI instances; 1/3 no high-ICI; hash chosen before predictions',
       matched_training='Official bbox IoU>=0.5 one-to-one matching, all noncrowd instances; fixed selected prediction coeff-layer features',
       sampling_pixels=256,training_mask='Original pycocotools annToMask; one annotation ID per instance. Uniform with-replacement sampling from fixed predicted-box support excluding crowd.',
       limitation='Frozen detection/prototype head-local learning pilot; not end-to-end supervised YOLO reproduction. Validation subset was used for mechanism diagnostics.'))
    model=YOLO(str(ROOT/'weights/yolo26m-seg.pt'));model.model.eval().requires_grad_(False);head=model.model.model[-1]
    branches=head.one2one_cv4 if head.end2end else head.cv4
    features={};outputcoeff={};hooks=[]
    for level,branch in enumerate(branches):
        def hook(mod,inputs,output,level=level):
            features[level]=inputs[0].detach().clone();outputcoeff[level]=output.detach().clone()
        hooks.append(branch[-1].register_forward_hook(hook))
    weight=torch.stack([branch[-1].weight.detach().flatten(1) for branch in branches]);bias=torch.stack([branch[-1].bias.detach() for branch in branches])
    np.savez(out/'initial_head.npz',weight=weight.cpu().numpy(),bias=bias.cpu().numpy())
    records=[];start=time.monotonic();maxreplay=0.
    for split,gt,image_ids in [('train',train,chosen),('val',val,vids)]:
        for n,iid in enumerate(image_ids,1):
            path=ROOT/'data/images'/f'{split}2017'/gt.imgs[iid]['file_name']
            features.clear();outputcoeff.clear()
            with torch.no_grad():
                model.predict(str(path),predictor=Capture,imgsz=640,conf=.001,max_det=300,iou=.7,device=0,rect=False,half=False,retina_masks=False,verbose=False)
                cap=model.predictor.capture;coeff=cap['coeff']
                flatfeat=torch.cat([features[k][0].flatten(1).T for k in range(3)]);flatc=torch.cat([outputcoeff[k][0].flatten(1).T for k in range(3)])
                levels=torch.cat([torch.full((outputcoeff[k][0].numel()//32,),k,dtype=torch.long,device='cuda') for k in range(3)])
                original=flatc
                if len(coeff):
                    # Match NMS-kept coefficients back to dense coefficient outputs.
                    # Verify exact coefficient equality and that a distinct feature vector
                    # is not hidden behind an equal-coefficient tie.
                    dist=torch.cdist(coeff.double(),original.double());indices=dist.argmin(1)
                    for j in range(len(coeff)):
                        ties=(dist[j]<1e-7).nonzero().flatten()
                        if len(ties)>1:assert torch.equal(flatfeat[ties],flatfeat[ties[:1]].expand(len(ties),-1))
                    assert float((coeff-flatc[indices]).abs().max())<1e-5
                    x=flatfeat[indices];level=levels[indices];replay=torch.einsum('nki,ni->nk',weight.to(x.device)[level],x)+bias.to(x.device)[level]
                    error=float((replay-coeff).abs().max());maxreplay=max(maxreplay,error);assert error<1e-5,error
                else:x=torch.empty((0,64),device='cuda');level=torch.empty(0,dtype=torch.long,device='cuda')
                mapping=ownership(gt,iid,cap['detections']);shape=cap['shape'];ishape=cap['input_shape'];p=cap['proto']
                arrays=dict(features=x.cpu().numpy(),level=level.cpu().numpy(),coeff=coeff.cpu().numpy(),boxes=cap['boxes'].cpu().numpy(),detections=cap['detections'].cpu().numpy(),
                    proto=p.cpu().numpy(),shape=shape,input_shape=ishape,mapping_gt=np.array(list(mapping),dtype=np.int64),mapping_pred=np.array(list(mapping.values()),dtype=np.int64))
                if split=='train':
                    anns=gt.imgToAnns[iid];raster={q['id']:torch.tensor(gt.annToMask(q).astype(bool),device='cuda') for q in anns};crowd=[raster[q['id']] for q in anns if q.get('iscrowd',0)]
                    valid=~torch.stack(crowd).any(0) if crowd else torch.ones(shape,dtype=torch.bool,device='cuda')
                    pf=ops.scale_masks(F.interpolate(p[None],ishape,mode='bilinear',align_corners=False)[0][:,None],shape)[:,0]
                    ids=list(mapping);predidx=list(mapping.values());Xs=[];ys=[];kept=[];targetids=[]
                    for aid,j in zip(ids,predidx):
                        box=cap['boxes'][j:j+1]
                        support=ops.crop_mask(torch.ones((1,*ishape),dtype=torch.uint8,device='cuda'),box)
                        support=(ops.scale_masks(support[:,None],shape)[0,0]>.5)&valid
                        coords=support.flatten().nonzero().flatten()
                        if not len(coords):continue
                        rng=np.random.default_rng(SEED+iid+aid);pick=coords[torch.tensor(rng.integers(len(coords),size=256),device='cuda')]
                        Xs.append(pf.flatten(1)[:,pick].T.cpu().numpy());ys.append(raster[aid].flatten()[pick].cpu().numpy());kept.append(j);targetids.append(aid)
                    # All matched instances retained; no mask-IoU success filtering.
                    arrays.update(sample_proto=np.stack(Xs) if Xs else np.empty((0,256,32),np.float32),sample_gt=np.stack(ys) if ys else np.empty((0,256),bool),
                                  sample_prediction=np.array(kept,dtype=np.int64),sample_annotation=np.array(targetids,dtype=np.int64))
                    arrays.pop('proto') # train uses sampled features only, saves storage
                np.savez_compressed(out/split/f'{iid}.npz',**arrays)
                records.append(dict(split=split,image_id=iid,detections=len(coeff),gt_noncrowd=sum(not q.get('iscrowd',0) for q in gt.imgToAnns[iid]),matched=len(mapping),head_replay_error=error if len(coeff) else 0))
            if n%50==0:
                progress=dict(split=split,completed=n,total=len(image_ids),seconds=round(time.monotonic()-start,1));write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    for h in hooks:h.remove()
    write_csv(out/'coverage.csv',records);write_json(out/'COMPLETE.json',dict(status='COMPLETE',training=False,train_images=len(chosen),val_images=len(vids),max_head_replay_error=maxreplay,seconds=time.monotonic()-start,script_sha256=sha(__file__)))

if __name__=='__main__':main()
