"""Raw COCO uniform image selection and input-grid supervision for matched pairs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,contextlib,hashlib,io,json,shutil,time,zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import ops
from pycocotools.coco import COCO
from frozen_mechanism_probe import ROOT,Capture,ownership,sha,write_json
from three_region_probe import write_csv
from crossimage_response_experiment import gt_input_regions
from ownership_ranking import input_features

def rank(value):return hashlib.sha256(f'ownership-ranking:20260912:{value}'.encode()).hexdigest()

def sample_proto(p,pos,shape):
    if tuple(p.shape[-2:])!=tuple(shape):raise RuntimeError('Prototype must already be on exact input grid')
    return p.flatten(1)[:,pos].T

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False);(out/'train').mkdir()
    configpath=Path(__file__).with_name('ownership_ranking_protocol_20260912.json');config=json.loads(configpath.read_text())
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_train2017.json'))
    chosen=sorted(sorted(gt.imgs,key=lambda iid:rank(f'train:{iid}'))[:12 if a.smoke else config['train_images']])
    valall={int(p.stem) for p in (ROOT/'diagnostics/full_val_cache_20260911/val').glob('*.npz')};assert len(valall)==5000
    vids=sorted(sorted(valall,key=lambda iid:rank(f'val:{iid}'))[:8 if a.smoke else config['eval_images']]);assert not set(chosen)&set(vids)
    write_json(out/'selection.json',dict(train=chosen,evaluation=vids,full_train_pool=len(gt.imgs),selection='all raw image IDs, no density filtering',smoke=a.smoke))
    dependencies=['ownership_ranking.py','local_coefficient_head.py','frozen_mechanism_probe.py','crossimage_response_experiment.py','ownership_ranking_protocol_20260912.json']
    write_json(out/'protocol.json',dict(config=config,script_sha256=sha(__file__),dependencies={n:sha(Path(__file__).with_name(n)) for n in dependencies},
        weight_sha256=sha(ROOT/'weights/yolo26m-seg.pt'),annotations_sha256={s:sha(ROOT/f'data/annotations/instances_{s}2017.json') for s in ['train','val']},selection_sha256=sha(out/'selection.json')))
    # Extract exactly the prespecified IDs; never substitute available dense images.
    needed={f'train2017/{gt.imgs[i]["file_name"]}' for i in chosen if not (ROOT/'data/images/train2017'/gt.imgs[i]['file_name']).exists()};extracted=0
    if needed:
        with zipfile.ZipFile('/autodl-pub/data/COCO2017/train2017.zip') as z:
            for entry in z.infolist():
                if entry.filename in needed:
                    dest=ROOT/'data/images'/entry.filename;dest.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(entry) as source,dest.open('wb') as target:shutil.copyfileobj(source,target)
                    if dest.stat().st_size!=entry.file_size:raise RuntimeError('JPEG extraction size mismatch')
                    extracted+=1
        assert extracted==len(needed)
    model=YOLO(str(ROOT/'weights/yolo26m-seg.pt'));model.model.eval().requires_grad_(False);head=model.model.model[-1];feats={};outputs={};hooks=[]
    branches=head.one2one_cv4 if head.end2end else head.cv4
    for level,branch in enumerate(branches):
        def hook(mod,inputs,output,level=level):feats[level]=inputs[0].detach();outputs[level]=output.detach()
        hooks.append(branch[-1].register_forward_hook(hook))
    categories=sorted(gt.cats);rows=[];hashes={};imagehash={};started=time.monotonic();maxreplay=0.
    for n,iid in enumerate(chosen,1):
        image=ROOT/'data/images/train2017'/gt.imgs[iid]['file_name'];imagehash[str(iid)]=sha(image)
        with torch.inference_mode():
            model.predict(str(image),predictor=Capture,imgsz=640,conf=.001,max_det=300,iou=.7,device=0,rect=False,half=False,retina_masks=False,verbose=False)
            cap=model.predictor.capture;p=cap['proto'].float();c=cap['coeff'];boxes=cap['boxes'];shape=tuple(map(int,cap['input_shape']));det=cap['detections']
            input_proto=F.interpolate(p[None],shape,mode='bilinear',align_corners=False)[0]
            flatfeat=torch.cat([feats[k][0].flatten(1).T for k in range(3)]);flatc=torch.cat([outputs[k][0].flatten(1).T for k in range(3)])
            levels=torch.cat([torch.full((outputs[k][0].numel()//32,),k,device='cuda',dtype=torch.long) for k in range(3)])
            if len(c):
                idx=torch.cdist(c.double(),flatc.double()).argmin(1)
                if float((c-flatc[idx]).abs().max())>1e-6:raise RuntimeError('feature coefficient replay')
                # Equal coefficients must not hide different branch inputs.
                for j in range(len(c)):
                    tied=(flatc==c[j]).all(1).nonzero().flatten()
                    if len(tied)>1 and not torch.equal(flatfeat[tied],flatfeat[tied[:1]].expand(len(tied),-1)):raise RuntimeError('ambiguous source feature')
                item=dict(features=flatfeat[idx],level=levels[idx],boxes=boxes,input_shape=shape)
                x=input_features(item);raw=F.interpolate((c@p.flatten(1)).reshape(1,-1,*p.shape[-2:]),shape,mode='bilinear',align_corners=False)[0]
                initial=ops.crop_mask((raw>0).to(torch.uint8),boxes);official=ops.process_mask(p,c,boxes,shape,upsample=True)
                if int((initial!=official).sum()):raise RuntimeError('official mask parity')
                support=ops.crop_mask(torch.ones_like(raw,dtype=torch.uint8),boxes).bool()
            else:x=torch.empty((0,73),device='cuda');raw=support=torch.empty((0,*shape),device='cuda')
            mapping=ownership(gt,iid,det);raster,union,valid=gt_input_regions(gt,iid,cap)
            targetids=[];predidx=[];ps=[];ys=[];positives=[];haspositive=[];samplepositions=[];positivepositions=[];statuses=[]
            for ann in gt.imgToAnns[iid]:
                if ann.get('iscrowd',0):continue
                aid=ann['id'];j=mapping.get(aid)
                if j is None:statuses.append(dict(annotation_id=aid,status='unmatched'));continue
                mask=support[j]&valid;pool=mask.flatten().nonzero().flatten()
                if not len(pool):statuses.append(dict(annotation_id=aid,status='empty_support'));continue
                rng=np.random.default_rng(20260912+iid*31+aid*7);pick=pool[torch.tensor(rng.integers(len(pool),size=256),device='cuda')]
                pp=sample_proto(input_proto,pick,shape);replay=float((pp@c[j]-raw[j].flatten()[pick]).abs().max());maxreplay=max(maxreplay,replay)
                if replay>2e-4:raise RuntimeError(('input sample replay',replay))
                poolpos=(mask&raster[aid]).flatten().nonzero().flatten();pv=bool(len(poolpos));pos=poolpos[torch.tensor(rng.integers(len(poolpos),size=64),device='cuda')] if pv else torch.zeros(64,device='cuda',dtype=torch.long)
                targetids.append(aid);predidx.append(j);ps.append(pp.cpu().numpy());ys.append(raster[aid].flatten()[pick].cpu().numpy());positives.append(sample_proto(input_proto,pos,shape).cpu().numpy() if pv else np.zeros((64,32),np.float32));haspositive.append(pv)
                samplepositions.append(pick.cpu().numpy());positivepositions.append(pos.cpu().numpy());statuses.append(dict(annotation_id=aid,status='matched',prediction_index=j,positive_pool=len(poolpos)))
            pairps=[];src=[];dst=[];pairpos=[];edges=[]
            for ii,aid in enumerate(targetids):
                for jj in range(ii+1,len(targetids)):
                    bid=targetids[jj];i,j=predidx[ii],predidx[jj]
                    if gt.anns[aid]['category_id']!=gt.anns[bid]['category_id']:continue
                    if int(det[i,5])!=int(det[j,5]) or categories[int(det[i,5])]!=gt.anns[aid]['category_id']:raise RuntimeError('class/assignment mismatch')
                    wh=(torch.minimum(boxes[i,2:],boxes[j,2:])-torch.maximum(boxes[i,:2],boxes[j,:2])).clamp_min(0);inter=wh.prod();areas=(boxes[[i,j],2:]-boxes[[i,j],:2]).clamp_min(0).prod(1)
                    if float(inter/(areas.sum()-inter).clamp_min(1e-9))<=.05:continue
                    edges.append([ii,jj]);common=support[i]&support[j]&valid
                    for s,t,aa,bb in [(ii,jj,aid,bid),(jj,ii,bid,aid)]:
                        pool=(common&raster[aa]&~raster[bb]).flatten().nonzero().flatten()
                        if not len(pool):continue
                        rng=np.random.default_rng(20260912+iid+aa*7+bb*13);pick=pool[torch.tensor(rng.integers(len(pool),size=64),device='cuda')]
                        assert (raster[aa].flatten()[pick]&~raster[bb].flatten()[pick]&common.flatten()[pick]).all()
                        pairps.append(sample_proto(input_proto,pick,shape).cpu().numpy());src.append(s);dst.append(t);pairpos.append(pick.cpu().numpy())
            t=len(targetids);ix=torch.tensor(predidx,device='cuda',dtype=torch.long)
            arrays=dict(x=x[ix].cpu().numpy(),c=c[ix].cpu().numpy(),p=np.stack(ps) if t else np.empty((0,256,32),np.float32),y=np.stack(ys) if t else np.empty((0,256),bool),
                positive_p=np.stack(positives) if t else np.empty((0,64,32),np.float32),positive_valid=np.array(haspositive,bool),pair_p=np.stack(pairps) if pairps else np.empty((0,64,32),np.float32),
                pair_source=np.array(src,np.int64),pair_neighbor=np.array(dst,np.int64),edges=np.array(edges,np.int64).reshape(-1,2),annotation_ids=np.array(targetids,np.int64),prediction_indices=np.array(predidx,np.int64),
                sampled_positions=np.stack(samplepositions) if t else np.empty((0,256),np.int64),positive_positions=np.stack(positivepositions) if t else np.empty((0,64),np.int64),pair_positions=np.stack(pairpos) if pairpos else np.empty((0,64),np.int64),input_shape=np.array(shape),original_shape=np.array(cap['shape']))
            path=out/'train'/f'{iid}.npz';np.savez_compressed(path,**arrays);hashes[path.name]=sha(path)
            rows.extend(dict(image_id=iid,**r) for r in statuses)
            if n%50==0 or n==len(chosen):
                progress=dict(images=n,total=len(chosen),seconds=time.monotonic()-started,targets=sum(r['status']=='matched' for r in rows));print(json.dumps(progress),flush=True);write_json(out/'progress.json',progress)
    for h in hooks:h.remove()
    write_csv(out/'gt_status.csv',rows);write_json(out/'CACHE_HASHES.json',hashes);write_json(out/'IMAGE_HASHES.json',imagehash)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',images=len(chosen),newly_extracted=extracted,targets=sum(r['status']=='matched' for r in rows),sample_replay_max_abs=maxreplay,seconds=time.monotonic()-started,
        runtime_end2end=bool(head.end2end),hashes={f.name:sha(f) for f in out.iterdir() if f.is_file()}))

if __name__=='__main__':main()
