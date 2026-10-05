"""No-training COCO diagnostic on official weights; fixed 5-degree primary probe.

Interventions change temporary coefficient tensors only. All model parameters,
prototypes, boxes, scores and GT ownership stay fixed. GT oracle is diagnostic.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse
import contextlib
import copy
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import ops

ROOT=Path(__file__).resolve().parent
SEED=20260911
FIELDS=('coverage','same_neighbor','neighbor','background','mask_iou')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def write_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def rank(value):return hashlib.sha256(f'{SEED}:{value}'.encode()).hexdigest()

def box_iou(a,b):
    a=np.asarray(a);b=np.asarray(b)
    wh=np.maximum(0,np.minimum(a[:2]+a[2:],b[:2]+b[2:])-np.maximum(a[:2],b[:2]))
    inter=wh.prod();return float(inter/(a[2:].prod()+b[2:].prod()-inter+1e-12))

def selection(gt,metadata,n):
    groups={'high':[],'low':[]}
    for iid in sorted(gt.imgs):
        anns=[a for a in gt.imgToAnns[iid] if not a.get('iscrowd',0)]
        pairs=[]
        for ai,a in enumerate(anns):
            for b in anns[ai+1:]:
                if a['category_id']!=b['category_id']:continue
                iou=box_iou(a['bbox'],b['bbox'])
                if iou<=.05:continue
                aa,bb=sorted([a['id'],b['id']])
                ici=max(float(metadata[aa]['ici_same']),float(metadata[bb]['ici_same']))
                pairs.append(dict(image_id=iid,annotation_a=aa,annotation_b=bb,gt_box_iou=iou,
                                  pair_group='high' if ici>.5 else 'low'))
        if pairs:
            # One fixed GT-only pair per image; no predictions in selection.
            pair=min(pairs,key=lambda p:rank(f"{iid}:{p['annotation_a']}:{p['annotation_b']}"))
            groups[pair['pair_group']].append(pair)
    high_n=(2*n)//3;low_n=n-high_n
    chosen=[]
    for group,count in [('high',high_n),('low',low_n)]:
        pool=sorted(groups[group],key=lambda p:rank(p['image_id']))
        if len(pool)<count:raise ValueError(f'Insufficient {group} GT-only support: {len(pool)} < {count}')
        chosen.extend(pool[:count])
    return sorted(chosen,key=lambda p:rank(p['image_id']))

class Capture(SegmentationPredictor):
    def construct_result(self,pred,img,orig_img,img_path,proto):
        raw=pred.detach().clone()
        result=super().construct_result(pred,img,orig_img,img_path,proto)
        replay=ops.process_mask(proto,raw[:,6:],raw[:,:4],img.shape[2:],upsample=True)
        keep=replay.amax((-2,-1))>0 if len(raw) else torch.zeros(0,dtype=torch.bool,device=raw.device)
        if keep.any():assert torch.equal(replay[keep],result.masks.data),'Official replay mismatch'
        self.capture=dict(proto=proto.detach().clone(),coeff=raw[keep,6:].detach().clone(),
            boxes=raw[keep,:4].detach().clone(),detections=result.boxes.data.detach().clone(),
            shape=tuple(orig_img.shape[:2]),input_shape=tuple(img.shape[2:]))
        return result

def ownership(gt,iid,boxes):
    category_ids=sorted(gt.cats)
    with contextlib.redirect_stdout(io.StringIO()):
        local=COCO();local.dataset=dict(info={},images=[gt.imgs[iid]],categories=list(gt.cats.values()),annotations=gt.imgToAnns[iid]);local.createIndex()
        preds=[dict(image_id=iid,category_id=category_ids[int(b[5])],score=float(b[4]),
                    bbox=[float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1])]) for b in boxes.cpu().numpy()]
        if preds:dt=local.loadRes(preds)
        else:
            dt=COCO();dt.dataset=dict(images=local.dataset['images'],categories=local.dataset['categories'],annotations=[]);dt.createIndex()
        ev=COCOeval(local,dt,'bbox');ev.params.imgIds=[iid];ev.params.iouThrs=np.array([.5]);ev.params.areaRng=[[0,1e10]];ev.params.areaRngLbl=['all'];ev.params.maxDets=[100];ev.evaluate()
    mapped={}
    for r in ev.evalImgs:
        if r is None:continue
        for k,aid in enumerate(r['gtIds']):
            if not r['gtIgnore'][k] and r['gtMatches'][0,k]>0:mapped[int(aid)]=int(r['gtMatches'][0,k])-1
    assert len(mapped.values())==len(set(mapped.values()))
    return mapped

def cosine(c):
    norms=np.linalg.norm(c,axis=1)
    if min(norms)<1e-10:raise ValueError('zero_norm')
    return float(np.clip(np.dot(c[0],c[1])/norms.prod(),-1,1))

def pair_rotations(c,degrees,seed):
    c=np.asarray(c,dtype=np.float64);norms=np.linalg.norm(c,axis=1);rho=cosine(c)
    phi=math.acos(rho)
    angle=min(math.radians(degrees),phi/2-1e-6,(math.pi-phi)/2-1e-6)
    if angle<=1e-6:raise ValueError('collinear')
    units=c/norms[:,None];u,v=units;e2=(v-rho*u)/math.sin(phi)
    cs,sn=math.cos(angle),math.sin(angle)
    push=np.stack([cs*u-sn*e2,math.cos(phi+angle)*u+math.sin(phi+angle)*e2])*norms[:,None]
    pull=np.stack([cs*u+sn*e2,math.cos(phi-angle)*u+math.sin(phi-angle)*e2])*norms[:,None]
    rng=np.random.default_rng(seed);t=rng.normal(size=c.shape);t-=(t*units).sum(1)[:,None]*units;t/=np.linalg.norm(t,axis=1)[:,None]
    variants=dict(push=push,pull=pull,random_plus=(cs*units+sn*t)*norms[:,None],random_minus=(cs*units-sn*t)*norms[:,None])
    assert cosine(push)<rho and cosine(pull)>rho
    for x in variants.values():
        np.testing.assert_allclose(np.linalg.norm(x,axis=1),norms,rtol=1e-9,atol=1e-9)
    return variants,angle

def original_logits(c,p,capture):
    logits=(c@p.flatten(1)).reshape(-1,*p.shape[1:])
    padded=F.interpolate(logits[None],capture['input_shape'],mode='bilinear',align_corners=False)[0]
    return ops.scale_masks(padded[:,None],capture['shape'])[:,0]

def decode(c,p,boxes,capture):
    logits=(c@p.flatten(1)).reshape(-1,*p.shape[1:])
    padded=F.interpolate(logits[None],capture['input_shape'],mode='bilinear',align_corners=False)[0]
    binary=(padded>0).to(torch.uint8)
    cropped=ops.crop_mask(binary.clone(),boxes)
    result={}
    for name,m in [('cropped',cropped),('raw',binary)]:
        result[name]=ops.scale_masks(m[:,None],capture['shape'])[:,0]>.5
    return result,logits

def pixel_metrics(m,own,union,same,valid):
    m=m&valid;own=own&valid;ga=own.sum().double()
    if ga==0:raise ValueError('no_valid_gt_pixels')
    tp=(m&own).sum((-2,-1)).double()
    near=(m&union&~own).sum((-2,-1)).double()
    snear=(m&same&~own).sum((-2,-1)).double()
    bg=(m&~union).sum((-2,-1)).double()
    assert torch.equal(tp+near+bg,m.sum((-2,-1)).double())
    values=[tp/ga,snear/ga,near/ga,bg/ga,tp/(ga+near+bg)]
    assert torch.allclose(values[-1],values[0]/(1+values[2]+values[3]),atol=1e-12)
    return {k:v.cpu().numpy() for k,v in zip(FIELDS,values)}

def oracle_direction(c,p,capture,own,same,union,valid,angle):
    # Gradients target temporary 2x32 coefficients, never model parameters.
    x=c.detach().clone().requires_grad_(True)
    z=original_logits(x,p,capture)
    terms=[]
    for j in range(2):
        domains=[own[j]&valid,same&~own[j]&valid,~union&valid]
        for k,region in enumerate(domains):
            if region.any():terms.append(F.softplus(-z[j][region] if k==0 else z[j][region]).mean())
    objective=sum(terms)
    grad=torch.autograd.grad(objective,x)[0]
    units=F.normalize(c,dim=1);t=-grad+((grad*units).sum(1)[:,None])*units
    tn=t.norm(dim=1);active=tn>1e-12
    t=t/tn.clamp_min(1e-12)[:,None]
    changed=(math.cos(angle)*units+math.sin(angle)*t)*c.norm(dim=1)[:,None]
    changed=torch.where(active[:,None],changed,c)
    return changed.detach().cpu().numpy(),dict(objective=float(objective.detach()),zero_directions=int((~active).sum()))

def probe(gt,pair,capture,mapping,metadata,out):
    iid=pair['image_id'];ids=[pair['annotation_a'],pair['annotation_b']]
    status={**pair,'matched_a':ids[0] in mapping,'matched_b':ids[1] in mapping}
    if not all(a in mapping for a in ids):return [],[],dict(**status,status='unmatched')
    indices=[mapping[a] for a in ids]
    p=capture['proto'].float();c=capture['coeff'][indices].float();b=capture['boxes'][indices].float()
    cnp=c.cpu().numpy();dev=p.device
    anns=gt.imgToAnns[iid]
    raster={a['id']:torch.as_tensor(gt.annToMask(a).astype(bool),device=dev) for a in anns}
    ordinary=[a for a in anns if not a.get('iscrowd',0)]
    union=torch.stack([raster[a['id']] for a in ordinary]).any(0)
    same=torch.stack([raster[a['id']] for a in ordinary if a['category_id']==gt.anns[ids[0]]['category_id']]).any(0)
    crowds=[raster[a['id']] for a in anns if a.get('iscrowd',0)]
    valid=~torch.stack(crowds).any(0) if crowds else torch.ones(capture['shape'],device=dev,dtype=torch.bool)
    own=[raster[a] for a in ids]
    variants=[('zero',0.,0.,cnp)]
    oracle_info=[]
    for deg in [2.,5.,10.]:
        changed,angle=pair_rotations(cnp,deg,SEED+iid)
        for name,x in changed.items():variants.append((name,deg,math.degrees(angle),x))
        x,info=oracle_direction(c,p,capture,own,same,union,valid,angle)
        variants.append(('spatial_oracle',deg,math.degrees(angle),x));oracle_info.append(info)
    stacked=torch.as_tensor(np.concatenate([x[3] for x in variants]),dtype=torch.float32,device=dev)
    masks,z=decode(stacked,p,b.repeat(len(variants),1),capture)
    official=ops.process_mask(p,c,b,capture['input_shape'],upsample=True)
    official=ops.scale_masks(official[:,None],capture['shape'])[:,0]>.5
    xor=int((masks['cropped'][:2]!=official).sum())
    assert xor==0,('zero_replay_mismatch',xor)
    z=z.reshape(len(variants),2,-1)
    # Exact coordinate rescaling: c'=cD, P'=D^-1 P, Z unchanged.
    gauge=[]
    for sign in [-1,1]:
        scale=2.**(sign*(torch.arange(c.shape[1],device=dev)%3-1).float())
        cz=c*scale;pp=p/scale[:,None,None]
        altered=cz@pp.flatten(1)
        maxdiff=float((altered-z[0]).abs().max());rawxor=int(((altered>0)!=(z[0]>0)).sum())
        gauge.append(dict(image_id=iid,sign=sign,original_cosine=cosine(cnp),rescaled_cosine=cosine(cz.cpu().numpy()),
                          max_logit_difference=maxdiff,raw_pixel_xor=rawxor))
        assert maxdiff<1e-4 and rawxor==0,('coordinate_rescale_parity',maxdiff,rawxor)
    positive=z.clamp_min(0);q=(positive[:,0]*positive[:,1]).mean(1)
    zcos=F.cosine_similarity(z[:,0],z[:,1],dim=1)
    raw=z>0;rawiou=(raw[:,0]&raw[:,1]).sum(1)/(raw[:,0]|raw[:,1]).sum(1).clamp_min(1)
    zrms=(z-z[0]).square().mean((1,2)).sqrt()/z[0].square().mean().sqrt().clamp_min(1e-12)
    rows=[]
    for domain,mask in masks.items():
        mask=mask.reshape(len(variants),2,*capture['shape'])
        values=[pixel_metrics(mask[:,j],own[j],union,same,valid) for j in range(2)]
        for k,(mode,degree,effective,x) in enumerate(variants):
            for j,aid in enumerate(ids):
                row=dict(image_id=iid,annotation_a=ids[0],annotation_b=ids[1],target_annotation=aid,
                    category_id=gt.anns[aid]['category_id'],target_ici=float(metadata[aid]['ici_same']),pair_group=pair['pair_group'],
                    gt_box_iou=pair['gt_box_iou'],domain=domain,mode=mode,degrees=degree,effective_degrees=effective,
                    coefficient_cosine=cosine(x),logit_cosine=float(zcos[k]),positive_product=float(q[k]),
                    raw_pair_iou=float(rawiou[k]),relative_logit_change=float(zrms[k]))
                row.update({field:float(values[j][field][k]) for field in FIELDS});rows.append(row)
    np.savez_compressed(out/'tensors'/f'{iid}.npz',proto=p.cpu().numpy(),coeff=cnp,boxes=b.cpu().numpy(),
        shape=capture['shape'],input_shape=capture['input_shape'],annotation_ids=ids,variants=stacked.cpu().numpy())
    return rows,gauge,dict(**status,status='ok',zero_replay_xor=xor,oracle=oracle_info)

def summarize(rows,selected,out,B=2000):
    image_ids=[p['image_id'] for p in selected];lookup={iid:i for i,iid in enumerate(image_ids)}
    rng=np.random.default_rng(SEED);draws=rng.integers(len(image_ids),size=(B,len(image_ids)))
    weights=np.asarray([np.bincount(d,minlength=len(image_ids)) for d in draws],dtype=np.float64)
    np.savez_compressed(out/'bootstrap_draws.npz',image_ids=image_ids,draws=draws)
    def estimate(records):
        sums=np.zeros(len(image_ids));counts=np.zeros(len(image_ids))
        for iid,v in records:sums[lookup[iid]]+=v;counts[lookup[iid]]+=1
        if not counts.sum():return None
        denominator=weights@counts;valid=denominator>0;boot=(weights[valid]@sums)/denominator[valid]
        return dict(mean=float(sums.sum()/counts.sum()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                    targets=int(counts.sum()),images=int((counts>0).sum()))
    zero={(r['image_id'],r['target_annotation'],r['domain']):r for r in rows if r['mode']=='zero'}
    stats=[]
    for domain in ['cropped','raw']:
        for group in ['all','high','low']:
            subset=[r for r in rows if r['domain']==domain and (group=='all' or (r['target_ici']>.5)==(group=='high'))]
            for deg in [2.,5.,10.]:
                arms={mode:{(r['image_id'],r['target_annotation']):r for r in subset if r['mode']==mode and r['degrees']==deg}
                      for mode in ['push','pull','random_plus','random_minus','spatial_oracle']}
                for mode in ['push','pull','spatial_oracle']:
                    for control in ['zero','random_mean']:
                        for metric in FIELDS:
                            values=[]
                            for key,r in arms[mode].items():
                                base=zero[(*key,domain)][metric] if control=='zero' else (arms['random_plus'][key][metric]+arms['random_minus'][key][metric])/2
                                values.append((key[0],r[metric]-base))
                            st=estimate(values)
                            if st:stats.append(dict(domain=domain,group=group,degrees=deg,contrast=mode+'_minus_'+control,metric=metric,**st))
    write_csv(out/'summary.csv',stats)
    return stats

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--images',type=int,default=300);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--resume',action='store_true');a=parser.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);(out/'tensors').mkdir(exist_ok=True);(out/'per_image').mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    weight=ROOT/'weights/yolo26m-seg.pt';gt_path=ROOT/'data/annotations/instances_val2017.json'
    manifest=ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv'
    assert sha(weight)=='16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5'
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(gt_path))
    with manifest.open(encoding='utf-8-sig',newline='') as f:meta={int(r['annotation_id']):r for r in csv.DictReader(f)}
    chosen=selection(gt,meta,a.images)
    provenance=dict(weight_sha256=sha(weight),gt_sha256=sha(gt_path),metadata_sha256=sha(manifest),script_sha256=sha(__file__),
        torch=torch.__version__,training=False,model='official pretrained YOLO26m-seg',primary_angle=5,
        inference=dict(imgsz=640,batch=1,rect=False,half=False,conf=.001,max_det=300),
        sampling='GT only: 2/3 images with selected pair max ICI>.5, 1/3 <=.5; one hash-selected same-class box IoU>.05 pair/image',
        limitations='Conditional bbox>=.5 matched-pair diagnostic; image stratification is enriched, not population-weighted. GT direction is oracle. Not AP or training effect.')
    previous=read_previous=out/'protocol.json'
    if previous.exists():assert json.loads(previous.read_text())==provenance,'Frozen protocol changed'
    else:write_json(previous,provenance)
    write_json(out/'selection.json',chosen)
    model=YOLO(str(weight));model.model.eval();model.model.requires_grad_(False)
    rows=[];gauges=[];statuses=[];start=time.monotonic()
    for k,pair in enumerate(chosen,1):
        saved=out/'per_image'/f"{pair['image_id']}.json"
        if saved.exists() and a.resume:
            item=json.loads(saved.read_text());new=item['rows'];g=item['gauge'];status=item['status']
        else:
            image=ROOT/'official_yolo/images/val2017'/gt.imgs[pair['image_id']]['file_name']
            with torch.no_grad():
                model.predict(str(image),predictor=Capture,imgsz=640,conf=.001,max_det=300,iou=.7,device=0,rect=False,half=False,retina_masks=False,verbose=False)
                # Clone outside inference_mode so temporary coefficient gradients are legal.
                captured={key:value.clone() if torch.is_tensor(value) else value for key,value in model.predictor.capture.items()}
                mapping=ownership(gt,pair['image_id'],captured['detections'])
            try:new,g,status=probe(gt,pair,captured,mapping,meta,out)
            except ValueError as e:new=[];g=[];status=dict(**pair,status=str(e))
            write_json(saved,dict(rows=new,gauge=g,status=status))
        rows.extend(new);gauges.extend(g);statuses.append(status)
        if k%10==0 or k==len(chosen):
            progress=dict(status='RUNNING',completed_images=k,total_images=len(chosen),valid_pairs=sum(s['status']=='ok' for s in statuses),elapsed_seconds=round(time.monotonic()-start,1))
            write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    write_csv(out/'interventions.csv',rows);write_csv(out/'coordinate_rescaling.csv',gauges);write_json(out/'statuses.json',statuses)
    statistics=summarize(rows,chosen,out)
    result=dict(status='COMPLETE',images=len(chosen),valid_pairs=sum(s['status']=='ok' for s in statuses),rows=len(rows),
                zero_replay_xor=sum(s.get('zero_replay_xor',0) for s in statuses),gauge_raw_xor=sum(g['raw_pixel_xor'] for g in gauges),
                elapsed_seconds=time.monotonic()-start,completed=datetime.now(timezone.utc).isoformat())
    write_json(out/'COMPLETE.json',result);write_json(out/'progress.json',result);print(json.dumps(result),flush=True)
    for r in statistics:
        if r['domain']=='cropped' and r['group']=='high' and r['degrees']==5 and r['contrast'] in ['push_minus_zero','spatial_oracle_minus_zero']:
            print(json.dumps(r),flush=True)

if __name__=='__main__':main()
