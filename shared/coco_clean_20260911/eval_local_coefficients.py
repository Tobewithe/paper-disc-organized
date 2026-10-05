"""Evaluate global versus spatially varying coefficient residuals using original COCO annotations.

Prediction sees only cached network outputs. GT is first consulted after an entire
image's masks have been constructed. Zero residual must replay official decoding.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,csv,gzip,io,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskutils
from ultralytics.utils import ops
from local_coefficient_head import LocalCoefficientHead,gates
from frozen_mechanism_probe import ROOT,write_json,write_csv,sha,box_iou,pixel_metrics


@torch.inference_mode()
def decode(item,models):
    from spatial_coefficient_head import features
    p=torch.as_tensor(item['proto'],device='cuda');c=torch.as_tensor(item['coeff'],device='cuda');boxes=torch.as_tensor(item['boxes'],device='cuda');shape=tuple(map(int,item['input_shape']));h,w=shape
    if not len(c):return {arm:torch.empty((0,h,w),dtype=torch.uint8,device='cuda') for arm in ['initial',*models]},0
    z=F.interpolate((c@p.float().flatten(1)).reshape(1,-1,*p.shape[-2:]),shape,mode='bilinear',align_corners=False)[0]
    initial=ops.crop_mask((z>0).to(torch.uint8),boxes);stock=ops.process_mask(p,c,boxes,shape,upsample=True);assert torch.equal(initial,stock)
    x=features(torch.tensor(item['features'],device='cuda'),torch.tensor(item['level'],device='cuda'),boxes,shape)[:,:73]
    x=(x-NORMALIZER['mean'])/NORMALIZER['std'];result={'initial':initial}
    yy,xx=torch.meshgrid(torch.arange(h,device='cuda')+.5,torch.arange(w,device='cuda')+.5,indexing='ij');points=torch.stack([xx,yy],-1)
    for arm,model in models.items():
        dc=model(x);binary=torch.zeros_like(initial)
        for j in range(len(c)):
            residual=F.interpolate((dc[j]@p.float().flatten(1)).reshape(1,4,*p.shape[-2:]),shape,mode='bilinear',align_corners=False)[0]
            if arm.startswith('global_'):rz=residual.mean(0)
            else:
                uv=(points-boxes[j,:2])/(boxes[j,2:]-boxes[j,:2]).clamp_min(1);g=gates(uv);rz=(residual*g.permute(2,0,1)).sum(0)
            binary[j]=(z[j]+rz>0).to(torch.uint8)
        result[arm]=ops.crop_mask(binary,boxes)
    return result,0

def evaluate(gt,meta,ids,pred,out,arm):
    with gzip.open(out/'predictions'/f'{arm}.json.gz','wt',encoding='utf-8') as f:json.dump(pred,f,separators=(',',':'))
    with contextlib.redirect_stdout(io.StringIO()):
        dt=gt.loadRes(pred);ev=COCOeval(gt,dt,'segm');ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
    row=dict(arm=arm,mask_ap=float(ev.stats[0]),mask_ap50=float(ev.stats[1]),mask_ap75=float(ev.stats[2]),predictions=len(pred))
    np.savez_compressed(out/f'cocoeval_{arm}.npz',precision=ev.eval['precision'],recall=ev.eval['recall'],scores=ev.eval['scores'])
    t=int(np.flatnonzero(np.isclose(ev.params.iouThrs,.75))[0]);recovered={};counts={g:[0,0] for g in ['all','high','low']};allgt=[]
    for r in ev.evalImgs:
        if r is None or r['aRng']!=[0,1e10] or r['maxDet']!=100:continue
        for k,aid in enumerate(r['gtIds']):
            if r['gtIgnore'][k]:continue
            aid=int(aid);hit=bool(r['gtMatches'][t,k]);recovered[aid]=hit;ici=float(meta[aid]['ici_same']);group='high' if ici>.5+1e-10 else 'low'
            for g in ['all',group]:counts[g][0]+=int(hit);counts[g][1]+=1
            allgt.append(dict(arm=arm,image_id=r['image_id'],target_annotation=aid,target_ici=ici,mask_recovered75=hit))
    for g,(hit,total) in counts.items():row['r75_'+g]=hit/max(total,1);row['gt_'+g]=total
    pairhit=pairtotal=0
    for iid in ids:
        anns=[a for a in gt.imgToAnns[iid] if not a.get('iscrowd',0)]
        for j,a in enumerate(anns):
            for b in anns[j+1:]:
                if a['category_id']==b['category_id'] and box_iou(a['bbox'],b['bbox'])>.05:
                    pairtotal+=1;pairhit+=recovered.get(a['id'],False) and recovered.get(b['id'],False)
    row['pair_recovery75']=pairhit/max(pairtotal,1);row['pairs']=pairtotal
    return row,allgt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--training',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    cache=a.cache.resolve();trained=a.training.resolve();out=a.out.resolve();out.mkdir(exist_ok=False,parents=True);(out/'predictions').mkdir()
    assert (cache/'COMPLETE.json').exists() and (trained/'COMPLETE.json').exists()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    global NORMALIZER
    NORMALIZER=torch.load(trained/'normalizer.pt',map_location='cuda',weights_only=True)
    models={};checkpoint_hashes={}
    for seed in [0,1,2]:
        for mode in ['global','local']:
            arm=f'{mode}_s{seed}';file=trained/arm/'checkpoints/epoch15.pt'
            model=LocalCoefficientHead().cuda().eval();model.load_state_dict(torch.load(file,map_location='cuda',weights_only=True));models[arm]=model;checkpoint_hashes[arm]=sha(file)
    paths=sorted((cache/'val').glob('*.npz'));paths=paths[:a.limit] if a.limit else paths;ids=[int(p.stem) for p in paths]
    write_json(out/'protocol.json',dict(script_sha256=sha(__file__),feature_script_sha256=sha(Path(__file__).with_name('local_coefficient_head.py')),checkpoint_sha256=checkpoint_hashes,train_protocol_sha256=sha(trained/'protocol.json'),cache_protocol_sha256=sha(cache/'protocol.json'),images=ids,arms=['initial',*models],GT_used_in_prediction=False,
        decoder='Upsample coefficient-prototype logits to input; add bilinearly mixed four coefficient residual maps inside predicted box; threshold zero; crop; stock scale_masks to original resolution. Zero residual is checked against stock process_mask on every image.',
        training_evaluation_sampling_difference='Training logits sampled after resizing prototypes to original pixels; evaluation refines input-grid logits before binary resize. Same pixel-center box coordinates; rounded letterbox padding may differ by <1 input pixel.',
        limitation='Frozen original nonempty detections at conf=.001 max_det=300, official COCOeval maxDets=100 per category. Dense-enriched exploratory validation, not full COCO or an untouched test set.'))
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    with (ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv').open(encoding='utf-8-sig') as f:meta={int(r['annotation_id']):r for r in csv.DictReader(f)}
    categories=sorted(gt.cats);preds={arm:[] for arm in ['initial',*models]};boxpred=[];pixels=[];skipped=[];start=time.monotonic();parity=0
    for n,path in enumerate(paths,1):
        iid=int(path.stem);item=np.load(path)
        masks,xor=decode({key:item[key] for key in ['proto','coeff','boxes','detections','input_shape','features','level']},models);parity+=xor
        # GT ownership is used only below this line, after all predictions exist.
        det=item['detections'];shape=tuple(map(int,item['shape']));mapping=dict(zip(map(int,item['mapping_gt']),map(int,item['mapping_pred'])))
        anns=gt.imgToAnns[iid];raster={a['id']:torch.tensor(gt.annToMask(a).astype(bool),device='cuda') for a in anns};ordinary=[a for a in anns if not a.get('iscrowd',0)]
        union=torch.stack([raster[a['id']] for a in ordinary]).any(0) if ordinary else torch.zeros(shape,dtype=torch.bool,device='cuda')
        same={cat:torch.stack([raster[a['id']] for a in ordinary if a['category_id']==cat]).any(0) for cat in {a['category_id'] for a in ordinary}}
        crowd=[raster[a['id']] for a in anns if a.get('iscrowd',0)];valid=~torch.stack(crowd).any(0) if crowd else torch.ones(shape,dtype=torch.bool,device='cuda')
        for d in det:boxpred.append(dict(image_id=iid,category_id=categories[int(d[5])],score=float(d[4]),bbox=[float(d[0]),float(d[1]),float(d[2]-d[0]),float(d[3]-d[1])]))
        for arm,binary in masks.items():
            for first in range(0,len(binary),48):
                chunk=binary[first:first+48];orig=ops.scale_masks(chunk[:,None],shape)[:,0]>.5;nonempty=chunk.flatten(1).any(1)
                for local in range(len(chunk)):
                    j=first+local
                    if bool(nonempty[local]):
                        rle=maskutils.encode(np.asfortranarray(orig[local].cpu().numpy().astype(np.uint8)));rle['counts']=rle['counts'].decode('ascii')
                        preds[arm].append(dict(image_id=iid,category_id=categories[int(det[j,5])],score=float(det[j,4]),segmentation=rle))
                for aid,j in mapping.items():
                    if first<=j<first+len(chunk):
                        if not (raster[aid]&valid).any():skipped.append(dict(arm=arm,image_id=iid,annotation_id=aid,reason='no_valid_gt_pixels'));continue
                        values=pixel_metrics(orig[j-first:j-first+1],raster[aid],union,same[gt.anns[aid]['category_id']],valid)
                        pixels.append(dict(arm=arm,image_id=iid,target_annotation=aid,target_ici=float(meta[aid]['ici_same']),**{k:float(v[0]) for k,v in values.items()}))
        if n%10==0:
            progress=dict(phase='decode',completed=n,total=len(paths),seconds=round(time.monotonic()-start,1));write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    write_csv(out/'spatial.csv',pixels);write_json(out/'spatial_skipped.json',skipped);statistics=[];allgt=[]
    for arm,pred in preds.items():
        row,rows=evaluate(gt,meta,ids,pred,out,arm);statistics.append(row);allgt.extend(rows);print(json.dumps(row),flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        dt=gt.loadRes(boxpred);ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
    write_csv(out/'summary.csv',statistics);write_csv(out/'gt_recovery.csv',allgt)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',images=len(ids),arms=len(preds),box_ap_common=float(ev.stats[0]),initial_replay_pixel_xor=parity,seconds=time.monotonic()-start,script_sha256=sha(__file__)))

if __name__=='__main__':main()

