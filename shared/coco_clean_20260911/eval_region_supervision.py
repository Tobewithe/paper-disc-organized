"""Original COCO evaluation of fixed-detection, region-supervised coefficient-head pilots.

Predict every cached detection before reading its GT ownership; scores/boxes fixed.
Only final epoch is evaluated. Official COCOeval provides AP and per-GT recall.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,csv,gzip,io,json,time
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskutils
from ultralytics.utils import ops
from frozen_mechanism_probe import ROOT,pixel_metrics,write_json,write_csv,sha,box_iou

METRICS=['coverage','same_neighbor','neighbor','background','mask_iou']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--training',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    cache=args.cache.resolve();trained=args.training.resolve();out=args.out.resolve();out.mkdir(exist_ok=False,parents=True);(out/'predictions').mkdir()
    assert (cache/'COMPLETE.json').exists() and (trained/'COMPLETE.json').exists()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    with (ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv').open(encoding='utf-8-sig') as f:meta={int(r['annotation_id']):r for r in csv.DictReader(f)}
    arms={'initial':np.load(cache/'initial_head.npz')}
    for seed in [0,1,2]:
        for mode in ['bce','neighbor','background']:arms[f'{mode}_s{seed}']=np.load(trained/f'{mode}_s{seed}/checkpoints/epoch15.npz')
    weights={k:{f:torch.tensor(v[f],device='cuda') for f in ['weight','bias']} for k,v in arms.items()}
    paths=sorted((cache/'val').glob('*.npz'));ids=[int(p.stem) for p in paths];categories=sorted(gt.cats)
    write_json(out/'protocol.json',dict(script_sha256=sha(__file__),train_protocol_sha256=sha(trained/'protocol.json'),cache_protocol_sha256=sha(cache/'protocol.json'),
        arms=list(arms),images=ids,GT_used_in_prediction=False,official_evaluator='pycocotools COCOeval segm and bbox, default thresholds and maxDets',
        masks='official process_mask then scale_masks, retained original detection score/category; empty masks removed as stock predictor',
        limitation='Frozen original nonempty detection candidate set; previously empty masks cannot become new detections. Conditional pilot AP on a dense-enriched 300-image subset, not full COCO AP.'))
    preds={k:[] for k in arms};boxpred=[];pixels=[];skipped=[];start=time.monotonic();parity=0
    for n,path in enumerate(paths,1):
        iid=int(path.stem);item=np.load(path);x=torch.tensor(item['features'],device='cuda');level=torch.tensor(item['level'],device='cuda');p=torch.tensor(item['proto'],device='cuda');boxes=torch.tensor(item['boxes'],device='cuda');det=item['detections'];shape=tuple(map(int,item['shape']));ishape=tuple(map(int,item['input_shape']))
        mapping={int(k):int(v) for k,v in zip(item['mapping_gt'],item['mapping_pred'])}
        anns=gt.imgToAnns[iid];raster={a['id']:torch.tensor(gt.annToMask(a).astype(bool),device='cuda') for a in anns};ordinary=[a for a in anns if not a.get('iscrowd',0)]
        union=torch.stack([raster[a['id']] for a in ordinary]).any(0) if ordinary else torch.zeros(shape,dtype=torch.bool,device='cuda')
        same={cat:torch.stack([raster[a['id']] for a in ordinary if a['category_id']==cat]).any(0) for cat in {a['category_id'] for a in ordinary}}
        crowd=[raster[a['id']] for a in anns if a.get('iscrowd',0)];valid=~torch.stack(crowd).any(0) if crowd else torch.ones(shape,dtype=torch.bool,device='cuda')
        for j,d in enumerate(det):boxpred.append(dict(image_id=iid,category_id=categories[int(d[5])],score=float(d[4]),bbox=[float(d[0]),float(d[1]),float(d[2]-d[0]),float(d[3]-d[1])]))
        for arm,state in weights.items():
            coeff=torch.tensor(item['coeff'],device='cuda') if arm=='initial' else torch.einsum('nki,ni->nk',state['weight'][level],x)+state['bias'][level]
            if arm=='initial':
                err=float((coeff-torch.tensor(item['coeff'],device='cuda')).abs().max()) if len(coeff) else 0;assert err<1e-5
            # Decode in chunks to bound memory for 300 candidates and original large images.
            for first in range(0,len(coeff),48):
                c=coeff[first:first+48];b=boxes[first:first+48];binary=ops.process_mask(p,c,b,ishape,upsample=True)
                if arm=='initial':
                    baseline=ops.process_mask(p,torch.tensor(item['coeff'][first:first+48],device='cuda'),b,ishape,upsample=True)
                    xor=int((binary!=baseline).sum());parity+=xor
                orig=ops.scale_masks(binary[:,None],shape)[:,0]>.5;nonempty=binary.flatten(1).any(1)
                for local in range(len(c)):
                    j=first+local
                    if bool(nonempty[local]):
                        rle=maskutils.encode(np.asfortranarray(orig[local].cpu().numpy().astype(np.uint8)));rle['counts']=rle['counts'].decode('ascii')
                        preds[arm].append(dict(image_id=iid,category_id=categories[int(det[j,5])],score=float(det[j,4]),segmentation=rle))
                for aid,j in mapping.items():
                    if first<=j<first+len(c):
                        if not (raster[aid]&valid).any():
                            skipped.append(dict(arm=arm,image_id=iid,target_annotation=aid,reason='no_valid_gt_pixels_after_crowd_exclusion'));continue
                        values=pixel_metrics(orig[j-first:j-first+1],raster[aid],union,same[gt.anns[aid]['category_id']],valid)
                        pixels.append(dict(arm=arm,image_id=iid,target_annotation=aid,target_ici=float(meta[aid]['ici_same']),**{m:float(values[m][0]) for m in METRICS}))
        if n%50==0:print(json.dumps(dict(phase='decode',completed=n,total=len(paths),seconds=round(time.monotonic()-start,1))),flush=True)
    # Floating last-layer replay can very rarely flip near-zero logits; disclose exact count.
    write_csv(out/'spatial.csv',pixels);write_json(out/'spatial_skipped.json',skipped);statistics=[];allgt=[]
    for arm,pred in preds.items():
        with gzip.open(out/'predictions'/f'{arm}.json.gz','wt',encoding='utf-8') as f:json.dump(pred,f,separators=(',',':'))
        with contextlib.redirect_stdout(io.StringIO()):
            dt=gt.loadRes(pred);ev=COCOeval(gt,dt,'segm');ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
        row=dict(arm=arm,mask_ap=float(ev.stats[0]),mask_ap50=float(ev.stats[1]),mask_ap75=float(ev.stats[2]),predictions=len(pred))
        np.savez_compressed(out/f'cocoeval_{arm}.npz',precision=ev.eval['precision'],recall=ev.eval['recall'],scores=ev.eval['scores'])
        t=int(np.flatnonzero(np.isclose(ev.params.iouThrs,.75))[0]);recovered={};counts={'all':[0,0],'high':[0,0],'low':[0,0]}
        for r in ev.evalImgs:
            if r is None or r['aRng']!=[0,1e10] or r['maxDet']!=100:continue
            for k,aid in enumerate(r['gtIds']):
                if r['gtIgnore'][k]:continue
                hit=bool(r['gtMatches'][t,k]);recovered[int(aid)]=hit;group='high' if float(meta[int(aid)]['ici_same'])>.5+1e-10 else 'low'
                for g in ['all',group]:counts[g][0]+=int(hit);counts[g][1]+=1
                allgt.append(dict(arm=arm,image_id=r['image_id'],target_annotation=int(aid),target_ici=float(meta[int(aid)]['ici_same']),mask_recovered75=hit))
        for g,(hit,total) in counts.items():row['r75_'+g]=hit/max(total,1);row['gt_'+g]=total
        pairhit=pairtotal=0
        for iid in ids:
            anns=[q for q in gt.imgToAnns[iid] if not q.get('iscrowd',0)]
            for k,a in enumerate(anns):
                for b in anns[k+1:]:
                    if a['category_id']==b['category_id'] and box_iou(a['bbox'],b['bbox'])>.05:
                        pairtotal+=1;pairhit+=recovered.get(a['id'],False) and recovered.get(b['id'],False)
        row['pair_recovery75']=pairhit/max(pairtotal,1);row['pairs']=pairtotal
        statistics.append(row);print(json.dumps(row),flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        dt=gt.loadRes(boxpred);ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
    write_csv(out/'summary.csv',statistics);write_csv(out/'gt_recovery.csv',allgt)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',images=len(ids),arms=len(arms),box_ap_common=float(ev.stats[0]),initial_replay_pixel_xor=parity,seconds=time.monotonic()-start,script_sha256=sha(__file__)))

if __name__=='__main__':main()

