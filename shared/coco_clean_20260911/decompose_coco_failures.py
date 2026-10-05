"""Full original COCO val2017 failure localization, no training or GT prediction.

Uses a fixed official bbox >= .5 assignment for attribution only. This attribution
recall is not official segmentation AP/recall because segmentation assignment can
differ. Oracle corrections below are diagnostic opportunities, not methods.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,contextlib,csv,io,json,time
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.utils import ops
from frozen_mechanism_probe import ROOT,Capture,ownership,write_json,write_csv,sha


def group(ici):
    if ici<=1e-10:return 'zero'
    if ici<=.1+1e-10:return 'mild'
    if ici<=.5+1e-10:return 'medium'
    return 'high'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    with (ROOT/'census/COCO_EVAL_INSTANCE_MANIFEST.csv').open(encoding='utf-8-sig') as f:meta={int(r['annotation_id']):r for r in csv.DictReader(f)}
    ids=sorted(gt.imgs);ids=ids[:a.limit] if a.limit else ids
    write_json(out/'protocol.json',dict(script_sha256=sha(__file__),official_weight_sha256=sha(ROOT/'weights/yolo26m-seg.pt'),gt_sha256=sha(ROOT/'data/annotations/instances_val2017.json'),
        image_ids=ids,training=False,all_gt=True,GT_used_in_prediction=False,assignment='Official COCOeval bbox >= .5, fixed class/score matching, maxDets=100 per category.',
        image_selection='All val2017 IDs in sorted order, no ICI filter.' if not a.limit else 'First IDs only, execution smoke.',
        inference=dict(conf=.001,max_det=300,iou=.7,imgsz=640,half=False,rect=False),
        metric='Exact per-instance integer pixel counts on original annToMask, union all noncrowd masks; exclude crowd pixels for spatial diagnostic.',
        bins='same-class ICI sum bbox intersections/self bbox area: zero, (0,.1], (.1,.5], >.5, tolerance 1e-10.',
        limitation='Fixed box-matched diagnostic IoU and correction opportunities are not official mask AP, causal attribution, or deployable improvements. Some unmatched GT can have mask matches; some assignment changes are possible.'))
    model=YOLO(str(ROOT/'weights/yolo26m-seg.pt'));model.model.eval().requires_grad_(False);rows=[];start=time.monotonic();parity=0
    for n,iid in enumerate(ids,1):
        with torch.inference_mode():
            model.predict(str(ROOT/'data/images/val2017'/gt.imgs[iid]['file_name']),predictor=Capture,imgsz=640,conf=.001,max_det=300,iou=.7,device=0,rect=False,half=False,retina_masks=False,verbose=False)
            cap=model.predictor.capture;p=cap['proto'];coeff=cap['coeff'];boxes=cap['boxes'];shape=cap['shape'];ishape=cap['input_shape']
            binary=ops.process_mask(p,coeff,boxes,ishape,upsample=True) if len(coeff) else torch.empty((0,*ishape),dtype=torch.uint8,device='cuda')
            # GT enters only after network predictions have been finalized.
            mapping=ownership(gt,iid,cap['detections']);anns=gt.imgToAnns[iid];ordinary=[q for q in anns if not q.get('iscrowd',0)]
            raster={q['id']:torch.tensor(gt.annToMask(q).astype(bool),device='cuda') for q in anns}
            crowd=[raster[q['id']] for q in anns if q.get('iscrowd',0)];valid=~torch.stack(crowd).any(0) if crowd else torch.ones(shape,dtype=torch.bool,device='cuda')
            union=torch.stack([raster[q['id']] for q in ordinary]).any(0) if ordinary else torch.zeros(shape,dtype=torch.bool,device='cuda')
            same={cat:torch.stack([raster[q['id']] for q in ordinary if q['category_id']==cat]).any(0) for cat in {q['category_id'] for q in ordinary}}
            decoded={};supports={}
            for first in range(0,len(binary),32):
                mask=ops.scale_masks(binary[first:first+32,None],shape)[:,0]>.5
                supp=ops.crop_mask(torch.ones_like(binary[first:first+32]),boxes[first:first+32]);supp=ops.scale_masks(supp[:,None],shape)[:,0]>.5
                for k in range(len(mask)):decoded[first+k]=mask[k];supports[first+k]=supp[k]
            for q in ordinary:
                aid=q['id'];own=raster[aid]&valid;area=int(own.sum());ici=float(meta[aid]['ici_same']);j=mapping.get(aid)
                row=dict(image_id=iid,annotation_id=aid,category_id=q['category_id'],ici=ici,group=group(ici),coco_area=q['area'],valid_gt_pixels=area,
                    status='no_valid_pixels' if not area else ('unmatched_box' if j is None else 'matched'),prediction_index=-1 if j is None else j,
                    iou=0.,coverage=0.,crop_ceiling=0.,remove_same_neighbor_iou=0.,remove_all_neighbor_iou=0.,remove_background_iou=0.,remove_all_fp_iou=0.,fill_inside_box_iou=0.,perfect_inside_box_iou=0.,
                    true_positive=0,false_negative=area,false_positive=0,same_neighbor_pixels=0,all_neighbor_pixels=0,background_pixels=0,false_negative_inside=0,false_negative_outside=0)
                if area and j is not None:
                    mask=decoded[j]&valid;supp=supports[j]&valid;I=int((mask&own).sum());FN=area-I;FP=int((mask&~own).sum());sn=int((mask&same[q['category_id']]&~own).sum());an=int((mask&union&~own).sum());bg=int((mask&~union).sum());inside=int((own&supp).sum());fnin=int((own&supp&~mask).sum());fnout=int((own&~supp&~mask).sum())
                    assert FP==an+bg and FN==fnin+fnout and I<=inside
                    row.update(iou=I/(area+FP),coverage=I/area,crop_ceiling=inside/area,
                        remove_same_neighbor_iou=I/(area+FP-sn),remove_all_neighbor_iou=I/(area+FP-an),remove_background_iou=I/(area+FP-bg),remove_all_fp_iou=I/area,
                        fill_inside_box_iou=(I+fnin)/(area+FP),perfect_inside_box_iou=inside/area,
                        true_positive=I,false_negative=FN,false_positive=FP,same_neighbor_pixels=sn,all_neighbor_pixels=an,background_pixels=bg,false_negative_inside=fnin,false_negative_outside=fnout)
                rows.append(row)
            del decoded,supports,binary,raster
        if n%100==0:
            progress=dict(completed=n,total=len(ids),targets=len(rows),seconds=round(time.monotonic()-start,1));write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    write_csv(out/'per_instance.csv',rows);summary=[]
    for g in ['all','zero','mild','medium','high']:
        cohort=[r for r in rows if r['valid_gt_pixels'] and (g=='all' or r['group']==g)];matched=[r for r in cohort if r['status']=='matched'];fail=[r for r in matched if r['iou']<.75];den=len(cohort)
        r=dict(group=g,gt=den,matched=len(matched),unmatched=den-len(matched),matched_fail75=len(fail),baseline_success75=sum(q['iou']>=.75 for q in matched),crop_blocked75=sum(q['crop_ceiling']<.75 for q in fail),crop_adequate_fail75=sum(q['crop_ceiling']>=.75 for q in fail))
        for key in ['remove_same_neighbor_iou','remove_all_neighbor_iou','remove_background_iou','remove_all_fp_iou','fill_inside_box_iou','perfect_inside_box_iou']:
            r[key+'_rescued75']=sum(q[key]>=.75 for q in fail)
        for key in ['iou','coverage','crop_ceiling']:
            r['matched_mean_'+key]=float(np.mean([q[key] for q in matched])) if matched else 0.
        summary.append(r);print(json.dumps(r),flush=True)
    write_csv(out/'summary.csv',summary);write_json(out/'COMPLETE.json',dict(status='COMPLETE',images=len(ids),gt=len(rows),seconds=time.monotonic()-start,script_sha256=sha(__file__)))

if __name__=='__main__':main()
