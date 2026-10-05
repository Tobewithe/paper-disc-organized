"""Fresh full-COCO dual-branch evaluation plus a frozen diagnostic panel."""
from __future__ import annotations
import argparse, contextlib, gc, gzip, io, json, shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import ops
from pycocotools.coco import COCO
from branch_evaluation import CocoValidator, TracePredictor
import coco_metrics as cm
import target_metrics as tm
from recording import atomic_json, now

def targeted(a,cfg,gt,branch):
    rows=[];selection=tm.read_csv(Path(cfg['selection']))
    assert len(selection)==438
    model=YOLO(str(a.weight))
    for n,item in enumerate(selection):
        ann=gt.anns[int(item['annotation_id'])];info=gt.imgs[int(ann['image_id'])]
        path=Path(cfg['coco_root'])/'images/val2017'/info['file_name']
        im=cv2.imread(str(path));assert im is not None,str(path)
        own,same,other,context=tm.image_context(gt,ann,im)
        result=model.predict(str(path),predictor=TracePredictor,imgsz=640,rect=False,conf=.001,iou=.7,
                             max_det=300,retina_masks=True,device=0,verbose=False,end2end=(branch=='one2one'))[0]
        assert bool(model.predictor.model.end2end)==(branch=='one2one')
        predictor=model.predictor
        boxes=ops.scale_boxes(predictor.input_hw,predictor.raw_boxes.clone(),im.shape[:2]).float().cpu().numpy()
        scores=predictor.raw_scores.float().cpu().numpy()
        x,y,w,h=ann['bbox'];gb=np.array([x,y,x+w,y+h]);ious=tm.box_iou(boxes,gb)
        k=int(ious.argmax());cls=tm.COCO80.index(ann['category_id']);detail=tm.box_details(boxes[k],gb)
        row={**item,'arm':a.arm,'checkpoint':a.checkpoint,'branch':branch,
             'raw_best_box_iou':float(ious[k]),'raw_p3_best_box_iou':float(ious[:predictor.p3_count].max()),
             'raw_center_error_norm':detail['box_center_error_norm'],'raw_correct_class_at_best':int(scores[k].argmax()==cls),
             'raw_same_class_box50':int(np.any((ious>=.5)&(scores.argmax(1)==cls)))}
        selected=None;pm=np.zeros_like(own);fb=np.zeros(4);confidence=0.
        if result.boxes is not None and len(result.boxes):
            ids=np.flatnonzero(result.boxes.cls.int().cpu().numpy()==cls)
            if len(ids):
                final=result.boxes.xyxy.float().cpu().numpy();selected=int(ids[tm.box_iou(final[ids],gb).argmax()])
                fb=final[selected];confidence=float(result.boxes.conf[selected])
                if result.masks is not None:pm=result.masks.data[selected].bool().cpu().numpy()
        row.update(final_candidate_exists=int(selected is not None),final_confidence=confidence,
                   final_box_iou=float(tm.box_iou(fb[None],gb)[0]))
        row.update(tm.mask_details(pm,own,same,other))
        for t in (.5,.75):
            row[f'raw_box{int(t*100)}']=int(ious[k]>=t)
            row[f'final_box{int(t*100)}']=int(row['final_box_iou']>=t)
            row[f'final_mask{int(t*100)}']=int(row['mask_iou']>=t)
        rows.append(row)
        if (n+1)%64==0:print('TARGET',a.arm,a.checkpoint,branch,n+1,flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    return rows

def main():
    p=argparse.ArgumentParser()
    for k in ['study','out','weight']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--arm',required=True);p.add_argument('--checkpoint',choices=['last','best'],required=True)
    a=p.parse_args();cfg=json.loads((a.study/'protocol.json').read_text())
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(Path(cfg['coco_root'])/'annotations/instances_val2017.json'))
    assert len(gt.imgs)==5000 and sorted(gt.cats)==tm.COCO80
    metrics=[];categories=[];targets=[]
    for branch in cfg['evaluation_branches']:
        model=YOLO(str(a.weight))
        result=model.val(validator=CocoValidator,data=str(Path(cfg['coco_root'])/'coco_full.yaml'),split='val',
                         imgsz=640,batch=8,workers=8,device=0,conf=.001,iou=.7,max_det=300,
                         end2end=(branch=='one2one'),save_json=True,plots=False,project=str(a.out/'exports'),
                         name=branch,exist_ok=False,verbose=False)
        export=Path(result.save_dir)/'predictions.json';predictions=json.loads(export.read_text())
        del model;gc.collect();torch.cuda.empty_cache()
        for task in ['bbox','segm']:
            processed=cm.task_predictions(predictions,{k:k for k in gt.cats},task,gt)
            row,percat=cm.evaluate(gt,processed,a.arm,task)
            row.update(branch=branch,checkpoint=a.checkpoint)
            for r in percat:r.update(branch=branch,checkpoint=a.checkpoint)
            metrics.append(row);categories.extend(percat)
            tm.write_csv(a.out/'official_metrics.csv',metrics)
            tm.write_csv(a.out/'official_per_category.csv',categories)
            print('OFFICIAL',json.dumps(row),flush=True)
            del processed
        del predictions;gc.collect()
        # Retain lossless raw exports while limiting disk use across 12 predictions.
        assert export.resolve().is_relative_to(a.out.resolve())
        with export.open('rb') as src,gzip.open(export.with_suffix('.json.gz'),'wb',compresslevel=1) as dst:
            shutil.copyfileobj(src,dst)
        export.unlink()
        targets.extend(targeted(a,cfg,gt,branch));tm.write_csv(a.out/'targeted_per_instance.csv',targets)
    atomic_json(a.out/'COMPLETE.json',dict(status='complete',completed_at=now(),arm=a.arm,checkpoint=a.checkpoint,
                weights=str(a.weight),full_images=5000,panel_targets_per_branch=438,
                branches=cfg['evaluation_branches'],category_id_space='coco',
                official_protocol='original COCOeval, all 5000 images, maxDets [1,10,100], RLE mask areas',
                target_protocol='GT-associated best same-class box; not unique COCO recall'))

if __name__=='__main__':main()
