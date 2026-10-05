"""Evaluate two existing checkpoints on both branches; never update weights."""
from __future__ import annotations
import argparse, contextlib, csv, gc, io, json, time
from pathlib import Path
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import ops
from ultralytics.utils.tal import make_anchors
import coco_metrics as cm
import target_metrics as tm

class CocoValidator(SegmentationValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        assert bool(self.end2end) == bool(self.args.end2end), 'Requested branch did not take effect'
        self.class_map = tm.COCO80
        # Standalone pycocotools below, without validator auto-download/eval.
        self.is_coco = False

class TracePredictor(SegmentationPredictor):
    def postprocess(self, preds, img, orig_imgs):
        raw = preds[1]
        if 'one2one' in raw:
            raw = raw['one2one']
        head = self.model.model.model[-1]
        anchors, strides = make_anchors(raw['feats'], head.stride, .5)
        self.raw_boxes = (head.decode_bboxes(head.dfl(raw['boxes']), anchors.T.unsqueeze(0), xywh=False)*strides.T)[0].T.detach()
        self.raw_scores = raw['scores'][0].T.sigmoid().detach()
        self.p3_count = raw['feats'][0].shape[-2]*raw['feats'][0].shape[-1]
        self.input_hw = tuple(img.shape[-2:])
        return super().postprocess(preds,img,orig_imgs)

def targeted(args, gt):
    selection = tm.read_csv(args.selection)
    records=[]
    for branch in ('one2many','one2one'):
        for arm, checkpoint in [('baseline',args.baseline),('method',args.method)]:
            model=YOLO(str(checkpoint))
            for number,item in enumerate(selection):
                ann=gt.anns[int(item['annotation_id'])];info=gt.imgs[int(ann['image_id'])]
                path=args.images/info['file_name']
                if not path.is_file():continue
                if args.limit and number>=args.limit:break
                im=cv2.imread(str(path)); own,same,other,context=tm.image_context(gt,ann,im)
                result=model.predict(str(path),predictor=TracePredictor,imgsz=640,rect=False,conf=.001,iou=.7,max_det=300,retina_masks=True,device=0,verbose=False,end2end=(branch=='one2one'))[0]
                assert bool(model.predictor.model.end2end)==(branch=='one2one')
                pred=model.predictor
                boxes=ops.scale_boxes(pred.input_hw,pred.raw_boxes.clone(),im.shape[:2]).float().cpu().numpy()
                scores=pred.raw_scores.float().cpu().numpy()
                x,y,w,h=ann['bbox'];gb=np.array([x,y,x+w,y+h])
                ious=tm.box_iou(boxes,gb); k=int(ious.argmax());cls=tm.COCO80.index(ann['category_id'])
                detail=tm.box_details(boxes[k],gb)
                row={**item,'branch':branch,'arm':arm,'raw_best_box_iou':float(ious[k]),'raw_p3_best_box_iou':float(ious[:pred.p3_count].max()),'raw_center_error_norm':detail['box_center_error_norm'],'raw_correct_class_at_best':int(scores[k].argmax()==cls),'raw_same_class_box50':int(np.any((ious>=.5)&(scores.argmax(1)==cls)))}
                selected=None;pm=np.zeros_like(own);fb=np.zeros(4);conf=0.
                if result.boxes is not None and len(result.boxes):
                    ids=np.flatnonzero(result.boxes.cls.int().cpu().numpy()==cls)
                    if len(ids):
                        final_boxes=result.boxes.xyxy.float().cpu().numpy();selected=int(ids[tm.box_iou(final_boxes[ids],gb).argmax()]);fb=final_boxes[selected];conf=float(result.boxes.conf[selected])
                        if result.masks is not None:pm=result.masks.data[selected].bool().cpu().numpy()
                row.update(final_candidate_exists=int(selected is not None),final_confidence=conf,final_box_iou=float(tm.box_iou(fb[None],gb)[0]))
                row.update(tm.mask_details(pm,own,same,other))
                for t in (.5,.75):
                    row[f'raw_box{int(t*100)}']=int(ious[k]>=t)
                    row[f'final_box{int(t*100)}']=int(row['final_box_iou']>=t)
                    row[f'final_mask{int(t*100)}']=int(row['mask_iou']>=t)
                records.append(row)
                if (number+1)%64==0:print('TARGET',branch,arm,number+1,flush=True)
            tm.write_csv(args.out/'targeted_per_instance.csv',records)
            del model;gc.collect();torch.cuda.empty_cache()
    keys=['raw_best_box_iou','raw_p3_best_box_iou','raw_center_error_norm','raw_same_class_box50','raw_correct_class_at_best','final_box_iou','mask_iou','target_coverage','prediction_purity','boundary_f1','final_box50','final_mask50','final_mask75','same_neighbor_leak_pred','background_leak_pred']
    summaries=[]
    for branch in ('one2many','one2one'):
        for cohort in sorted({r['cohort'] for r in records}):
            by={(r['annotation_id'],r['arm']):r for r in records if r['branch']==branch and r['cohort']==cohort}
            ids=sorted({a for a,arm in by})
            for key in keys:
                base=np.array([by[(a,'baseline')][key] for a in ids],float);method=np.array([by[(a,'method')][key] for a in ids],float)
                delta,lo,hi=tm.bootstrap((method-base).tolist(),915)
                summaries.append(dict(branch=branch,cohort=cohort,metric=key,n=len(ids),baseline=float(base.mean()),method=float(method.mean()),delta=delta,ci_low=lo,ci_high=hi))
    tm.write_csv(args.out/'targeted_summary.csv',summaries)

def full(args,gt):
    metrics=[];cats=[]
    for branch in ('one2many','one2one'):
        for arm,checkpoint in [('baseline',args.baseline),('method',args.method)]:
            if branch=='one2one':
                folder='baseline_s0' if arm=='baseline' else 'cfp3r_s0'
                predictions=json.loads((args.prior_exports/folder/'predictions.json').read_text())
                mapping={i+1:k for i,k in enumerate(tm.COCO80)}
            else:
                model=YOLO(str(checkpoint))
                result=model.val(validator=CocoValidator,data=str(args.data),split='val',imgsz=640,batch=8,workers=8,device=0,conf=.001,iou=.7,max_det=300,end2end=False,save_json=True,plots=False,project=str(args.out/'exports'),name=arm,exist_ok=False,verbose=False)
                predictions=json.loads((Path(result.save_dir)/'predictions.json').read_text());mapping={k:k for k in gt.cats}
                del model;gc.collect();torch.cuda.empty_cache()
            for task in ('bbox','segm'):
                processed=cm.task_predictions(predictions,mapping,task,gt)
                row,percat=cm.evaluate(gt,processed,arm,task)
                row['branch']=branch
                for r in percat:r['branch']=branch
                metrics.append(row);cats.extend(percat)
                tm.write_csv(args.out/'official_metrics.csv',metrics)
                print('OFFICIAL',json.dumps(row),flush=True)
                del processed
            del predictions;gc.collect()
    tm.write_csv(args.out/'official_per_category.csv',cats)

def main():
    p=argparse.ArgumentParser()
    for key in ('data','annotation','images','selection','baseline','method','prior-exports','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--target-only',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(args.annotation))
    targeted(args,gt)
    if not args.target_only:full(args,gt)
    (args.out/'COMPLETE.json').write_text(json.dumps(dict(status='complete',target_limit=args.limit,full_images=0 if args.target_only else len(gt.imgs),branches=['one2many','one2one'],target_matching='GT-associated best same-class box, not unique COCO matching',official_protocol='same native val settings; one2one prior exports reused with explicit ID conversion; one2many branch asserted',training=False),indent=2))

if __name__=='__main__':main()
