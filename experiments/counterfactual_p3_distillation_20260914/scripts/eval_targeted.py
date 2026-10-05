"""Paired target-level evaluation on the frozen diagnostic cohort."""
from __future__ import annotations
import contextlib,csv,io,json,sys,time
from pathlib import Path
import numpy as np,torch
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import ops

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
SHARED=ROOT/'shared/coco_clean_20260911';sys.path.insert(0,str(SHARED))
COCO80=[1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
class EvalTrace(SegmentationPredictor):
    """Capture dense boxes while leaving 8.4.100 official mask construction untouched."""
    def postprocess(self,preds,img,orig_imgs):
        raw=preds[0][0] if isinstance(preds[0],tuple) else preds[0]
        self.dense=raw.detach().clone()
        return super().postprocess(preds,img,orig_imgs)
    def construct_result(self,pred,img,orig_img,img_path,proto):
        result=super().construct_result(pred,img,orig_img,img_path,proto)
        self.capture={'shape':tuple(orig_img.shape[:2]),'input_shape':tuple(img.shape[2:])}
        return result
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    fields=[]
    for r in rows:fields += [k for k in r if k not in fields]
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def iou(boxes,target):
    lt=np.maximum(boxes[:,:2],target[:2]);rb=np.minimum(boxes[:,2:],target[2:]);inter=np.maximum(rb-lt,0).prod(1);a=np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(1);b=np.maximum(target[2:]-target[:2],0).prod();return inter/np.maximum(a+b-inter,1e-12)
def boot(x,seed=0,reps=3000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);m=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(reps)]);q=np.quantile(m,[.025,.975]);return float(x.mean()),float(q[0]),float(q[1])
def main():
    weights={'pretrained':ROOT/'assets/models/coco_clean_20260911/yolo26m-seg.pt','baseline':OUT/'runs/baseline_s0/weights/best.pt','cfp3':OUT/'runs/cfp3_s0/weights/best.pt','cfp3r':OUT/'runs/cfp3r_s0/weights/best.pt'}
    if not all(p.exists() for p in weights.values()):raise FileNotFoundError(weights)
    chosen=read(ROOT/'experiments/small_raw_geometry_origin_20260914/selection.csv')
    with contextlib.redirect_stdout(io.StringIO()):coco=COCO(str(ROOT/'assets/datasets/coco/annotations/instances_val2017.json'))
    images=SHARED/'local_readout_runtime_20260912/data/images/val2017';rows=[];start=time.time()
    for arm,path in weights.items():
        model=YOLO(str(path));model.model.eval().requires_grad_(False);model.model.model[-1].end2end=False
        for num,item in enumerate(chosen,1):
            ann=coco.anns[int(item['annotation_id'])];info=coco.imgs[int(item['image_id'])];x,y,w,h=ann['bbox'];gtbox=np.array([x,y,x+w,y+h]);gtmask=coco.annToMask(ann).astype(bool)
            with torch.inference_mode():result=model.predict(str(images/info['file_name']),predictor=EvalTrace,imgsz=640,rect=False,conf=.001,iou=.7,max_det=300,retina_masks=True,device=0,verbose=False,end2end=False)[0]
            raw=model.predictor.dense[0];rawbox=ops.xywh2xyxy(raw[:4].T.clone());rawbox=ops.scale_boxes((640,640),rawbox,model.predictor.capture['shape']).float().cpu().numpy();riou=iou(rawbox,gtbox);raw_best=float(riou.max(initial=0))
            if result.boxes is None or not len(result.boxes):fb=fm=0.0
            else:
                boxes=result.boxes.xyxy.float().cpu().numpy();cls=result.boxes.cls.int().cpu().numpy();valid=np.array([0<=c<80 and COCO80[c]==ann['category_id'] for c in cls]);ids=np.flatnonzero(valid)
                if not len(ids):fb=fm=0.0
                else:
                    vals=iou(boxes[ids],gtbox);best=int(ids[int(vals.argmax())]);fb=float(vals.max());pred=result.masks.data[best].bool().cpu().numpy() if result.masks is not None else np.zeros_like(gtmask)
                    if pred.shape!=gtmask.shape:
                        import cv2;pred=cv2.resize(pred.astype(np.uint8),(gtmask.shape[1],gtmask.shape[0]),interpolation=cv2.INTER_NEAREST).astype(bool)
                    fm=float((pred & gtmask).sum()/max((pred | gtmask).sum(),1))
            rows.append({**item,'arm':arm,'raw_best_box_iou':raw_best,'raw_box50':int(raw_best>=.5),'final_box_iou':fb,'final_box50':int(fb>=.5),'final_mask_iou':fm,'final_mask75':int(fm>=.75)})
            if num%96==0:write(OUT/'targeted_per_target.csv',rows);print(f'[{arm} {num}/{len(chosen)}] {time.time()-start:.1f}s',flush=True)
    write(OUT/'targeted_per_target.csv',rows);by={(r['annotation_id'],r['arm']):r for r in rows};summary=[]
    for cohort in sorted({r['cohort'] for r in rows}):
        ids=sorted({r['annotation_id'] for r in rows if r['cohort']==cohort})
        for arm in ['baseline','cfp3','cfp3r']:
            rec={'cohort':cohort,'arm':arm,'n':len(ids)}
            for metric in ['raw_best_box_iou','raw_box50','final_box_iou','final_box50','final_mask_iou','final_mask75']:
                vals=[float(by[(aid,arm)][metric])-float(by[(aid,'baseline')][metric]) for aid in ids] if arm!='baseline' else [float(by[(aid,arm)][metric]) for aid in ids];mean,lo,hi=boot(vals);rec[metric+('_delta' if arm!='baseline' else '_mean')]=mean;rec[metric+'_ci_low']=lo;rec[metric+'_ci_high']=hi
            summary.append(rec)
    write(OUT/'targeted_summary.csv',summary);(OUT/'TARGETED_COMPLETE.json').write_text(json.dumps({'status':'complete','targets':len(chosen),'models':list(weights),'branch':'one-to-many+NMS','elapsed_s':time.time()-start},indent=2),encoding='utf-8')
if __name__=='__main__':main()
