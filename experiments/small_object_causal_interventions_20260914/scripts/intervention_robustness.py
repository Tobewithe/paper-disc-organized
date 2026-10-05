"""Dose response and sham controls for GT-assisted image interventions."""
from __future__ import annotations
import contextlib,csv,io,json,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from pycocotools.coco import COCO
from ultralytics import YOLO
from causal_image_probe import read_csv,write_csv,clean_ring,measure,bootstrap

def luminance_delta(image,target,all_objects,magnitude):
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32);fg=target>0;ring,_=clean_ring(target,all_objects)
    fm=float(lab[...,0][fg].mean());bm=float(lab[...,0][ring].mean()) if ring.any() else fm;sign=1 if fm>=bm else -1
    if abs(fm-bm)<2:sign=1 if fm<127.5 else -1
    return sign*magnitude
def apply(image,alpha,delta):
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32);lab[...,0]=np.clip(lab[...,0]+alpha*delta,0,255);return cv2.cvtColor(lab.astype(np.uint8),cv2.COLOR_LAB2BGR)
def shifted(mask,bbox):
    h,w=mask.shape;x,y,bw,bh=bbox
    options=[(round(1.5*bw),0),(-round(1.5*bw),0),(0,round(1.5*bh)),(0,-round(1.5*bh))]
    best=None
    for dx,dy in options:
        m=np.zeros_like(mask);ys,xs=np.nonzero(mask);xx=xs+dx;yy=ys+dy;ok=(xx>=0)&(xx<w)&(yy>=0)&(yy<h);m[yy[ok],xx[ok]]=1
        score=int(m.sum())-10*int(((m>0)&(mask>0)).sum())
        if best is None or score>best[0]:best=(score,m)
    return best[1]
def neighbor_mask(coco,ann,target,all_objects):
    _,rr=clean_ring(target,all_objects);radius=max(5,rr*2);k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1));zone=cv2.dilate(target,k)>0;m=np.zeros_like(target,np.uint8)
    for oid in coco.getAnnIds(imgIds=[ann['image_id']],iscrowd=None):
        o=coco.anns[oid]
        if oid!=ann['id'] and not o.get('iscrowd',0):m[((coco.annToMask(o)>0)&zone&(target==0))]=1
    return m
def main():
    out=Path(__file__).resolve().parents[1];root=out.parents[1];src=root/'experiments/small_raw_geometry_origin_20260914';shared=root/'shared/coco_clean_20260911';sys.path.insert(0,str(shared));from structure_candidate_trace import TraceCapture
    selected=read_csv(src/'selection.csv')
    with contextlib.redirect_stdout(io.StringIO()):coco=COCO(str(root/'assets/datasets/coco/annotations/instances_val2017.json'))
    images=shared/'local_readout_runtime_20260912/data/images/val2017';model=YOLO(str(root/'assets/models/coco_clean_20260911/yolo26m-seg.pt'));model.model.eval().requires_grad_(False);model.model.model[-1].end2end=False
    cached=read_csv(out/'per_target_intervention.csv');cache={(r['annotation_id'],r['intervention']):r for r in cached};rows=[];start=time.time()
    for num,item in enumerate(selected,1):
        ann=coco.anns[int(item['annotation_id'])];info=coco.imgs[int(item['image_id'])];image=cv2.imread(str(images/info['file_name']));target=coco.annToMask(ann).astype(np.uint8);allobj=np.zeros_like(target)
        for oid in coco.getAnnIds(imgIds=[ann['image_id']],iscrowd=None):
            if not coco.anns[oid].get('iscrowd',0):allobj|=coco.annToMask(coco.anns[oid]).astype(np.uint8)
        base=cache[(item['annotation_id'],'original')];full=cache[(item['annotation_id'],'contrast_increase')]
        rows.extend([{**item,'arm':'original',**{k:base[k] for k in ['best_any_box_iou','best_any_true_score','best_true_box_iou','best_true_score','local_true_score','local_box_iou','local_top1','raw_box50']}},{**item,'arm':'mask_hard_28',**{k:full[k] for k in ['best_any_box_iou','best_any_true_score','best_true_box_iou','best_true_score','local_true_score','local_box_iou','local_top1','raw_box50']}}])
        delta28=luminance_delta(image,target,allobj,28);dist=cv2.distanceTransform(target,cv2.DIST_L2,5);soft=np.clip(dist/4,0,1)
        x,y,bw,bh=ann['bbox'];yy,xx=np.ogrid[:target.shape[0],:target.shape[1]];cx=x+bw/2;cy=y+bh/2;ellipse=(((xx-cx)/max(bw*.75,1))**2+((yy-cy)/max(bh*.75,1))**2<=1).astype(np.float32)
        nm=neighbor_mask(coco,ann,target,allobj);blur=cv2.GaussianBlur(image,(0,0),5);neighbor_blur=image.copy();neighbor_blur[nm>0]=blur[nm>0]
        ring,_=clean_ring(target,allobj);mean=image[ring].mean(0) if ring.any() else image.reshape(-1,3).mean(0);neighbor_flat=image.copy();neighbor_flat[nm>0]=mean
        variants={
            'mask_hard_7':apply(image,target.astype(float),luminance_delta(image,target,allobj,7)),
            'mask_hard_14':apply(image,target.astype(float),luminance_delta(image,target,allobj,14)),
            'mask_hard_42':apply(image,target.astype(float),luminance_delta(image,target,allobj,42)),
            'mask_soft_28':apply(image,soft,delta28),
            'ellipse_roi_28':apply(image,ellipse,delta28),
            'shifted_mask_28':apply(image,shifted(target,ann['bbox']).astype(float),delta28),
            'neighbor_blur':neighbor_blur,
            'neighbor_flat':neighbor_flat,
        }
        for arm,edited in variants.items():rows.append({**item,'arm':arm,**measure(model,TraceCapture,edited,ann,(info['height'],info['width']))})
        if num%32==0 or num==len(selected):write_csv(out/'robustness_per_target.csv',rows);print(f'[{num}/{len(selected)}] {time.time()-start:.1f}s',flush=True)
    by={(r['annotation_id'],r['arm']):r for r in rows};summary=[];arms=sorted({r['arm'] for r in rows if r['arm']!='original'})
    for cohort in sorted({r['cohort'] for r in rows}):
        ids=sorted({r['annotation_id'] for r in rows if r['cohort']==cohort})
        for arm in arms:
            rec={'cohort':cohort,'arm':arm,'n':len(ids),'box50_recovered':''}
            for metric in ['best_any_box_iou','best_any_true_score','local_true_score','local_box_iou','raw_box50']:
                d=[float(by[(aid,arm)][metric])-float(by[(aid,'original')][metric]) for aid in ids];mean,lo,hi=bootstrap(d);rec[metric+'_delta']=mean;rec[metric+'_ci_low']=lo;rec[metric+'_ci_high']=hi
            if cohort=='raw_geometry_small':rec['box50_recovered']=sum(int(float(by[(aid,arm)]['raw_box50'])) for aid in ids)
            summary.append(rec)
    write_csv(out/'robustness_summary.csv',summary);(out/'ROBUSTNESS_COMPLETE.json').write_text(json.dumps({'status':'complete','targets':len(selected),'arms':['original']+arms,'elapsed_s':time.time()-start},indent=2),encoding='utf-8')
if __name__=='__main__':main()
