"""Separate class evidence from box regression failure for small raw-geometry misses."""
from __future__ import annotations
import argparse,contextlib,csv,io,json,random,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.utils import ops

COCO80=[1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
def read(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(p,rows):
    if not rows:return
    with Path(p).open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def iou(boxes,target):
    lt=np.maximum(boxes[:,:2],target[:2]);rb=np.minimum(boxes[:,2:],target[2:]);inter=np.maximum(rb-lt,0).prod(1);a=np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(1);b=np.maximum(target[2:]-target[:2],0).prod();return inter/np.maximum(a+b-inter,1e-12)
def transform_box(b,shape):
    h,w=shape;r=min(640/h,640/w);nw,nh=round(w*r),round(h*r);left,top=round((640-nw)/2-.1),round((640-nh)/2-.1);x,y,bw,bh=b;return np.array([x*r+left,y*r+top,(x+bw)*r+left,(y+bh)*r+top])
def grid():
    points=[];strides=[];levels=[]
    for level,(side,stride) in enumerate(((80,8),(40,16),(20,32))):
        yy,xx=np.meshgrid(np.arange(side)+.5,np.arange(side)+.5,indexing="ij");points.append(np.c_[xx.ravel()*stride,yy.ravel()*stride]);strides.extend([stride]*(side*side));levels.extend([level]*(side*side))
    return np.concatenate(points),np.asarray(strides),np.asarray(levels)

def choose(rows,n,seed):
    failures=[r for r in rows if r["first_loss"]=="raw_geometry" and r["area_bin"]=="small"]
    controls=[r for r in rows if r["first_loss"]=="matched" and r["area_bin"]=="small"]
    rng=random.Random(seed);rng.shuffle(failures);used=set();chosen=[]
    pools=defaultdict(list)
    for r in controls:pools[int(r["category_id"])].append(r)
    pair_id=0
    for f in failures:
        if int(f["image_id"]) in used:continue
        candidates=[r for r in pools[int(f["category_id"])] if int(r["image_id"]) not in used and int(r["image_id"])!=int(f["image_id"])]
        if not candidates:continue
        target_log=np.log(max(float(f["area"]),1e-9));r=min(candidates,key=lambda q:abs(np.log(max(float(q["area"]),1e-9))-target_log))
        pair_id+=1;used.add(int(f["image_id"]));used.add(int(r["image_id"]));chosen.append(("raw_geometry_small",f,pair_id));chosen.append(("matched_small_control",r,pair_id))
        if pair_id==n:break
    if pair_id<n:raise RuntimeError(f"Only {pair_id} exact-category nearest-area pairs for requested {n}")
    return [{"cohort":c,"pair_id":pid,"image_id":int(r["image_id"]),"annotation_id":int(r["annotation_id"]),"category_id":int(r["category_id"]),"area_bin":r["area_bin"],"coco_area":float(r["area"]),"density_e4":r["density_e4"]} for c,r,pid in chosen]

def boot(x,seed,reps=2000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);m=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(reps)]);q=np.quantile(m,[.025,.975]);return float(x.mean()),float(q[0]),float(q[1])

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--per-cohort",type=int,default=384);ap.add_argument("--seed",type=int,default=0);ap.add_argument("--limit",type=int,default=0);a=ap.parse_args();out=Path(__file__).resolve().parents[1];root=out.parents[1];shared=root/"shared/coco_clean_20260911";sys.path.insert(0,str(shared));from structure_candidate_trace import TraceCapture
    chosen=choose(read(root/"experiments/no_final_slot_lineage_20260914/per_gt.csv"),a.per_cohort,a.seed)
    if a.limit:chosen=chosen[:a.limit]
    write(out/"selection.csv",chosen)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(root/"assets/datasets/coco/annotations/instances_val2017.json"))
    model=YOLO(str(root/"assets/models/coco_clean_20260911/yolo26m-seg.pt"));model.model.eval().requires_grad_(False);model.model.model[-1].end2end=False;images=shared/"local_readout_runtime_20260912/data/images/val2017";points,strides,levels=grid();rows=[];start=time.time()
    for num,item in enumerate(chosen,1):
        ann=gt.anns[item["annotation_id"]];shape=(gt.imgs[item["image_id"]]["height"],gt.imgs[item["image_id"]]["width"]);target=transform_box(ann["bbox"],shape);label=COCO80.index(item["category_id"])
        with torch.inference_mode():model.predict(str(images/gt.imgs[item["image_id"]]["file_name"]),predictor=TraceCapture,imgsz=640,rect=False,conf=.001,iou=.7,max_det=300,retina_masks=False,device=0,verbose=False,end2end=False)
        raw=model.predictor.dense[0];boxes=ops.xywh2xyxy(raw[:4].T).float().cpu().numpy();scores=raw[4:84].T.float().cpu().numpy();ious=iou(boxes,target);cx=(target[0]+target[2])/2;cy=(target[1]+target[3])/2;tw=target[2]-target[0];th=target[3]-target[1]
        local=(points[:,0]>=target[0]-strides/2)&(points[:,0]<=target[2]+strides/2)&(points[:,1]>=target[1]-strides/2)&(points[:,1]<=target[3]+strides/2)&(levels==0)
        ids=np.flatnonzero(local);best_score=int(ids[np.argmax(scores[ids,label])]);nearest=int(np.argmin(((points-np.array([cx,cy]))**2).sum(1)+(levels!=0)*1e9));best_iou=int(np.argmax(ious));pb=boxes[best_score];pc=np.array([(pb[0]+pb[2])/2,(pb[1]+pb[3])/2]);pw=max(pb[2]-pb[0],1e-6);ph=max(pb[3]-pb[1],1e-6)
        rows.append({**item,"gt_input_w":tw,"gt_input_h":th,"p3_local_cells":len(ids),"local_true_score_max":float(scores[best_score,label]),"local_top1_correct":int(scores[best_score].argmax()==label),"local_box_iou":float(ious[best_score]),"local_box_center_error_norm":float(np.linalg.norm(pc-[cx,cy])/max(np.hypot(tw,th),1e-6)),"local_log_width_error":float(abs(np.log(pw/max(tw,1e-6)))),"local_log_height_error":float(abs(np.log(ph/max(th,1e-6)))),"nearest_true_score":float(scores[nearest,label]),"nearest_top1_correct":int(scores[nearest].argmax()==label),"nearest_box_iou":float(ious[nearest]),"best_any_box_iou":float(ious[best_iou]),"best_any_source_level":int(levels[best_iou]),"best_any_source_distance_norm":float(np.linalg.norm(points[best_iou]-[cx,cy])/max(np.hypot(tw,th),1e-6)),"best_true_score_global":float(scores[:,label].max())})
        if num%64==0 or num==len(chosen):write(out/"per_target.csv",rows);(out/"progress.json").write_text(json.dumps({"done":num,"total":len(chosen),"elapsed_s":time.time()-start},indent=2),encoding="utf-8");print(f"[{num}/{len(chosen)}] {time.time()-start:.1f}s",flush=True)
    metrics=[k for k in rows[0] if k not in chosen[0] and k not in ("cohort",)];summary=[]
    for cohort in sorted({r["cohort"] for r in rows}):
        rr=[r for r in rows if r["cohort"]==cohort];o={"cohort":cohort,"n":len(rr)}
        for m in metrics:
            mean,lo,hi=boot([float(r[m]) for r in rr],a.seed);o[m+"_mean"]=mean;o[m+"_ci_low"]=lo;o[m+"_ci_high"]=hi
        summary.append(o)
    write(out/"summary.csv",summary);(out/"COMPLETE.json").write_text(json.dumps({"status":"complete","targets":len(rows),"exact_category_matched_controls":True,"ultralytics":"8.4.100","end2end":False,"elapsed_s":time.time()-start},indent=2),encoding="utf-8")
if __name__=="__main__":main()
