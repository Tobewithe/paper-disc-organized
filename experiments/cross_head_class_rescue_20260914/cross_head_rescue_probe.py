"""Can YOLO26's semantic branch repair class-head failures on geometrically valid raw candidates?"""

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

def read(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(path,rows):
    if not rows:return
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def miou(boxes,target):
    lt=np.maximum(boxes[:,:2],target[:2]);rb=np.minimum(boxes[:,2:],target[2:]);inter=np.maximum(rb-lt,0).prod(1);a=np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(1);b=np.maximum(target[2:]-target[:2],0).prod();return inter/np.maximum(a+b-inter,1e-12)
def rank(scores,label):return int(1+(scores>scores[label]).sum())

def choose(rows,n,seed):
    groups={"correct_class_failure":lambda r:r["first_loss"]=="correct_class","score_failure":lambda r:r["first_loss"]=="score_0001","matched_control":lambda r:r["first_loss"]=="matched"}
    rng=random.Random(seed);used=set();out=[]
    for name,pred in groups.items():
        pools=defaultdict(list)
        for r in rows:
            if pred(r) and int(r["image_id"]) not in used:pools[r["area_bin"]].append(r)
        for p in pools.values():rng.shuffle(p)
        picked=[]
        while len(picked)<n:
            changed=False
            for size in ("small","medium","large"):
                p=pools[size]
                while p and int(p[-1]["image_id"]) in used:p.pop()
                if p and len(picked)<n:r=p.pop();used.add(int(r["image_id"]));picked.append(r);changed=True
            if not changed:break
        if len(picked)<n:raise RuntimeError(f"Only {len(picked)} for {name}")
        for r in picked:out.append({"cohort":name,"image_id":int(r["image_id"]),"annotation_id":int(r["annotation_id"]),"category_id":int(r["category_id"]),"area_bin":r["area_bin"],"density_e4":r["density_e4"]})
    return out

def pool_semantic(logits,box,inner=1.0):
    x0,y0,x1,y1=map(float,box);cx=(x0+x1)/2;cy=(y0+y1)/2;hw=(x1-x0)*inner/2;hh=(y1-y0)*inner/2
    x0=max(0,int(np.floor((cx-hw)/8)));x1=min(80,int(np.ceil((cx+hw)/8)));y0=max(0,int(np.floor((cy-hh)/8)));y1=min(80,int(np.ceil((cy+hh)/8)))
    if x1<=x0 or y1<=y0:return logits[:,max(0,min(79,y0)),max(0,min(79,x0))]
    return logits[:,y0:y1,x0:x1].mean((1,2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--per-cohort",type=int,default=192);ap.add_argument("--seed",type=int,default=0);ap.add_argument("--limit",type=int,default=0);a=ap.parse_args()
    out=Path(__file__).resolve().parent;root=out.parents[1];shared=root/"shared/coco_clean_20260911";sys.path.insert(0,str(shared));from structure_candidate_trace import TraceCapture
    chosen=choose(read(root/"diagnostics/no_final_slot_lineage_20260914/per_gt.csv"),a.per_cohort,a.seed)
    if a.limit:chosen=chosen[:a.limit]
    write(out/"selection.csv",chosen)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(root/"assets/datasets/coco/annotations/instances_val2017.json"))
    images=shared/"local_readout_runtime_20260912/data/images/val2017";model=YOLO(str(root/"assets/models/coco_clean_20260911/yolo26m-seg.pt"));model.model.eval().requires_grad_(False);head=model.model.model[-1];head.end2end=False;head.proto.fuse=lambda:None
    capsem={}
    def hook(module,args):
        x=args[0];feat=x[0]
        for i,r in enumerate(module.feat_refine):feat=feat+torch.nn.functional.interpolate(r(x[i+1]),size=feat.shape[2:],mode="nearest")
        capsem["logits"]=module.semseg(feat).detach().float().cpu()[0].numpy()
    handle=head.proto.register_forward_pre_hook(hook);rows=[];start=time.time();alphas=(0.25,0.5,1.0,2.0,4.0)
    for number,item in enumerate(chosen,1):
        ann=gt.anns[item["annotation_id"]];capsem.clear()
        with torch.inference_mode():model.predict(str(images/gt.imgs[item["image_id"]]["file_name"]),predictor=TraceCapture,imgsz=640,rect=False,conf=.001,iou=.7,max_det=300,retina_masks=False,device=0,verbose=False,end2end=False)
        raw=model.predictor.dense[0];rawboxes_input=ops.xywh2xyxy(raw[:4].T.clone());rawboxes_orig=ops.scale_boxes((640,640),rawboxes_input.clone(),model.predictor.capture["shape"]).cpu().numpy();scores=raw[4:84].T.float().cpu().numpy()
        x,y,w,h=ann["bbox"];target=np.array([x,y,x+w,y+h]);ious=miou(rawboxes_orig,target);label=COCO80.index(item["category_id"])
        geometry=np.flatnonzero(ious>=.5)
        if not len(geometry):
            continue
        raw_argmax=scores.argmax(1)
        if item["cohort"]=="correct_class_failure":
            # The failure definition guarantees no geometry-valid candidate has the correct argmax.
            # Select the candidate with the strongest true-class evidence to estimate correctability.
            best=int(geometry[np.argmax(scores[geometry,label])])
        else:
            correct=geometry[raw_argmax[geometry]==label]
            if not len(correct):
                continue
            best=int(correct[np.argmax(scores[correct,label])])
        det=scores[best];sem_full=pool_semantic(capsem["logits"],rawboxes_input[best].cpu().numpy(),1.0);sem_center=pool_semantic(capsem["logits"],rawboxes_input[best].cpu().numpy(),0.5)
        logdet=np.log(np.clip(det,1e-9,1.0));record={**item,"best_raw_index":best,"box_iou":float(ious[best]),"raw_top_class":COCO80[int(det.argmax())],"raw_true_score":float(det[label]),"raw_true_rank":rank(det,label),"semantic_full_true_rank":rank(sem_full,label),"semantic_center_true_rank":rank(sem_center,label),"semantic_full_margin_vs_rawtop":float(sem_full[label]-sem_full[int(det.argmax())]),"semantic_center_margin_vs_rawtop":float(sem_center[label]-sem_center[int(det.argmax())])}
        for alpha in alphas:
            for name,sem in (("full",sem_full),("center",sem_center)):
                fused=logdet+alpha*sem;record[f"fused_{name}_a{alpha:g}_rank"]=rank(fused,label);record[f"fused_{name}_a{alpha:g}_correct"]=int(fused.argmax()==label)
        rows.append(record)
        if number%48==0 or number==len(chosen):write(out/"per_target.csv",rows);(out/"progress.json").write_text(json.dumps({"done":number,"total":len(chosen),"elapsed_s":time.time()-start},indent=2),encoding="utf-8");print(f"[{number}/{len(chosen)}] {time.time()-start:.1f}s",flush=True)
    handle.remove();summary=[]
    for cohort in sorted({r["cohort"] for r in rows}):
        rr=[r for r in rows if r["cohort"]==cohort]
        base={"cohort":cohort,"n":len(rr),"box_iou_mean":np.mean([r["box_iou"] for r in rr]),"raw_top1":np.mean([r["raw_true_rank"]==1 for r in rr]),"semantic_full_top1":np.mean([r["semantic_full_true_rank"]==1 for r in rr]),"semantic_center_top1":np.mean([r["semantic_center_true_rank"]==1 for r in rr]),"semantic_full_better_rank":np.mean([r["semantic_full_true_rank"]<r["raw_true_rank"] for r in rr]),"semantic_center_better_rank":np.mean([r["semantic_center_true_rank"]<r["raw_true_rank"] for r in rr])}
        for alpha in alphas:
            for name in ("full","center"):base[f"fused_{name}_a{alpha:g}_top1"]=np.mean([r[f"fused_{name}_a{alpha:g}_correct"] for r in rr])
        summary.append(base)
    write(out/"summary.csv",summary);(out/"COMPLETE.json").write_text(json.dumps({"status":"complete","targets":len(rows),"ultralytics":"8.4.100","end2end":False,"gt_role":"select best geometric raw candidate and measure true-class rank only","elapsed_s":time.time()-start},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
