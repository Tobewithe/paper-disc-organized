"""Paired 640/1280 intervention on exact-category nearest-area small-object pairs."""
from __future__ import annotations
import contextlib,csv,io,json,sys,time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from pycocotools.coco import COCO
from ultralytics import YOLO
def read(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(p,rows):
    if not rows:return
    with Path(p).open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def boot(x,seed=0,reps=3000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);m=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(reps)]);q=np.quantile(m,[.025,.975]);return float(x.mean()),float(q[0]),float(q[1])
def main():
    out=Path(__file__).resolve().parents[1];root=out.parents[1];roi=root/"experiments/failure_object_roi_probe_20260914/scripts";sys.path.insert(0,str(roi));from roi_reinference_probe import infer,measure
    shared=root/"shared/coco_clean_20260911";images=shared/"local_readout_runtime_20260912/data/images/val2017";selected=read(out/"selection.csv")
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(root/"assets/datasets/coco/annotations/instances_val2017.json"))
    model=YOLO(str(root/"assets/models/coco_clean_20260911/yolo26m-seg.pt"));model.model.model[-1].end2end=False;rows=[];start=time.time()
    for num,item in enumerate(selected,1):
        iid=int(item["image_id"]);aid=int(item["annotation_id"]);cat=int(item["category_id"]);ann=gt.anns[aid];info=gt.imgs[iid];image=cv2.imread(str(images/info["file_name"]));target=gt.annToMask(ann).astype(bool);same=np.zeros_like(target);others=np.zeros_like(target)
        for oid in gt.getAnnIds(imgIds=[iid],iscrowd=None):
            if oid==aid or gt.anns[oid].get("iscrowd",0):continue
            om=gt.annToMask(gt.anns[oid]).astype(bool);others|=om
            if gt.anns[oid]["category_id"]==cat:same|=om
        x,y,w,h=ann["bbox"];box=np.array([x,y,x+w,y+h])
        for imgsz in (640,1280):
            result=infer(model,image,imgsz);m=measure(result,cat,box,target,same,others,(0,0));rows.append({**item,"imgsz":imgsz,**m})
        if num%64==0 or num==len(selected):write(out/"resolution_per_target.csv",rows);(out/"resolution_progress.json").write_text(json.dumps({"done":num,"total":len(selected),"elapsed_s":time.time()-start},indent=2),encoding="utf-8");print(f"[{num}/{len(selected)}] {time.time()-start:.1f}s",flush=True)
    base={(r["annotation_id"]):r for r in rows if r["imgsz"]==640};summary=[]
    for cohort in sorted({r["cohort"] for r in rows}):
        rr=[r for r in rows if r["cohort"]==cohort and r["imgsz"]==1280];d=[float(r["mask_iou"])-float(base[r["annotation_id"]]["mask_iou"]) for r in rr];bd=[float(r["box_iou"])-float(base[r["annotation_id"]]["box_iou"]) for r in rr];dm,dl,dh=boot(d);bm,bl,bh=boot(bd);summary.append({"cohort":cohort,"n":len(rr),"delta_mask_iou":dm,"delta_mask_iou_ci_low":dl,"delta_mask_iou_ci_high":dh,"delta_box_iou":bm,"delta_box_iou_ci_low":bl,"delta_box_iou_ci_high":bh,"box50_recovered":sum(float(base[r["annotation_id"]]["box_iou"])<.5 and float(r["box_iou"])>=.5 for r in rr),"box75_recovered":sum(float(base[r["annotation_id"]]["box_iou"])<.75 and float(r["box_iou"])>=.75 for r in rr),"mask75_recovered":sum(float(base[r["annotation_id"]]["mask_iou"])<.75 and float(r["mask_iou"])>=.75 for r in rr),"box50_lost":sum(float(base[r["annotation_id"]]["box_iou"])>=.5 and float(r["box_iou"])<.5 for r in rr),"mask75_lost":sum(float(base[r["annotation_id"]]["mask_iou"])>=.75 and float(r["mask_iou"])<.75 for r in rr)})
    write(out/"resolution_summary.csv",summary);(out/"RESOLUTION_COMPLETE.json").write_text(json.dumps({"status":"complete","targets":len(selected),"arms":[640,1280],"gt_role":"paired target measurement only","elapsed_s":time.time()-start},indent=2),encoding="utf-8")
if __name__=="__main__":main()
