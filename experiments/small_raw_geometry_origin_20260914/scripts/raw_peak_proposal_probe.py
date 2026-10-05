"""Inference-visible ROI proposals from unmatched P3 classification peaks."""
from __future__ import annotations
import contextlib,csv,io,json,sys,time
from pathlib import Path
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO
def read(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(p,rows):
    with Path(p).open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def target_grid(ann,shape):
    h,w=shape;r=min(640/h,640/w);nw,nh=round(w*r),round(h*r);left,top=round((640-nw)/2-.1),round((640-nh)/2-.1);x,y,bw,bh=ann["bbox"];return np.array([(x*r+left)/8,(y*r+top)/8,((x+bw)*r+left)/8,((y+bh)*r+top)/8])
def peaks(heat,centers,k=20):
    q=heat.copy()
    for x,y in centers:
        cv2.circle(q,(int(round(x)),int(round(y))),3,0.0,-1)
    loc=q>=cv2.dilate(q,np.ones((5,5),np.float32))-1e-12;cand=np.argwhere(loc&(q>0));cand=sorted(cand.tolist(),key=lambda p:float(q[p[0],p[1]]),reverse=True);out=[]
    for y,x in cand:
        if all((y-yy)**2+(x-xx)**2>=25 for yy,xx,_ in out):out.append((y,x,float(q[y,x])))
        if len(out)==k:break
    return out
def main():
    out=Path(__file__).resolve().parents[1];root=out.parents[1];shared=root/"shared/coco_clean_20260911";sys.path.insert(0,str(shared));from structure_candidate_trace import TraceCapture
    selected=read(out/"selection.csv");resolution=read(out/"resolution_per_target.csv");rm={(r["annotation_id"],int(r["imgsz"])):r for r in resolution}
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(root/"assets/datasets/coco/annotations/instances_val2017.json"))
    model=YOLO(str(root/"assets/models/coco_clean_20260911/yolo26m-seg.pt"));model.model.eval().requires_grad_(False);model.model.model[-1].end2end=False;images=shared/"local_readout_runtime_20260912/data/images/val2017";rows=[];start=time.time()
    for num,r in enumerate(selected,1):
        iid=int(r["image_id"]);ann=gt.anns[int(r["annotation_id"])];info=gt.imgs[iid]
        with torch.inference_mode():model.predict(str(images/info["file_name"]),predictor=TraceCapture,imgsz=640,rect=False,conf=.001,iou=.7,max_det=300,retina_masks=False,device=0,verbose=False,end2end=False)
        raw=model.predictor.dense[0].float().cpu().numpy();heat=raw[4:84,:6400].max(0).reshape(80,80);boxes=model.predictor.capture["boxes"].float().cpu().numpy();centers=[((b[0]+b[2])/16,(b[1]+b[3])/16) for b in boxes];pp=peaks(heat,centers);tb=target_grid(ann,(info["height"],info["width"]));tb[:2]-=1;tb[2:]+=1;rank=next((i for i,(y,x,s) in enumerate(pp,1) if tb[0]<=x<=tb[2] and tb[1]<=y<=tb[3]),0);base=rm[(r["annotation_id"],640)];hi=rm[(r["annotation_id"],1280)];recoverable=r["cohort"]=="raw_geometry_small" and float(base["box_iou"])<.5 and float(hi["box_iou"])>=.5
        rows.append({**r,"recoverable_1280":int(recoverable),"proposal_rank":rank,"hit_at1":int(0<rank<=1),"hit_at3":int(0<rank<=3),"hit_at5":int(0<rank<=5),"hit_at10":int(0<rank<=10),"hit_at20":int(0<rank<=20),"target_local_peak":float(heat[max(0,int(tb[1])):min(80,int(tb[3])+1),max(0,int(tb[0])):min(80,int(tb[2])+1)].max(initial=0)),"top_proposal_score":pp[0][2] if pp else 0.0})
        if num%64==0 or num==len(selected):write(out/"raw_peak_proposals.csv",rows);print(f"[{num}/{len(selected)}] {time.time()-start:.1f}s",flush=True)
    summary=[]
    for name,rr in [("recoverable",[r for r in rows if r["recoverable_1280"]]),("unrecoverable_failure",[r for r in rows if r["cohort"]=="raw_geometry_small" and not r["recoverable_1280"]]),("matched_control",[r for r in rows if r["cohort"]=="matched_small_control"])]:
        summary.append({"group":name,"n":len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in ("hit_at1","hit_at3","hit_at5","hit_at10","hit_at20","target_local_peak","top_proposal_score")}})
    write(out/"raw_peak_summary.csv",summary);(out/"RAW_PEAK_COMPLETE.json").write_text(json.dumps({"status":"complete","targets":len(rows),"proposal":"top local P3 max-class peaks after suppressing retained detection centers","gt_role":"hit measurement only","elapsed_s":time.time()-start},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
