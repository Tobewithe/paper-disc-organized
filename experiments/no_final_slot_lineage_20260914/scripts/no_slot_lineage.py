"""Trace every COCO GT through YOLO26's one-to-many candidate lifecycle.

The experiment identifies the first stage at which a GT loses all same-class Box50 candidates.
Final matching uses the official per-image COCO bbox matcher; stage availability itself is a
nonexclusive diagnostic, so an unmatched GT with a surviving candidate is assignment competition.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.utils import nms, ops


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def matrix_iou(boxes: np.ndarray, gtboxes: np.ndarray) -> np.ndarray:
    lt=np.maximum(gtboxes[:,None,:2],boxes[None,:,:2]); rb=np.minimum(gtboxes[:,None,2:],boxes[None,:,2:])
    inter=np.maximum(rb-lt,0).prod(2); ga=np.maximum(gtboxes[:,2:]-gtboxes[:,:2],0).prod(1); ba=np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(1)
    return inter/np.maximum(ga[:,None]+ba[None,:]-inter,1e-12)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--images",type=int,default=5000); a=ap.parse_args()
    out=Path(__file__).resolve().parents[1]; root=out.parents[1]; shared=root/"shared/coco_clean_20260911"; sys.path.insert(0,str(shared))
    from structure_candidate_trace import TraceCapture
    from frozen_mechanism_probe import ownership
    ann_path=root/"assets/datasets/coco/annotations/instances_val2017.json"; image_root=shared/"local_readout_runtime_20260912/data/images/val2017"; weight=root/"assets/models/coco_clean_20260911/yolo26m-seg.pt"
    with contextlib.redirect_stdout(io.StringIO()): gt=COCO(str(ann_path))
    ids=sorted(gt.imgs)[:a.images]; categories=sorted(gt.cats); catindex={cat:i for i,cat in enumerate(categories)}
    historical={}
    with (root/"experiments/coco_failure_dimensions_20260913/instances_improved.csv").open(encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f): historical[int(r["annotation_id"])]=r
    model=YOLO(str(weight)); model.model.eval().requires_grad_(False); model.model.model[-1].end2end=False
    stages=["raw_geometry","correct_class","score_0001","nms","top300","nonempty","eval100"]
    rows=[]; start=time.time()
    for number,iid in enumerate(ids,1):
        with torch.inference_mode():
            model.predict(str(image_root/gt.imgs[iid]["file_name"]),predictor=TraceCapture,imgsz=640,rect=False,conf=.001,iou=.7,max_det=300,retina_masks=False,device=0,verbose=False,end2end=False)
        pred=model.predictor; raw=pred.dense; cap=pred.capture; before=pred.before_empty
        allnms,idx=nms.non_max_suppression(raw.clone(),conf_thres=.001,iou_thres=.7,nc=80,max_det=raw.shape[-1],return_idxs=True,end2end=False)
        keep_all=idx[0].flatten().to(dtype=torch.long); keep300=keep_all[:300]
        if not torch.equal(allnms[0][:300],before): raise RuntimeError(f"NMS replay mismatch {iid}")
        replay=ops.process_mask(cap["proto"],before[:,6:],before[:,:4],cap["input_shape"],upsample=True)
        nonempty_mask=replay.flatten(1).any(1).bool(); keep=keep300[nonempty_mask]
        rawboxes=ops.xywh2xyxy(raw[0,:4].T.clone()); scaled=ops.scale_boxes(cap["input_shape"],rawboxes.clone(),cap["shape"]).cpu().numpy()
        scores=raw[0,4:84].T.float().cpu().numpy(); cls=scores.argmax(1); conf=scores.max(1); total=len(cls)
        pools={"raw_geometry":np.arange(total),"correct_class":np.arange(total),"score_0001":np.flatnonzero(conf>.001),"nms":keep_all.cpu().numpy(),"top300":keep300.cpu().numpy(),"nonempty":keep.cpu().numpy()}
        filtered=[]
        for label in range(80): filtered.extend([int(j) for j in pools["nonempty"] if cls[j]==label][:100])
        pools["eval100"]=np.asarray(filtered,dtype=int)
        ordinary=[q for q in gt.imgToAnns[iid] if not q.get("iscrowd",0)]
        if not ordinary: continue
        gtboxes=np.asarray([[q["bbox"][0],q["bbox"][1],q["bbox"][0]+q["bbox"][2],q["bbox"][1]+q["bbox"][3]] for q in ordinary])
        overlap=matrix_iou(scaled,gtboxes); mapping=ownership(gt,iid,cap["detections"])
        for gi,ann in enumerate(ordinary):
            label=catindex[ann["category_id"]]; available={}
            for stage,pool in pools.items():
                good=overlap[gi,pool]>=.5
                if stage!="raw_geometry": good &= cls[pool]==label
                available[stage]=int(good.sum())
            first_loss=next((stage for stage in stages if available[stage]==0),"assignment_competition" if ann["id"] not in mapping else "matched")
            geom=overlap[gi]>=.5
            h=historical.get(ann["id"],{})
            rows.append({"image_id":iid,"annotation_id":ann["id"],"category_id":ann["category_id"],"area":ann["area"],"area_bin":h.get("area_bin",""),"density_e4":h.get("density_e4",""),"historical_scope":h.get("failure_scope",""),"current_bbox50_matched":ann["id"] in mapping,"first_loss":first_loss,"best_raw_box_iou":float(overlap[gi].max()),"best_geometry_trueclass_score":float(scores[geom,label].max()) if geom.any() else 0.0,**{f"{s}_count":available[s] for s in stages}})
        if number%100==0 or number==len(ids):
            write_csv(out/"per_gt.csv",rows); (out/"progress.json").write_text(json.dumps({"images":number,"total":len(ids),"gt":len(rows),"elapsed_s":time.time()-start},indent=2),encoding="utf-8"); print(f"[{number}/{len(ids)}] gt={len(rows)} elapsed={time.time()-start:.1f}s",flush=True)
    summaries=[]
    for scope,rr in [("all",rows),("current_unmatched",[r for r in rows if not r["current_bbox50_matched"]]),("historical_no_final_slot",[r for r in rows if r["historical_scope"]=="no_final_slot"])]:
        counts=Counter(r["first_loss"] for r in rr)
        for stage,count in counts.items(): summaries.append({"scope":scope,"first_loss":stage,"count":count,"fraction":count/max(len(rr),1),"total":len(rr)})
    write_csv(out/"summary.csv",summaries)
    (out/"COMPLETE.json").write_text(json.dumps({"status":"complete","images":len(ids),"gt":len(rows),"current_unmatched":sum(not r["current_bbox50_matched"] for r in rows),"ultralytics":"8.4.100","end2end":False,"elapsed_s":time.time()-start},ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__": main()
