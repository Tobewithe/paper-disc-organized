"""Can inference-visible P3 features identify 1280-recoverable small-object locations?"""
from __future__ import annotations
import contextlib,csv,io,json,sys,time
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO
def read(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(p,rows):
    with Path(p).open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    out=Path(__file__).resolve().parents[1];root=out.parents[1];shared=root/"shared/coco_clean_20260911";sys.path.insert(0,str(shared));from structure_candidate_trace import TraceCapture
    selected=read(out/"selection.csv");res=read(out/"resolution_per_target.csv");rm={(r["annotation_id"],int(r["imgsz"])):r for r in res}
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(root/"assets/datasets/coco/annotations/instances_val2017.json"))
    model=YOLO(str(root/"assets/models/coco_clean_20260911/yolo26m-seg.pt"));model.model.eval().requires_grad_(False);head=model.model.model[-1];head.end2end=False;captured={}
    def hook(module,args):captured["p3"]=args[0][0].detach().float().cpu()[0].numpy()
    handle=head.register_forward_pre_hook(hook);features=[];records=[];images=shared/"local_readout_runtime_20260912/data/images/val2017";start=time.time()
    for num,r in enumerate(selected,1):
        iid=int(r["image_id"]);aid=int(r["annotation_id"]);ann=gt.anns[aid];info=gt.imgs[iid];h,w=info["height"],info["width"];scale=min(640/h,640/w);nw,nh=round(w*scale),round(h*scale);left,top=round((640-nw)/2-.1),round((640-nh)/2-.1);x,y,bw,bh=ann["bbox"];gx=int(np.clip(((x+bw/2)*scale+left)/8-.5,0,79));gy=int(np.clip(((y+bh/2)*scale+top)/8-.5,0,79));captured.clear()
        with torch.inference_mode():model.predict(str(images/info["file_name"]),predictor=TraceCapture,imgsz=640,rect=False,conf=.001,iou=.7,max_det=300,retina_masks=False,device=0,verbose=False,end2end=False)
        p3=captured["p3"];y0=max(0,gy-1);y1=min(80,gy+2);x0=max(0,gx-1);x1=min(80,gx+2);center=p3[:,gy,gx];mean=p3[:,y0:y1,x0:x1].mean((1,2));mx=p3[:,y0:y1,x0:x1].max((1,2));raw=model.predictor.dense[0].float().cpu().numpy();idx=gy*80+gx;scores=raw[4:84,idx];box=raw[:4,idx];source=np.array([(gx+.5)*8,(gy+.5)*8]);extra=np.array([scores.max(),-(np.clip(scores,1e-9,1)*np.log(np.clip(scores,1e-9,1))).sum(),box[2]/640,box[3]/640,np.linalg.norm(box[:2]-source)/640])
        feat=np.r_[center,mean,mx,extra];base=rm[(r["annotation_id"],640)];hi=rm[(r["annotation_id"],1280)];positive=r["cohort"]=="raw_geometry_small" and float(base["box_iou"])<.5 and float(hi["box_iou"])>=.5;features.append(feat);records.append({**r,"grid_x":gx,"grid_y":gy,"recoverable":int(positive),"score_max":extra[0],"box_w_norm":extra[2],"box_h_norm":extra[3]})
        if num%64==0 or num==len(selected):print(f"[{num}/{len(selected)}] {time.time()-start:.1f}s",flush=True)
    handle.remove();write(out/"p3_router_targets.csv",records);X=np.asarray(features);y=np.asarray([r["recoverable"] for r in records]);groups=np.asarray([int(r["pair_id"]) for r in records]);cv=StratifiedGroupKFold(5,shuffle=True,random_state=0);models={"linear":make_pipeline(StandardScaler(),LogisticRegression(C=.05,max_iter=3000,class_weight="balanced")),"extra_trees":ExtraTreesClassifier(n_estimators=500,min_samples_leaf=8,max_features=.25,class_weight="balanced",random_state=0,n_jobs=-1)};results={}
    for name,m in models.items():pred=cross_val_predict(m,X,y,groups=groups,cv=cv,method="predict_proba",n_jobs=1)[:,1];results[name]={"auc":float(roc_auc_score(y,pred)),"positive_mean_score":float(pred[y==1].mean()),"negative_mean_score":float(pred[y==0].mean())}
    (out/"P3_ROUTER_ANALYSIS.json").write_text(json.dumps({"targets":len(y),"positives":int(y.sum()),"feature_scope":"GT-center P3 cell and 3x3 patch; values are inference-visible but location is GT-assisted","five_fold_grouped_by_pair":results,"decision":"Only a strong target-level signal justifies building a dense proposal head."},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
