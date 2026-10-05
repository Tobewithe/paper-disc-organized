"""Phenotype analysis for exact-category, nearest-area paired small-object failures."""
from __future__ import annotations
import contextlib,csv,io,json,math
from pathlib import Path
import cv2
import numpy as np
from pycocotools.coco import COCO
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def read(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(p,rows):
    if not rows:return
    with Path(p).open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def boot(x,seed=0,reps=4000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);m=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(reps)]);q=np.quantile(m,[.025,.975]);return float(x.mean()),float(q[0]),float(q[1])
def main():
    out=Path(__file__).resolve().parents[1];root=out.parents[1];shared=root/"shared/coco_clean_20260911";ann_path=root/"assets/datasets/coco/annotations/instances_val2017.json";images=shared/"local_readout_runtime_20260912/data/images/val2017"
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ann_path))
    base=read(out/"per_target.csv");meta={r["annotation_id"]:r for r in read(root/"experiments/coco_failure_dimensions_20260913/instances_improved.csv")};rows=[]
    for i,r in enumerate(base,1):
        ann=gt.anns[int(r["annotation_id"])];info=gt.imgs[int(r["image_id"])];mask=gt.annToMask(ann).astype(np.uint8);h,w=mask.shape;image=cv2.imread(str(images/info["file_name"]));lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);perim=sum(cv2.arcLength(c,True) for c in contours);area=float(mask.sum());x,y,bw,bh=ann["bbox"];box_area=max(float(bw*bh),1e-9);dilate=cv2.dilate(mask,np.ones((7,7),np.uint8));erode=cv2.erode(mask,np.ones((3,3),np.uint8));outer=(dilate>0)&(mask==0);boundary=(dilate>0)&(erode==0);inside=mask>0
        other=np.zeros_like(mask,bool)
        for oid in gt.getAnnIds(imgIds=[ann["image_id"]],iscrowd=None):
            if oid!=ann["id"] and not gt.anns[oid].get("iscrowd",0):other|=gt.annToMask(gt.anns[oid]).astype(bool)
        clean_outer=outer&~other
        grad=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3))
        target_lab=lab[inside].mean(0) if inside.any() else np.zeros(3);outer_lab=lab[clean_outer].mean(0) if clean_outer.any() else target_lab
        x0=max(0,int(x-bw));y0=max(0,int(y-bh));x1=min(w,int(x+2*bw));y1=min(h,int(y+2*bh));local=np.zeros_like(mask,bool);local[y0:y1,x0:x1]=True;local_bg=local&~inside&~other
        m=meta[r["annotation_id"]]
        rows.append({**r,"bbox_fill":area/box_area,"abs_log_aspect":abs(math.log(max(bw,1e-9)/max(bh,1e-9))),"input_min_side":min(float(r["gt_input_w"]),float(r["gt_input_h"])),"input_max_side":max(float(r["gt_input_w"]),float(r["gt_input_h"])),"compactness":4*math.pi*area/max(perim*perim,1e-9),"perimeter_over_sqrt_area":perim/max(math.sqrt(area),1e-9),"connected_components":cv2.connectedComponents(mask)[0]-1,"annotation_parts":len(ann["segmentation"]) if isinstance(ann.get("segmentation"),list) else 1,"border_distance_norm":min(x,y,w-(x+bw),h-(y+bh))/max(math.sqrt(box_area),1e-9),"lab_target_outer_contrast":float(np.linalg.norm(target_lab-outer_lab)),"gray_target_outer_contrast":float(abs(gray[inside].mean()-gray[clean_outer].mean())) if clean_outer.any() else 0.0,"boundary_gradient":float(grad[boundary].mean()) if boundary.any() else 0.0,"target_gray_std":float(gray[inside].std()) if inside.any() else 0.0,"local_background_gradient":float(grad[local_bg].mean()) if local_bg.any() else 0.0,"same_exposure4":float(m["same_boundary_exposure4"] or 0),"different_exposure4":float(m["different_boundary_exposure4"] or 0),"any_overlap_fraction":float(m["any_overlap_fraction"] or 0),"ici":float(m["ici"] or 0)})
        if i%100==0:print(f"[{i}/{len(base)}]",flush=True)
    write(out/"phenotypes.csv",rows)
    features=["bbox_fill","abs_log_aspect","input_min_side","input_max_side","compactness","perimeter_over_sqrt_area","connected_components","annotation_parts","border_distance_norm","lab_target_outer_contrast","gray_target_outer_contrast","boundary_gradient","target_gray_std","local_background_gradient","same_exposure4","different_exposure4","any_overlap_fraction","ici"]
    bypair={}
    for r in rows:bypair.setdefault(r["pair_id"],{})[r["cohort"]]=r
    effects=[]
    for f in features:
        diffs=[float(v["raw_geometry_small"][f])-float(v["matched_small_control"][f]) for v in bypair.values() if len(v)==2];mean,lo,hi=boot(diffs);allv=np.asarray([float(r[f]) for r in rows]);effects.append({"feature":f,"failure_minus_control":mean,"ci_low":lo,"ci_high":hi,"pooled_sd":float(allv.std()),"standardized_effect":mean/max(float(allv.std()),1e-9),"pairs":len(diffs)})
    write(out/"paired_effects.csv",sorted(effects,key=lambda q:abs(q["standardized_effect"]),reverse=True))
    X=np.asarray([[float(r[f]) for f in features] for r in rows]);y=np.asarray([r["cohort"]=="raw_geometry_small" for r in rows],int);groups=np.asarray([int(r["pair_id"]) for r in rows]);cv=StratifiedGroupKFold(5,shuffle=True,random_state=0);pred=np.zeros(len(rows));coefs=[]
    for tr,te in cv.split(X,y,groups):
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.2,max_iter=2000,class_weight="balanced"));model.fit(X[tr],y[tr]);pred[te]=model.predict_proba(X[te])[:,1];coefs.append(model[-1].coef_[0])
    coef=np.mean(coefs,0);write(out/"predictive_coefficients.csv",sorted([{"feature":f,"standardized_logit_coefficient":float(c)} for f,c in zip(features,coef)],key=lambda q:abs(q["standardized_logit_coefficient"]),reverse=True));(out/"ANALYSIS.json").write_text(json.dumps({"pairs":len(bypair),"features":features,"image_disjoint_grouped_cv_auc":float(roc_auc_score(y,pred)),"interpretation":"Associations after exact-category nearest-area pairing; no causal claim."},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
