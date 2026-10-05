"""Describe which raw-geometry failures are recovered by 1280 inference."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict
def read(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(p,rows):
    with Path(p).open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
out=Path(__file__).resolve().parents[1];phen={r["annotation_id"]:r for r in read(out/"phenotypes.csv") if r["cohort"]=="raw_geometry_small"};res=read(out/"resolution_per_target.csv");by={}
for r in res:
    if r["cohort"]=="raw_geometry_small":by.setdefault(r["annotation_id"],{})[int(r["imgsz"])]=r
features=["bbox_fill","abs_log_aspect","input_min_side","input_max_side","compactness","perimeter_over_sqrt_area","connected_components","annotation_parts","border_distance_norm","lab_target_outer_contrast","gray_target_outer_contrast","boundary_gradient","target_gray_std","local_background_gradient","same_exposure4","different_exposure4","any_overlap_fraction","ici","local_true_score_max","local_top1_correct","local_box_center_error_norm","local_log_width_error","local_log_height_error","best_true_score_global"]
ids=[aid for aid,v in by.items() if aid in phen and 640 in v and 1280 in v];X=np.asarray([[float(phen[aid][f]) for f in features] for aid in ids]);y=np.asarray([float(by[aid][640]["box_iou"])<.5 and float(by[aid][1280]["box_iou"])>=.5 for aid in ids],int);cv=StratifiedKFold(5,shuffle=True,random_state=0);model=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=8,max_features=.7,class_weight="balanced",random_state=0,n_jobs=-1);pred=cross_val_predict(model,X,y,cv=cv,method="predict_proba")[:,1];model.fit(X,y);imp=sorted([{"feature":f,"importance":float(v)} for f,v in zip(features,model.feature_importances_)],key=lambda q:q["importance"],reverse=True);write(out/"responder_feature_importance.csv",imp);(out/"RESPONDER_ANALYSIS.json").write_text(json.dumps({"targets":len(ids),"box50_recovered":int(y.sum()),"recovery_rate":float(y.mean()),"five_fold_auc":float(roc_auc_score(y,pred)),"boundary":"Most features use GT geometry/masks and describe the recoverable object; they are not deployment inputs."},ensure_ascii=False,indent=2),encoding="utf-8")
