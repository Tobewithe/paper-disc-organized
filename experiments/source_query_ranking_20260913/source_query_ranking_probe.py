"""S072: diagnose donor query calibration versus pixel ordering.

Uses saved S071 tensors, no model forward or learning. At the same exact input
crop, compare rank AUC and GT-controlled equal-coverage precision. This cannot
be a deployable threshold or method AP, but can reject a pure calibration
explanation for source-query failure.
"""
from pathlib import Path
import argparse
import contextlib
import io
import json
import shutil
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from joint_failure_decoder_probe import ROOT,ANNOTATION,COCO,torch,F,cv2,dump,save_csv,ops
from source_location_coefficient_probe import gt_input


def auc(pos,neg):
    if not len(pos) or not len(neg):return np.nan
    ranks=rankdata(np.r_[pos,neg],method="average")
    return float((ranks[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))


def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    selection=pd.read_csv(a.source/"selection.csv")
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ANNOTATION))
    dump(a.out/"protocol.json",dict(experiment="S072_SOURCE_QUERY_RANKING",training=False,model_forward=False,
        question="Is S071 donor deterioration explained solely by threshold/area calibration, or does the donor order target-vs-negative pixels worse?",
        decision="Only keep source-query line if own query improves ordering vs original under both AUC and equal-coverage controls; otherwise stop untrained query relocation/averaging recipe.",
        population="All96 S071 matched targets; no outcome-based exclusion; valid-region denominator and missing comparison classes explicit.",
        metrics="Within exact original640crop minus crowd, AUC(own vs all negatives / same-neighbor / background), average ranks for ties. Equal coverage threshold matches original z>0 own coverage; entire boundary-tie group included and achieved coverage recorded.",
        caveat="GT sets diagnostic threshold and rank labels, no deployment claim. Input640 representation masks differ from originalCOCO output evaluation; no official AP. CPU logits computed identically for all saved coefficients. Checks here compare continuous ordering, not exact binary replay of S071 GPU GEMM."))
    rows=[]
    for r in selection.to_dict("records"):
        iid,aid=int(r["image_id"]),int(r["annotation_id"])
        item=np.load(a.source/"tensors"/f"{aid}.npz")
        own,inp,_,_,_=gt_input(gt,iid,aid)
        same=np.zeros_like(inp);any_other=np.zeros_like(inp);crowd=np.zeros_like(inp)
        for ann in gt.imgToAnns[iid]:
            if ann["id"]==aid:continue
            mask=gt_input(gt,iid,ann["id"])[1]
            if ann.get("iscrowd",0) or ann.get("ignore",0):crowd|=mask
            else:
                any_other|=mask
                if ann["category_id"]==int(r["category_id"]):same|=mask
        c=torch.tensor(item["coeff"],dtype=torch.float32);proto=torch.tensor(item["proto"],dtype=torch.float32)
        z=F.interpolate((c@proto.flatten(1)).reshape(1,len(c),*proto.shape[-2:]),(640,640),mode="bilinear",align_corners=False)[0].numpy()
        support=ops.crop_mask(torch.ones((1,640,640),dtype=torch.uint8),torch.tensor(item["pred_box"])[None]).numpy()[0].astype(bool)
        valid=support&~crowd;pos=valid&inp;negative=valid&~inp
        neighbour=negative&same;background=negative&~any_other
        npos=int(pos.sum());nneg=int(negative.sum())
        original_index=list(item["arms"]).index("original")
        target_count=int((z[original_index][pos]>0).sum())
        for arm,score in zip(item["arms"],z):
            rr=dict(annotation_id=aid,image_id=iid,pair_id=int(r["pair_id"]),density=r["density"],prior_residual=r["prior_residual"],arm=str(arm),
                npos=npos,nneg=nneg,nsame=int(neighbour.sum()),nbg=int(background.sum()),target_positive_count=target_count,
                auc_all=auc(score[pos],score[negative]),auc_same=auc(score[pos],score[neighbour]),auc_background=auc(score[pos],score[background]))
            if npos and target_count:
                # >= includes threshold ties; never arbitrarily pick GT-aware
                # negatives out of a tied score to improve precision.
                threshold=np.sort(score[pos])[-target_count]
                pp=(score>=threshold)&valid;tp=int((pp&pos).sum());fp=int((pp&negative).sum())
                rr.update(threshold=float(threshold),achieved_positive_count=tp,coverage=tp/npos,
                    coverage_excess=(tp-target_count)/npos,precision=tp/max(tp+fp,1),fp_per_gt=fp/npos,
                    same_fp_per_gt=int((pp&neighbour).sum())/npos,bg_fp_per_gt=int((pp&background).sum())/npos)
            else:
                rr.update(threshold=None,achieved_positive_count=None,coverage=None,coverage_excess=None,
                    precision=None,fp_per_gt=None,same_fp_per_gt=None,bg_fp_per_gt=None)
            rows.append(rr)
    save_csv(a.out/"ranking.csv",rows)
    d=pd.DataFrame(rows);effects=[]
    for density,g in d.groupby("density"):
        for arm in ["own_query","reflected_nonown","local_mean","predicted_interior"]:
            for metric in ["auc_all","auc_same","auc_background","precision","fp_per_gt","same_fp_per_gt","bg_fp_per_gt"]:
                w=g.pivot(index="annotation_id",columns="arm",values=metric)[["original",arm]].dropna()
                change=(w[arm]-w.original).to_numpy()
                if len(change):
                    rng=np.random.default_rng(20260913);boot=change[rng.integers(len(change),size=(2000,len(change)))].mean(1)
                    ci=np.quantile(boot,[.025,.975])
                    effects.append(dict(density=density,arm=arm,metric=metric,n=len(change),
                        original=float(w.original.mean()),new=float(w[arm].mean()),delta=float(change.mean()),ci_low=ci[0],ci_high=ci[1]))
    pd.DataFrame(effects).to_csv(a.out/"contrasts.csv",index=False)
    dump(a.out/"COMPLETE.json",dict(status="COMPLETE",targets=len(selection),rows=len(rows),
        max_coverage_excess=float(d.coverage_excess.max()),valid_equal_coverage=int(d.precision.notna().sum())))
    shutil.copy2(__file__,a.out/Path(__file__).name)
    print(pd.DataFrame(effects).query("arm=='own_query'").round(4).to_string(index=False))
    print(json.dumps(dict(max_coverage_excess=float(d.coverage_excess.max()))))


if __name__=="__main__":main()
