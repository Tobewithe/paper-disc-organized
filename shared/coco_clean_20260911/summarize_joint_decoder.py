"""Analyze S070 without interpreting crop/anchor observations as network causes."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    a=p.parse_args()
    assert json.loads((a.run/"COMPLETE.json").read_text())["full_val_task_evaluated"]
    d=pd.read_csv(a.run/"instance_effects.csv")
    assert len(d)==7690 and d.annotation_id.nunique()==len(d)
    assert (d.crop_fn+d.response_fn == d.gt_area-d.original_tp).all()
    assert (d.exact_support_coverage+1e-12 >= d.original_tp/d.gt_area).all()
    d["original_coverage"]=d.original_tp/d.gt_area
    d["raw_coverage"]=d.raw_tp/d.gt_area
    d["gt_crop_coverage"]=d.gt_crop_tp/d.gt_area
    d["source_not_own"]=d.source_center_location.ne("own")
    bad=d[d.original_state.eq("box_bad__mask_bad")]
    control=d[d.original_state.eq("box_bad__mask_good")]
    rows=[]
    for scope,dd in [("joint_failure",bad),("poor_box_mask_good_control",control)]:
        for group in ["all","low","middle","high","undefined"]:
            z=dd if group=="all" else dd[dd.density.eq(group)]
            rows.append(dict(scope=scope,density=group,n=len(z),
                original_iou=z.original_iou.mean(),gt_crop_iou=z.gt_crop_iou.mean(),delta_iou=z.delta_iou.mean(),
                new_mask75=int(z.gt_crop75.sum()),new_mask75_rate=z.gt_crop75.mean(),
                original_coverage=z.original_coverage.mean(),raw_coverage=z.raw_coverage.mean(),gt_crop_coverage=z.gt_crop_coverage.mean(),
                crop_fn_fraction=z.crop_fn_fraction.mean(),response_fn_fraction=z.response_fn_fraction.mean(),
                exact_support_below75=int((z.exact_support_coverage<.75).sum()),
                raw_coverage_below75=int((z.raw_coverage<.75).sum()),gt_support_below95=int((z.gt_support_coverage<.95).sum()),
                source_not_own=z.source_not_own.mean(),source_cell_own_fraction=z.source_cell_own_fraction.mean()))
    summary=pd.DataFrame(rows)
    summary.to_csv(a.run/"diagnostic_summary.csv",index=False)
    residual=[]
    for (group,state),z in bad.groupby(["density","prior_residual"]):
        residual.append(dict(density=group,state=state,n=len(z),gt_crop75=int(z.gt_crop75.sum()),gt_crop75_rate=z.gt_crop75.mean(),
            original_iou=z.original_iou.mean(),gt_crop_iou=z.gt_crop_iou.mean(),original_coverage=z.original_coverage.mean(),
            raw_coverage=z.raw_coverage.mean(),gt_crop_coverage=z.gt_crop_coverage.mean(),crop_fn_fraction=z.crop_fn_fraction.mean(),
            response_fn_fraction=z.response_fn_fraction.mean(),exact_support_below75=int((z.exact_support_coverage<.75).sum()),
            raw_coverage_below75=int((z.raw_coverage<.75).sum()),
            source_not_own=z.source_not_own.mean()))
    pd.DataFrame(residual).to_csv(a.run/"prior_residual_transitions.csv",index=False)
    d.groupby(["density","original_state","source_center_location"]).size().reset_index(name="n").to_csv(a.run/"source_location_counts.csv",index=False)
    d.groupby(["density","original_state","source_stride"]).size().reset_index(name="n").to_csv(a.run/"source_stride_counts.csv",index=False)
    d["after_gtcrop_state"]=np.select([d.gt_crop75,d.raw_coverage<.75],
        ["recovered","raw_positive_target_support_below75"],default="raw_target_support_exists_but_mask_still_bad")
    d.to_csv(a.run/"instances_analyzed.csv",index=False)
    # Conditional source-location comparison in common class/size/stride strata.
    # No selection of a 'best' threshold or outcome-dependent parameter.
    composition=[]
    keys=["category_id","area_bin","source_stride"]
    for density in ["low","high"]:
        z=d[d.density.eq(density)].copy()
        counts=z.groupby(keys+["original_state"]).size().unstack(fill_value=0)
        common=counts[(counts["box_bad__mask_bad"]>=5)&(counts["box_bad__mask_good"]>=5)]
        q=z.merge(common.reset_index()[keys],on=keys,how="inner",validate="many_to_one")
        weights=common.sum(axis=1)/common.to_numpy().sum()
        row=dict(density=density,common_strata=len(common),retained_by_state=q.groupby("original_state").size().to_dict())
        for state,rr in q.groupby("original_state"):
            prob=rr.groupby(keys).source_not_own.mean()
            row[state+"_source_not_own_standardized"]=float((weights*prob).sum())
        composition.append(row)
    (a.run/"source_composition.json").write_text(json.dumps(composition,indent=2),encoding="utf-8")
    # Correct image-cluster uncertainty for conditional per-GT IoU changes.
    image_ids=np.sort(d.image_id.unique());lookup={iid:i for i,iid in enumerate(image_ids)}
    rng=np.random.default_rng(20260913);draws=rng.integers(len(image_ids),size=(2000,len(image_ids)))
    cis={}
    for density in ["low","high"]:
        z=bad[bad.density.eq(density)]
        sums=np.zeros((len(image_ids),2));ix=np.array([lookup[x] for x in z.image_id])
        np.add.at(sums[:,0],ix,z.delta_iou.to_numpy());np.add.at(sums[:,1],ix,1)
        boot=sums[draws].sum(1)
        values=100*boot[:,0]/np.where(boot[:,1]>0,boot[:,1],np.nan)
        cis[density]=dict(delta_iou_points=float(100*z.delta_iou.mean()),ci95_points=np.nanquantile(values,[.025,.975]).tolist())
    (a.run/"conditional_iou_ci.json").write_text(json.dumps(cis,indent=2),encoding="utf-8")
    shutil.copy2(__file__,a.run/"source"/Path(__file__).name)
    print(summary.round(4).to_string(index=False))
    print(pd.DataFrame(residual).query("density=='high'").round(4).to_string(index=False))
    print(json.dumps(composition,indent=2))
    print(json.dumps(cis,indent=2))


if __name__=="__main__":main()
