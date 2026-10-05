"""S071 paired contrasts; resample matched pairs and retain all harms."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args()
    assert (a.run/"COMPLETE.json").exists()
    d=pd.read_csv(a.run/"effects.csv")
    assert not d.duplicated(["annotation_id","arm","crop"]).any()
    counts=d.groupby(["arm","crop"]).size()
    assert counts.nunique()==1
    summary=d.groupby(["crop","density","arm"]).agg(n=("iou","size"),iou=("iou","mean"),
        coverage=("coverage","mean"),fp_per_gt=("fp_per_gt","mean"),hits75=("mask75","sum"),r75=("mask75","mean")).reset_index()
    summary.to_csv(a.run/"summary.csv",index=False)
    comparisons=[("own_query","original"),("own_query","reflected_nonown"),("own_query","local_mean"),
                 ("predicted_interior","original"),("local_mean","original")]
    rows=[]
    for crop,z in d.groupby("crop"):
        for group in ["all","requires_target_pixel_recovery","sufficient_true_pixels_but_residual_false_pixels"]:
            q=z if group=="all" else z[z.prior_residual==group]
            for new,base in comparisons:
                for metric in ["iou","coverage","fp_per_gt","mask75"]:
                    w=q.pivot(index=["pair_id","density"],columns="arm",values=metric).astype(float)
                    delta=(w[new]-w[base]).unstack("density").sort_index()
                    assert delta.notna().all().all()
                    rng=np.random.default_rng(20260913);draw=rng.integers(len(delta),size=(2000,len(delta)))
                    sampled=delta.to_numpy()[draw].mean(1)*100
                    for i,density in enumerate(delta.columns):
                        ci=np.quantile(sampled[:,i],[.025,.975])
                        rows.append(dict(crop=crop,residual=group,new=new,base=base,metric=metric,density=density,
                            n=len(delta),delta_points=100*delta[density].mean(),ci_low=ci[0],ci_high=ci[1]))
                    h,l=list(delta.columns).index("high"),list(delta.columns).index("low")
                    ci=np.quantile(sampled[:,h]-sampled[:,l],[.025,.975])
                    rows.append(dict(crop=crop,residual=group,new=new,base=base,metric=metric,density="high_minus_low",
                        n=len(delta),delta_points=100*(delta.high-delta.low).mean(),ci_low=ci[0],ci_high=ci[1]))
    effects=pd.DataFrame(rows);effects.to_csv(a.run/"paired_contrasts.csv",index=False)
    # Which cases improve, rather than only the mean.
    w=d[d.crop.eq("original_predicted_box")].pivot(index=["annotation_id","density"],columns="arm",values="iou")
    result=[]
    for density,z in w.groupby(level="density"):
        for arm in ["own_query","reflected_nonown","local_mean","predicted_interior"]:
            dd=z[arm]-z.original
            result.append(dict(density=density,arm=arm,n=len(z),improved=int((dd>1e-10).sum()),
                worsened=int((dd<-1e-10).sum()),unchanged=int((dd.abs()<=1e-10).sum())))
    pd.DataFrame(result).to_csv(a.run/"case_changes.csv",index=False)
    shutil.copy2(__file__,a.run/"source"/Path(__file__).name)
    print(summary.round(4).to_string(index=False))
    print(effects[(effects.residual=="all") & (effects.metric=="iou")].round(3).to_string(index=False))


if __name__=="__main__":main()
