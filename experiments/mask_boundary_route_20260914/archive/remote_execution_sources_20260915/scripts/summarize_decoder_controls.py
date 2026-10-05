"""Summarize recorded controls with image-cluster bootstrap of fixed-slot R75.

The bootstrap below is for the diagnostic recall, not for COCO AP.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--annotations",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    source=Path(args.input)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((source/"SUMMARY.json").read_text(encoding="utf-8"))
    ids=json.loads((source/"image_ids.json").read_text(encoding="utf-8"))
    columns={image_id:i for i,image_id in enumerate(ids)}
    truth=json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    totals=np.zeros(len(ids),dtype=np.int64)
    for ann in truth["annotations"]:
        if ann["image_id"] in columns and not ann.get("iscrowd",0):
            totals[columns[ann["image_id"]]]+=1
    base=np.zeros(len(ids),dtype=np.int64)
    keys=list(summary["metrics"])
    gains={key:np.zeros(len(ids),dtype=np.int64) for key in keys}
    repairs={key:0 for key in keys}
    damage={key:0 for key in keys}
    with (source/"instance_records.csv").open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key=row["variant"]
            i=columns[int(row["image_id"])]
            before=float(row["baseline_iou"])>=.75
            after=float(row["iou"])>=.75
            if key=="official_zero":
                base[i]+=int(before)
            gains[key][i]+=int(after)-int(before)
            repairs[key]+=int(not before and after)
            damage[key]+=int(before and not after)
    assert int(totals.sum())==summary["ordinary_gt"]
    rng=np.random.default_rng(20260915)
    draws=rng.integers(0,len(ids),size=(2000,len(ids)))
    denominators=totals[draws].sum(axis=1)
    estimates={}
    for key in keys:
        delta=100*gains[key].sum()/totals.sum()
        bootstrap=100*gains[key][draws].sum(axis=1)/denominators
        estimates[key]=dict(delta_r75_percentage_points=float(delta),
            baseline_r75=100*float(base.sum())/totals.sum(),
            r75=100*float(base.sum()+gains[key].sum())/totals.sum(),
            repaired=repairs[key],damaged=damage[key],
            image_bootstrap_95ci=[float(x) for x in np.quantile(bootstrap,[.025,.975])])
    comparisons={}
    for method,control in (("smooth_gated","official_gated"),("official_gated","official_global")):
        if method in gains and control in gains:
            difference=gains[method]-gains[control]
            sampled=100*difference[draws].sum(axis=1)/denominators
            comparisons[f"{method}_vs_{control}"]=dict(
                delta_r75_percentage_points=float(100*difference.sum()/totals.sum()),
                image_bootstrap_95ci=[float(x) for x in np.quantile(sampled,[.025,.975])])
    scope="development" if summary.get("start",0)==0 and len(ids)==500 else "frozen-rule confirmation"
    payload=dict(source_run_id=summary.get("run_id"),analysis_run_id=os.environ.get("RESEARCH_RUN_ID"),
        images=len(ids),ordinary_gt=int(totals.sum()),metric="fixed-slot Mask Recall@.75, all ordinary GT denominator",
        bootstrap="2000 paired image cluster resamples, seed 20260915; diagnostic recall only, not AP CI",
        scope=scope,estimates=estimates,paired_comparisons=comparisons,
        ap_delta={key:100*summary["delta_vs_official"][key][0] for key in keys})
    (out/"ANALYSIS.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    show=[key for key in keys if key!="official_zero"]
    labels={"official_gated":"Scale gate","official_global":"Global threshold","official_erode":"Small-mask erosion",
        "smooth_gated":"Smooth scale","input_area_gated":"Input-scale gate","legacy_zero":"Legacy zero",
        "legacy_gated":"Legacy scale gate"}
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
    y=np.arange(len(show))
    ap=[payload["ap_delta"][key] for key in show]
    axes[0].barh(y,ap,color=["#147d92" if x>=0 else "#c66047" for x in ap])
    axes[0].set_yticks(y,[labels.get(key,key) for key in show])
    axes[0].invert_yaxis()
    axes[0].axvline(0,color="#333333",linewidth=.8)
    axes[0].set_xlabel("Mask AP change (points)")
    axes[0].set_title("Paired decoder comparison")
    values=np.array([estimates[key]["delta_r75_percentage_points"] for key in show])
    ci=np.array([estimates[key]["image_bootstrap_95ci"] for key in show])
    axes[1].errorbar(values,y,xerr=np.maximum(0,np.array([values-ci[:,0],ci[:,1]-values])),fmt="o",color="#147d92",capsize=3)
    axes[1].axvline(0,color="#333333",linewidth=.8)
    axes[1].set_yticks(y,[f'+{repairs[key]} / -{damage[key]}' for key in show])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Fixed-slot R75 change (percentage points)")
    axes[1].set_title("Repair / damage; 95% image bootstrap CI")
    fig.suptitle(f"COCO {scope} {len(ids)} images | YOLO26m-seg | {summary['branch']}")
    for extension in ("png","pdf","svg"):
        fig.savefig(out/f"decoder_controls.{extension}",dpi=180)
    plt.close(fig)
    print(json.dumps(payload,indent=2))


if __name__=="__main__":
    main()
