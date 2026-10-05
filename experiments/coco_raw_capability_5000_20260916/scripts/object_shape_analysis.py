"""Exploratory object-property analysis on the full fixed GT population."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    for k in ("root","evaluation","out"):ap.add_argument("--"+k,type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    gt=json.loads((a.root/"assets/datasets/coco/annotations/instances_val2017.json").read_text())
    annotations={r["id"]:r for r in gt["annotations"]}
    rows=[json.loads(line) for line in (a.evaluation/"gt_joined.jsonl").read_text().splitlines()]
    eligible=[]
    for r in rows:
        ann=annotations[r["annotation_id"]];w,h=ann["bbox"][2:]
        r["gt_fill_fraction"]=r["mask_area"]/max(w*h,1e-9)
        r["gt_box_aspect"]=max(w,h)/max(min(w,h),1e-9)
        r["fill_bin"]= "<0.25" if r["gt_fill_fraction"]<.25 else "0.25-0.5" if r["gt_fill_fraction"]<.5 else "0.5-0.75" if r["gt_fill_fraction"]<.75 else ">=0.75"
        r["mask_failure"]=r["mask_max"]<.75
        if r["box_max"]>=.75:eligible.append(r)
    def summarize(rs):
        return dict(n=len(rs),failures=sum(r["mask_failure"] for r in rs),
            failure_rate=sum(r["mask_failure"] for r in rs)/len(rs) if rs else None,
            mean_mask_max=float(np.mean([r["mask_max"] for r in rs])) if rs else None)
    strata=defaultdict(lambda:[[],[]])
    for r in eligible:
        group=0 if r["gt_fill_fraction"]<.5 else 1 if r["gt_fill_fraction"]>=.75 else None
        if group is not None:
            # Match original category and a one-octave GT-area bin.
            strata[(r["category_id"],int(np.floor(np.log2(max(r["area"],1)))))][group].append(r)
    differences=[]
    for (cat,area_bin),(lo,hi) in strata.items():
        if len(lo)<5 or len(hi)<5:continue
        lp,hp=summarize(lo),summarize(hi)
        differences.append(dict(category_id=cat,area_log2_bin=area_bin,low_n=len(lo),high_n=len(hi),
            risk_difference=lp["failure_rate"]-hp["failure_rate"],weight=min(len(lo),len(hi))))
    total_weight=sum(r["weight"] for r in differences)
    summary=dict(source_evaluation=a.evaluation.name,total_gt=len(rows),cohort="GT with at least one Box75 raw candidate, score/class independent",
        eligible=summarize(eligible),by_fill={k:summarize([r for r in eligible if r["fill_bin"]==k]) for k in ("<0.25","0.25-0.5","0.5-0.75",">=0.75")},
        by_area_fill={g:{k:summarize([r for r in eligible if r["area_group"]==g and r["fill_bin"]==k]) for k in ("<0.25","0.25-0.5","0.5-0.75",">=0.75")} for g in ("small","medium","large")},
        aspect_ge3=summarize([r for r in eligible if r["gt_box_aspect"]>=3]),aspect_lt3=summarize([r for r in eligible if r["gt_box_aspect"]<3]),
        class_area_stratified=dict(strata=len(differences),overlap_weight=total_weight,
            weighted_risk_difference=sum(r["weight"]*r["risk_difference"] for r in differences)/total_weight if total_weight else None),
        limitations=["Exploratory association, not a causal feature intervention.","GT mask filling fraction is not directly available at deployment.",
            "Stratification controls measured class and coarse area only, not all shape/visibility/annotation confounding.",
            "Rasterized mask pixels and continuous COCO bbox area can make fill slightly exceed one."])
    (a.out/"SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    (a.out/"matched_strata.json").write_text(json.dumps(differences),encoding="utf-8")
    (a.out/"COMPLETE.json").write_text(json.dumps(dict(images=5000,gt=len(rows))),encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
