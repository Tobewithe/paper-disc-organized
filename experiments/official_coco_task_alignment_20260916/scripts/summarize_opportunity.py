"""Distinguish diagnostic slot failures from task failures; compare native branches."""
import argparse
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    gt=list(csv.DictReader((args.cache/"gt_inventory.csv").open(encoding="utf-8")))
    official={int(r["annotation_id"]):r for r in json.loads((args.cache/"official_gt_mask75.json").read_text()) if not r["ignore"]}
    table=defaultdict(Counter);repair=Counter();repair_task_failed=Counter()
    for r in gt:
        state=r["state"];aid=int(r["annotation_id"])
        table[state]["n"]+=1
        table[state]["official_mask75_matched"]+=int(official[aid]["matched"])
        table[state]["any_sameclass_exported_mask75"]+=int(float(r["best_exported_mask_iou"])>=.75)
        repair[r["pixel_direction"]]+=1
        if not official[aid]["matched"]:
            repair_task_failed[r["pixel_direction"]]+=1
    instances=json.loads((args.probe/"instances.json").read_text())
    idx={(r["annotation_id"],r["branch"]):r for r in instances}
    candidates={}
    with (args.probe/"candidates.jsonl").open() as f:
        for line in f:
            r=json.loads(line);candidates[(r["annotation_id"],r["branch"],r["raw_index"])]=r
    detailed=[]
    for one in instances:
        if one["branch"]!="one2one":continue
        many=idx[(one["annotation_id"],"one2many")]
        same=candidates.get((one["annotation_id"],"one2many",one["reference_raw_index"]))
        pool=[v for k,v in candidates.items() if k[:2]==(one["annotation_id"],"one2many") and v["ownership"]=="own_gt"]
        bestbox=max(pool,key=lambda r:r["box_iou"]) if pool else None
        bestscore=max(pool,key=lambda r:r["gt_score"]) if pool else None
        bestmask=max(pool,key=lambda r:r["fixed_mask_iou"]) if pool else None
        r={"annotation_id":one["annotation_id"],"image_id":one["image_id"],"cohort":one["cohort"],"pair_id":one["pair_id"],
           "official_mask75_matched":official[one["annotation_id"]]["matched"],
           "original_iou":one["reference_mask_iou"],"o2o_best_native":one["best_native_iou"],"o2o_best_fixed":one["best_fixed_iou"],
           "o2o_best_box_mask":one["best_box_mask_iou"],"o2o_own_fixed":one["best_own_fixed_iou"],
           "o2m_same_position_fixed":same["fixed_mask_iou"] if same else None,
           "o2m_own_bestmask_fixed":many["best_own_fixed_iou"],
           "o2m_own_bestbox_fixed":bestbox["fixed_mask_iou"] if bestbox else None,
           "o2m_own_bestscore_fixed":bestscore["fixed_mask_iou"] if bestscore else None,
           "o2m_bestmask_stride":bestmask["stride"] if bestmask else None,
           "o2m_bestmask_gt_score":bestmask["gt_score"] if bestmask else None,
           "o2o_reference_stride":one["reference_stride"]}
        detailed.append(r)
    columns=[k for k in detailed[0] if k.startswith("o2") and ("fixed" in k or "mask" in k or k=="o2o_best_native") and k not in ("o2m_bestmask_stride","o2m_bestmask_gt_score")]
    summaries={};rng=np.random.default_rng(16)
    for cohort in ("failure","control","official_task_failures"):
        group=[r for r in detailed if (r["cohort"]==cohort if cohort!="official_task_failures" else r["cohort"]=="failure" and not r["official_mask75_matched"])]
        metrics={}
        for key in columns:
            eligible=[r for r in group if r[key] is not None]
            if not eligible:continue
            values=np.array([r[key]-r["original_iou"] for r in eligible])
            means=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
            metrics[key]={"n":len(eligible),"mean":float(np.mean([r[key] for r in eligible])),"mean_delta":float(values.mean()),
                "delta95":list(map(float,np.quantile(means,[.025,.975]))),
                "mask75":sum(r[key]>=.75 for r in eligible),
                "repaired":sum(r["original_iou"]<.75<=r[key] for r in eligible),
                "broken":sum(r[key]<.75<=r["original_iou"] for r in eligible)}
        metrics["official_mask75_matched"]=sum(r["official_mask75_matched"] for r in group)
        metrics["o2o_native_rescues_also_bestbox"]=sum(r["o2o_best_native"]>=.75 and r["o2o_best_box_mask"]>=.75 for r in group)
        metrics["o2m_bestmask_rescues_also_bestbox"]=sum(r["o2m_own_bestmask_fixed"]>=.75 and r["o2m_own_bestbox_fixed"]>=.75 for r in group)
        metrics["o2m_mask_rescue_beyond_box_score"]=sum(r["o2m_own_bestmask_fixed"]>=.75 and max(r["o2m_own_bestbox_fixed"],r["o2m_own_bestscore_fixed"])<.75 for r in group)
        for comparator in ("o2m_own_bestbox_fixed","o2m_own_bestscore_fixed"):
            diff=np.array([r["o2m_own_bestmask_fixed"]-r[comparator] for r in group])
            means=diff[rng.integers(0,len(diff),(2000,len(diff)))].mean(1)
            metrics["mask_teacher_minus_"+comparator]={"mean":float(diff.mean()),"bootstrap95":list(map(float,np.quantile(means,[.025,.975])))}
        summaries[cohort]=metrics
    output=dict(full_gt_state_cross_task={k:dict(v) for k,v in table.items()},repair_directions=dict(repair),
                repair_directions_official_task_failed=dict(repair_task_failed),probe_branch_comparison=summaries,details=detailed,
                interpretation_limits=["Coefficients share the image prototype; fixed crop is original retained O2O box", "Same-position O2M mask only where already in decoded candidate pool", "Best-mask selection uses GT; best-box also uses GT; best-score still restricted to GT-assigned candidates", "Mask75 crossings do not imply AP gains, and fixed-slot failures may already be matched in official evaluation"])
    (args.out/"SUMMARY.json").write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in output.items() if k!="details"}),flush=True)


if __name__=="__main__":main()
