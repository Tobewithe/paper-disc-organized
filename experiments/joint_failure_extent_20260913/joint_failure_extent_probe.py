"""S069: remove only mask spill beyond a target's ideal rectangular extent.

This GT-assisted output intervention cannot fill missing pixels. It isolates
coarse extent spill from errors that persist INSIDE the target rectangle. It
does not simulate changing the pre-upsampling process_mask crop.
"""
from pathlib import Path
import argparse
import contextlib
import csv
import gc
import io
import json
import shutil
import time
from collections import defaultdict
import numpy as np
import pandas as pd

from mask_error_ap_probe import (ROOT, ANNOTATION, CENSUS, COCO, mu, dump,
    read_gz, encode, evaluate, save_csv, save_gz, paired_stats, sha)

SOURCE = ROOT / "diagnostics/mask_error_ap_20260913"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "patches").mkdir()
    start = time.monotonic()
    with CENSUS.open(encoding="utf-8-sig") as f:
        metadata = {int(r["annotation_id"]): r for r in csv.DictReader(f)}
    targets = [r for r in metadata.values() if r["state"] == "box_bad__mask_bad"]
    assert len(targets) == 6584
    byimage = defaultdict(list)
    for r in targets:
        byimage[int(r["image_id"])].append(r)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANNOTATION))
    ids = sorted(gt.imgs)
    dump(a.out / "protocol.json", dict(experiment="S069_JOINT_FAILURE_RECTANGULAR_EXTENT",
        question="How much joint box/mask failure is recoverable solely by removing spill outside ideal target rectangular extent?",
        decision="Large recovery prioritizes coarse spatial support; small recovery prioritizes within-rectangle discrimination or missing target pixels.",
        source=str(SOURCE), census_sha256=sha(CENSUS), script_sha256=sha(__file__),
        targets=6584, images=5000, ordinary_gt=36335,
        selection="Same fixed slots as S067 badbox_mask_gt, without post-intervention selection.",
        intervention="Final decoded binary mask intersected with integer tight bounding rectangle of full own COCO GT mask. GT mask pixels never filled. Scores/classes/order/candidate identities unchanged.",
        rationale="Tight rectangle derived from raster GT ensures it contains ALL target GT pixels; tests coarse extent leakage without silently clipping own mask due annotation-box rounding.",
        caveat="GT oracle, output-space cropping; NOT process_mask crop intervention, NOT exact achievable decoder upper bound, NOT an automatic method or box-only training effect. Explored val.",
        metric="Full official COCOeval AP; fixed-slot IoU rescue and errors saved separately; E4 grouping after global matching.",
        training=False, model_forward=False))
    shutil.copy2(__file__, a.out / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name("mask_error_ap_probe.py"), a.out / "mask_error_ap_probe.py")
    predictions, details = [], []
    for number, iid in enumerate(ids, 1):
        saved = read_gz(SOURCE / "images" / f"{iid}.json.gz")
        pp = saved["original"]
        patches = {}
        for row in byimage[iid]:
            aid, slot = int(row["annotation_id"]), int(row["prediction_slot"])
            pred = pp[slot]
            assert slot not in patches
            assert pred["category_id"] == int(row["category_id"]) and abs(pred["score"]-float(row["score"])) < 1e-12
            own = gt.annToMask(gt.anns[aid]).astype(bool)
            mask = mu.decode(pred["segmentation"]).astype(bool)
            y, x = np.nonzero(own)
            assert len(x)
            x0,x1,y0,y1=int(x.min()),int(x.max())+1,int(y.min()),int(y.max())+1
            cropped = np.zeros_like(mask)
            cropped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
            tp = int((mask & own).sum())
            assert int((cropped & own).sum()) == tp
            original_iou = tp / max(int((mask | own).sum()),1)
            assert abs(original_iou-float(row["mask_iou"])) < 1e-10
            new_iou = tp / max(int((cropped | own).sum()),1)
            fp = int((mask & ~own).sum())
            removed = int(mask.sum()-cropped.sum())
            assert removed >= 0 and new_iou >= original_iou-1e-12
            details.append(dict(annotation_id=aid,image_id=iid,slot=slot,density=row["mask_density"],
                area_bin=row["area_bin"],category_id=int(row["category_id"]),
                official_miss=row["official_mask75"]=="False", any_retained_mask75=row["any_kept_mask75"]=="True",
                support_proxy=float(row["box_gt_pixel_coverage"]),
                old_iou=original_iou,new_iou=new_iou,delta_iou=new_iou-original_iou,
                recovered75=new_iou>=.75,own_pixels=int(own.sum()),old_fp=fp,
                removed_fp=removed,removed_fp_fraction=removed/max(fp,1),
                residual_fp=fp-removed,own_fn=int((own & ~mask).sum()),
                tight_box=[x0,y0,x1,y1]))
            if removed:
                patches[slot]=encode(cropped)
        if patches:
            save_gz(a.out / "patches" / f"{iid}.json.gz", {str(k):v for k,v in patches.items()})
        predictions.extend(dict(pred,segmentation=patches[slot]) if slot in patches else pred for slot,pred in enumerate(pp))
        if number % 1000 == 0:
            dump(a.out / "progress.json", dict(stage="PREPARE", images=number, targets=len(details)))
    assert len(details) == 6584 and len(predictions) == 446097
    save_csv(a.out / "fixed_slot_effects.csv",details)
    dump(a.out / "progress.json", dict(stage="COCO_EVALUATE",targets=len(details)))
    print("All 6584 unchanged true-positive supports and original mask IoUs verified; evaluating AP",flush=True)
    summary, records = evaluate(gt,predictions,ids,metadata,"joint_failure_gt_rectangle")
    save_csv(a.out / "gt_recovery.csv",records)
    reference = json.loads((SOURCE / "original.json").read_text())["summary"]
    summary["delta_ap_points"] = 100*(summary["mask_ap"]-reference["mask_ap"])
    summary["delta_gap_points"] = 100*(summary["gap_low_high"]-reference["gap_low_high"])
    dump(a.out / "task_summary.json",summary)
    del predictions
    gc.collect()
    base=pd.read_csv(SOURCE / "original_gt.csv").to_dict("records")
    dump(a.out / "paired_stats.json",paired_stats(base,records))
    dd=pd.DataFrame(details)
    groups=[]
    for group in ["all","low","middle","high","undefined"]:
        z=dd if group=="all" else dd[dd.density.eq(group)]
        groups.append(dict(group=group,n=len(z),recovered75=int(z.recovered75.sum()),
            rescue_rate=float(z.recovered75.mean()),mean_old_iou=float(z.old_iou.mean()),mean_new_iou=float(z.new_iou.mean()),
            mean_delta_iou=float(z.delta_iou.mean()),mean_removed_fp_fraction=float(z.removed_fp_fraction.mean()),
            median_removed_fp_fraction=float(z.removed_fp_fraction.median()),
            mean_own_coverage=float((1-z.own_fn/z.own_pixels).mean())))
    save_csv(a.out / "fixed_slot_summary.csv",groups)
    dump(a.out / "COMPLETE.json",dict(status="COMPLETE",seconds=time.monotonic()-start,
        images=5000,targets=6584,task_summary_sha256=sha(a.out / "task_summary.json")))
    dump(a.out / "progress.json",dict(stage="COMPLETE"))
    print(json.dumps(summary),flush=True)
    print(pd.DataFrame(groups).round(4).to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
