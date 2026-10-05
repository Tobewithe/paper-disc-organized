"""Describe repair/damage with explicit denominators; no parameter fitting.

GT geometry and coverage are explanatory variables, not deployment features.
Prediction-only geometry is merged by explicit candidate ID when available.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def bucket(value, edges, labels):
    if value is None:
        return "unknown"
    return labels[int(np.searchsorted(edges, value, side="right"))]


def summarize(rows):
    n = len(rows)
    if not n:
        return dict(n=0)
    before = np.asarray([r["baseline_iou"] >= .75 for r in rows])
    after = np.asarray([r["iou"] >= .75 for r in rows])
    repaired = int((~before & after).sum())
    damaged = int((before & ~after).sum())
    failed = int((~before).sum())
    succeeded = int(before.sum())
    result = dict(n=n, baseline_failures=failed, baseline_successes=succeeded,
        repaired=repaired, damaged=damaged, net=repaired-damaged,
        repair_rate_among_failures_pct=100*repaired/failed if failed else None,
        damage_rate_among_successes_pct=100*damaged/succeeded if succeeded else None,
        net_per_all_matched_pct=100*(repaired-damaged)/n)
    for key in ("baseline_iou", "iou", "baseline_recall", "recall", "baseline_purity", "purity",
                "removed_tp", "removed_fp", "gt_area", "predicted_area", "gt_elongation",
                "gt_compactness", "grid_compactness", "mask_elongation", "mask_extent", "score"):
        values = np.array([r[key] for r in rows if r.get(key) is not None], dtype=float)
        if len(values):
            result[key] = dict(mean=float(values.mean()), median=float(np.median(values)),
                               q25=float(np.quantile(values,.25)), q75=float(np.quantile(values,.75)), n=len(values))
    result["removed_tp_over_gt_mean"] = float(np.mean([r["removed_tp"]/max(r["gt_area"],1) for r in rows]))
    result["removed_fp_over_gt_mean"] = float(np.mean([r["removed_fp"]/max(r["gt_area"],1) for r in rows]))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--geometry-cache", help="Reuse a previous objects.csv for GT geometry by annotation ID")
    p.add_argument("--summary", help="Recovered summary stored in a separate scoring Run")
    args = p.parse_args()
    start = time.perf_counter()
    source, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/"ANALYSIS.json").exists():
        raise RuntimeError("Use a fresh Run")
    previous = json.loads((Path(args.summary) if args.summary else source/"SUMMARY.json").read_text(encoding="utf-8"))
    ids = json.loads((source/"image_ids.json").read_text(encoding="utf-8"))
    image_index = {iid:i for i,iid in enumerate(ids)}
    coco = COCO(args.annotations)
    rows = []
    by_image = defaultdict(list)
    with (source/"instance_records.csv").open(newline="",encoding="utf-8") as handle:
        for value in csv.DictReader(handle):
            if value["variant"] != "smooth_gated":
                continue
            row = {k:(v if k in ("baseline_state","variant","state") else float(v)) for k,v in value.items()}
            for key in ("image_id","annotation_id","category_id","candidate_index"):
                row[key] = int(row[key])
            row["predicted_area"] = (row["gt_area"]*row["baseline_recall"]/row["baseline_purity"]
                                      if row["baseline_purity"]>0 else None)
            row["area_source"] = "recovered_from_exact_TP_and_purity" if row["predicted_area"] is not None else "unknown_TP_zero"
            before, after = row["baseline_iou"]>=.75, row["iou"]>=.75
            row["outcome"] = "repair" if not before and after else "damage" if before and not after else "stable_success" if before else "stable_failure"
            row["mechanism_cohort"] = int(row["box_iou"]>=.75 and row["baseline_recall"]>=.95 and not before)
            by_image[row["image_id"]].append(row)
            rows.append(row)
    candidates = source/"candidate_records.csv"
    if candidates.exists():
        lookup = {(r["image_id"],r["candidate_index"]):r for r in rows}
        with candidates.open(newline="",encoding="utf-8") as handle:
            for value in csv.DictReader(handle):
                key = (int(value["image_id"]),int(value["candidate_index"]))
                if key not in lookup:
                    continue
                row = lookup[key]
                area = float(value["baseline_area"])
                if row["predicted_area"] is not None and abs(area-row["predicted_area"])>1e-4:
                    raise RuntimeError("Candidate/GT area mismatch")
                row["predicted_area"] = area
                row["area_source"] = "direct_candidate_record"
                for field in ("grid_compactness","mask_extent","mask_elongation","score","removed_fraction"):
                    row[field] = float(value[field])
    cache = {}
    geometry_keys = ["gt_elongation","gt_extent","gt_compactness","gt_any_mask_neighbor_1px",
                     "gt_same_class_mask_neighbor_1px","gt_other_mask_overlap_fraction","max_gt_box_iou"]
    if args.geometry_cache:
        with Path(args.geometry_cache).open(newline="",encoding="utf-8") as handle:
            cache = {int(r["annotation_id"]):{k:float(r[k]) for k in geometry_keys} for r in csv.DictReader(handle)}
    totals = np.zeros(len(ids),dtype=np.int64)
    for ordinal,iid in enumerate(ids):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[iid],iscrowd=False))
        totals[image_index[iid]] = len(anns)
        selected = by_image[iid]
        if selected and all(r["annotation_id"] in cache for r in selected):
            for row in selected:
                row.update(cache[row["annotation_id"]])
        elif selected:
            masks = [mu.decode(coco.annToRLE(a)).astype(np.uint8) for a in anns]
            counts = np.sum(masks,axis=0,dtype=np.uint16)
            ann_index = {a["id"]:i for i,a in enumerate(anns)}
            boxes = np.array([a["bbox"] for a in anns],dtype=float)
            boxes[:,2:] += boxes[:,:2]
            class_counts = {}
            for row in selected:
                a = ann_index[row["annotation_id"]]
                mask = masks[a]
                area = int(mask.sum())
                _,_,w,h = mu.toBbox(mu.encode(np.asfortranarray(mask)))
                perimeter = (np.count_nonzero(mask[:,1:]!=mask[:,:-1]) + np.count_nonzero(mask[1:,:]!=mask[:-1,:]) +
                             int(mask[:,0].sum()+mask[:,-1].sum()+mask[0,:].sum()+mask[-1,:].sum()))
                others = counts-mask
                dilated = cv2.dilate(mask,np.ones((3,3),np.uint8))
                category = row["category_id"]
                if category not in class_counts:
                    class_counts[category] = np.sum([m for m,ann in zip(masks,anns) if ann["category_id"]==category],axis=0,dtype=np.uint16)
                row.update(gt_elongation=float(max(w,h)/max(min(w,h),1)),gt_extent=area/max(float(w*h),1),
                    gt_compactness=perimeter**2/max(4*np.pi*area,1),
                    gt_any_mask_neighbor_1px=int(np.any((others>0)&(dilated>0))),
                    gt_same_class_mask_neighbor_1px=int(np.any((class_counts[category]-mask>0)&(dilated>0))),
                    gt_other_mask_overlap_fraction=float(np.count_nonzero((others>0)&(mask>0))/max(area,1)))
                intersection = np.maximum(np.minimum(boxes[:,2:],boxes[a,2:])-np.maximum(boxes[:,:2],boxes[a,:2]),0).prod(1)
                box_areas = np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(1)
                overlaps = intersection/np.maximum(box_areas+box_areas[a]-intersection,1e-12)
                overlaps[a] = 0
                row["max_gt_box_iou"] = float(overlaps.max())
        if (ordinal+1)%250==0 or ordinal+1==len(ids):
            progress = dict(stage="object_geometry",images=ordinal+1,total=len(ids),elapsed_seconds=time.perf_counter()-start)
            save_json(out/"progress.json",progress)
            print(json.dumps(progress),flush=True)
    assert int(totals.sum()) == previous["ordinary_gt"]
    groups = defaultdict(list)
    for row in rows:
        tags = {
            "outcome":row["outcome"], "baseline_state":row["baseline_state"],
            "predicted_area":bucket(row["predicted_area"],[1024,2304,9216],["lt32sq","32sq_to48sq","48sq_to96sq","ge96sq"]),
            "baseline_recall":bucket(row["baseline_recall"],[.8,.9,.95],["lt.80",".80_to.90",".90_to.95","ge.95"]),
            "baseline_purity":bucket(row["baseline_purity"],[.8,.9,.95],["lt.80",".80_to.90",".90_to.95","ge.95"]),
            "gt_elongation":bucket(row["gt_elongation"],[2,4],["lt2","2_to4","ge4"]),
            "gt_compactness":bucket(row["gt_compactness"],[2,4],["lt2","2_to4","ge4"]),
            "gt_same_class_mask_neighbor_1px":str(int(row["gt_same_class_mask_neighbor_1px"])),
            "gt_any_mask_neighbor_1px":str(int(row["gt_any_mask_neighbor_1px"])),
            "category":str(row["category_id"]),
            "mechanism_cohort":str(row["mechanism_cohort"])}
        for key,edges,labels in (("mask_elongation",[2,4],["lt2","2_to4","ge4"]),
                                 ("grid_compactness",[2,4],["lt2","2_to4","ge4"]),
                                 ("mask_extent",[.3,.6],["lt.3",".3_to.6","ge.6"]),
                                 ("score",[.1,.5],["lt.1",".1_to.5","ge.5"])):
            if key in row:
                tags[key] = bucket(row[key],edges,labels)
        tags["area_x_recall"] = tags["predicted_area"]+"/"+tags["baseline_recall"]
        tags["area_x_gt_neighbor"] = tags["predicted_area"]+"/"+tags["gt_same_class_mask_neighbor_1px"]
        for dimension,value in tags.items():
            groups[(dimension,value)].append(row)
    overall = summarize(rows)
    groups_report = {dimension:{} for dimension,_ in groups}
    for (dimension,label),items in groups.items():
        groups_report[dimension][label] = summarize(items)
    before_per_image = np.zeros(len(ids),dtype=np.int64)
    net_per_image = np.zeros(len(ids),dtype=np.int64)
    for row in rows:
        i = image_index[row["image_id"]]
        before_per_image[i] += int(row["baseline_iou"]>=.75)
        net_per_image[i] += int(row["iou"]>=.75)-int(row["baseline_iou"]>=.75)
    rng = np.random.default_rng(20260915)
    sampled = []
    for _ in range(2000):
        draw = rng.integers(0,len(ids),size=len(ids))
        sampled.append(100*net_per_image[draw].sum()/totals[draw].sum())
    columns = sorted({key for row in rows for key in row})
    with (out/"objects.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    flat = []
    for dimension,table in groups_report.items():
        for label,stats in table.items():
            flat.append(dict(dimension=dimension,group=label,**{k:v for k,v in stats.items() if not isinstance(v,dict)}))
    with (out/"group_rates.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    payload=dict(run_id=os.environ.get("RESEARCH_RUN_ID"),study_id="STUDY_8fb3468ebb704682a2225ebed0e16206",
        source_run_id=previous.get("run_id"),images=len(ids),ordinary_gt=int(totals.sum()),
        matched=len(rows),unmatched=int(totals.sum())-len(rows),overall=overall,
        r75=dict(baseline=100*float(before_per_image.sum())/totals.sum(),
            calibrated=100*float(before_per_image.sum()+net_per_image.sum())/totals.sum(),
            delta_pp=100*float(net_per_image.sum())/totals.sum(),
            paired_image_bootstrap_95ci_pp=[float(v) for v in np.quantile(sampled,[.025,.975])]),
        groups=groups_report,candidate_features_available=candidates.exists(),
        definitions=dict(gt_neighbor="non-crowd masks touch/overlap within 1 original-image pixel in Chebyshev distance; not causal occlusion",
            compactness="four-connected perimeter squared / (4*pi*area); grid-oriented, not rotation invariant",
            recovered_predicted_area="GT_area*baseline_recall/baseline_purity; exactly prediction area when TP>0; unknown otherwise",
            denominator="within each matched group repair / baseline failures, damage / baseline successes, net / all; overall R75 includes unmatched ordinary GT",
            bootstrap="2000 paired image resamples; R75 only, not AP CI"),
        limitations=["exploratory groups, no parameter fitting", "GT geometry/coverage are explanatory only",
            "group differences do not demonstrate causality; class and size may confound", "bbox elongation is only a shape proxy, not actual local thickness"],
        elapsed_seconds=time.perf_counter()-start)
    save_json(out/"ANALYSIS.json",payload)
    save_json(out/"progress.json",dict(stage="completed",images=len(ids),total=len(ids)))
    print(json.dumps({k:v for k,v in payload.items() if k not in ("groups",)},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
