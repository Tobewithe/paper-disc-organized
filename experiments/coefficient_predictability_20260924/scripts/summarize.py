"""Paired image-level probe comparisons and actual-output membership audit."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
import torch


def main(a):
    records=json.loads((a.probe/"VAL_ROWS.json").read_text())
    oracle=json.loads((a.oracle/"ROWS.json").read_text())
    if a.extra:
        extra={r["annotation_id"]:r for r in json.loads((a.extra/"VAL_ROWS.json").read_text())}
        for r in records:
            other=extra[r["annotation_id"]]
            assert other["raw_id"]==r["raw_id"] and abs(other["iou"]["original"]-r["iou"]["original"])<1e-8
            r["iou"].update(other["iou"])
    coco=COCO(str(a.annotations))
    cat_to_cls={c:i for i,c in enumerate(sorted(coco.cats))}
    groups=defaultdict(list)
    for r in records:
        groups[r["image_id"]].append(r)
    for iid,rows in groups.items():
        image=torch.load(a.bank/"images"/f"{iid:012d}.pt",weights_only=False,map_location="cpu")
        ids=image["top_ids"].flatten().tolist()
        classes=image["top_classes"].flatten().tolist()
        scores=image["top_scores"].flatten().tolist()
        retained={(raw_id,cls) for raw_id,cls,score in zip(ids,classes,scores) if score>.001}
        for r in rows:
            cls=cat_to_cls[coco.anns[r["annotation_id"]]["category_id"]]
            r["retained_with_gt_class"]=(r["raw_id"],cls) in retained
    rng=np.random.default_rng(20260924)
    arms=list(records[0]["iou"])
    result={"subgroups":{},"paired":{},"oracle_actual_output":{}}
    masks={"all_mapped":lambda r:True,"retained_gt_class":lambda r:r["retained_with_gt_class"],
           "not_retained_gt_class":lambda r:not r["retained_with_gt_class"],
           "box75":lambda r:r["box_iou"]>=.75,
           "box75_retained_gt_class":lambda r:r["box_iou"]>=.75 and r["retained_with_gt_class"]}
    for name,keep in masks.items():
        rr=[r for r in records if keep(r)]
        result["subgroups"][name]={}
        for arm in arms:
            result["subgroups"][name][arm]=dict(n=len(rr),mean_iou=float(np.mean([r["iou"][arm] for r in rr])) if rr else None,
                mask75=sum(r["iou"][arm]>=.75 for r in rr),
                repairs=sum(r["iou"]["original"]<.75<=r["iou"][arm] for r in rr),
                damages=sum(r["iou"][arm]<.75<=r["iou"]["original"] for r in rr))
        for comparator in [arm for arm in ("mlp","mlp_large","condition_mean","condition_shuffle") if arm in arms]:
            by_image=defaultdict(list)
            for r in rr:
                by_image[r["image_id"]].append([r["iou"]["condition_true"]-r["iou"][comparator],
                    int(r["iou"]["condition_true"]>=.75)-int(r["iou"][comparator]>=.75)])
            if not by_image:
                continue
            counts=np.array([len(v) for v in by_image.values()])
            values=np.array([np.sum(v,axis=0) for v in by_image.values()])
            draws=rng.integers(0,len(values),size=(2000,len(values)))
            boot=values[draws].sum(1)/counts[draws].sum(1)[:,None]
            result["paired"][name+":true-vs-"+comparator]=dict(
                mean_iou_delta=float(values[:,0].sum()/counts.sum()),
                iou_ci=np.quantile(boot[:,0],[.025,.975]).tolist(),
                mask75_rate_delta=float(values[:,1].sum()/counts.sum()),
                mask75_ci=np.quantile(boot[:,1],[.025,.975]).tolist())
    membership={r["annotation_id"]:r["retained_with_gt_class"] for r in records}
    for name,predicate in {"final_failed":lambda r:r["final_iou_before"]<.75,
                           "final_passed":lambda r:r["final_iou_before"]>=.75,
                           "retained_final_failed":lambda r:r["final_iou_before"]<.75 and membership[r["annotation_id"]]}.items():
        rr=[r for r in oracle if predicate(r)]
        result["oracle_actual_output"][name]=dict(n=len(rr),
            before=float(np.mean([r["final_iou_before"] for r in rr])) if rr else None,
            after=float(np.mean([r["final_iou_after"] for r in rr])) if rr else None,
            mask75_after=sum(r["final_iou_after"]>=.75 for r in rr))
    result["scope"]="Same raw candidate geometry matching; retained subset is not official score-ordered GT matching or AP. Image bootstrap is exploratory, single seed."
    (a.out/"SUMMARY.json").write_text(json.dumps(result,indent=2))
    (a.out/"MERGED_ROWS.json").write_text(json.dumps(records,indent=2))
    (a.out/"COMPLETE.json").write_text(json.dumps({"instances":len(records)}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("bank","probe","oracle","annotations","out"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--extra",type=Path)
    main(p.parse_args())
