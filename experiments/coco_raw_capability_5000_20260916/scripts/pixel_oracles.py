"""TIDE-inspired independent mask repairs on fixed, correctly localized outputs.

Each variant starts from the same normal predictions. Pair identities are the
unmodified Box75 COCO true positives. Scores, labels, boxes and counts are fixed.
This estimates repair value, not a learnable method's achieved performance.
"""
import argparse
import contextlib
import gc
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--raw",type=Path,required=True)
    ap.add_argument("--evaluation",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    sys.path.insert(0,str(a.root/"shared/vendor/ultralytics_8_4_100"))
    import torch
    from ultralytics.utils import ops
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mu
    torch.set_num_threads(4)
    a.out.mkdir(parents=True,exist_ok=True)
    while not (a.evaluation/"COMPLETE.json").exists():
        status=json.loads((a.evaluation/"run.json").read_text()).get("status")
        if status in ("failed","error","interrupted","cancelled"):
            raise RuntimeError(f"Upstream evaluation {status}")
        time.sleep(2)
    start=time.monotonic()
    baseline=json.loads((a.evaluation/"SUMMARY.json").read_text())
    preds=json.loads((a.evaluation/"predictions_with_identity.json").read_text())
    rows=[json.loads(line) for line in (a.evaluation/"gt_joined.jsonl").read_text().splitlines()]
    by_image=defaultdict(list)
    for row in rows:
        if row["bbox_coco75"]:
            by_image[row["image_id"]].append(row)
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO(str(a.root/"assets/datasets/coco/annotations/instances_val2017.json"))
    modes=("remove_fp","fill_fn_within_support","perfect_within_support","perfect_full_gt")
    replacements={mode:{} for mode in modes}
    metrics=[]
    for number,(image_id,group) in enumerate(sorted(by_image.items())):
        with np.load(a.raw/"images"/f"{image_id:012d}.npz") as z:
            shape=tuple(z["input_shape"].tolist())
            orig=tuple(z["original_shape"].tolist())
            ids=[r["bbox_raw_id"] for r in group]
            boxes=torch.as_tensor(z["boxes_input"][ids],device="cuda")
        with torch.inference_mode():
            support=ops.crop_mask(torch.ones((len(ids),*shape),device="cuda"),boxes)
            support=ops.scale_masks(support[None],orig)[0].byte().cpu().numpy().astype(bool)
        for row,sp in zip(group,support):
            pid=row["bbox_prediction_id"]
            p=preds[pid-1]
            assert p["raw_id"]==row["bbox_raw_id"] and p["category_id"]==row["category_id"]
            pred=mu.decode(p["segmentation"]).astype(bool)
            gt=coco.annToMask(coco.anns[row["annotation_id"]]).astype(bool)
            assert not (pred & ~sp).any(), "Support reconstruction differs from raw decoder"
            variants=dict(remove_fp=pred&gt,fill_fn_within_support=pred|(gt&sp),
                          perfect_within_support=gt&sp,perfect_full_gt=gt)
            result=dict(image_id=image_id,annotation_id=row["annotation_id"],prediction_id=pid,
                        area_group=row["area_group"],geometry_state=row["geometry_state"],
                        original_mask_iou=float((pred&gt).sum()/max((pred|gt).sum(),1)),
                        support=float((gt&sp).sum()/gt.sum()))
            for mode,mask in variants.items():
                rle=mu.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle["counts"]=rle["counts"].decode("ascii")
                replacements[mode][pid]=rle
                result[mode]=float((mask&gt).sum()/max((mask|gt).sum(),1))
            metrics.append(result)
        if (number+1)%500==0:
            progress=dict(stage="constructing_repairs",images=number+1,instances=len(metrics),elapsed_s=round(time.monotonic()-start,2))
            print(json.dumps(progress),flush=True);dump(a.out/"progress.json",progress)
    dump(a.out/"fixed_pairs.json",metrics)
    dump(a.out/"replacement_masks.json",replacements)
    results={}
    for mode in modes:
        print("Evaluating "+mode,flush=True)
        dump(a.out/"progress.json",dict(stage="cocoeval_"+mode,elapsed_s=round(time.monotonic()-start,2)))
        changed=replacements[mode]
        dets=[dict(image_id=p["image_id"],category_id=p["category_id"],score=p["score"],
                   segmentation=changed.get(i+1,p["segmentation"])) for i,p in enumerate(preds)]
        dt=coco.loadRes(dets)
        ev=COCOeval(coco,dt,"segm")
        ev.params.imgIds=sorted(coco.imgs)
        ev.evaluate();ev.accumulate();ev.summarize()
        names=("AP","AP50","AP75","APS","APM","APL","AR1","AR10","AR100","ARS","ARM","ARL")
        results[mode]=dict(zip(names,[float(v) for v in ev.stats]))
        results[mode]["delta_AP_points"]=100*(results[mode]["AP"]-baseline["coco"]["segm"]["AP"])
        dump(a.out/(mode+".json"),results[mode])
        del ev,dt,dets
        gc.collect()
    summary=dict(baseline=baseline["coco"]["segm"],results=results,edited_instances=len(metrics),
        edited_images=len(by_image),source_raw=a.raw.name,source_evaluation=a.evaluation.name,
        protocol="Fixed Box75 COCO true-positive pairs; each repair independently starts from baseline; normal labels/scores/boxes/count unchanged.",
        limits=["Uses evaluation GT: diagnostic repair value, not a proposed method or generalization result.",
          "The cohort covers already-correctly-localized normal outputs, not all failures or full-raw maxima.",
          "Individual delta AP values are not additive.",
          "perfect_full_gt deliberately relaxes crop support; other variants preserve it."])
    dump(a.out/"SUMMARY.json",summary)
    dump(a.out/"COMPLETE.json",dict(images=5000,edited_instances=len(metrics),elapsed_s=round(time.monotonic()-start,2)))
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
