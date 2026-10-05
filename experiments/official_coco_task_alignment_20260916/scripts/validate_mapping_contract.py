"""Exercise mapping edge cases, then explain existing fixed-pair/COCO discrepancies."""
import argparse
import contextlib
import csv
import io
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.root/"shared/evaluation"))
    from gt_prediction_mapping import associate_instances,pixel_metrics
    tests=[]
    def run(name,boxes,masks,matches,gtcats=None,pcats=None,ignored=(),eligible=None):
        p=len(boxes);g=len(boxes[0])
        result=associate_instances(gt_ids=list(range(1,g+1)),gt_categories=gtcats or [1]*g,
            prediction_ids=list(range(101,101+p)),prediction_categories=pcats or [1]*p,
            scores=np.linspace(.9,.1,p),box_ious=boxes,mask_ious=masks,task_matches=matches,
            eligible_prediction_ids=eligible if eligible is not None else list(range(101,101+p)),ignored_prediction_ids=ignored)
        tests.append(name);return result
    r=run("box-best candidate cannot override actual mask success",[[.9],[.7]],[[.6],[.8]],{1:102})
    assert r["gt_rows"][0]["prediction_id"]==102 and r["gt_rows"][0]["box_iou"]==.7
    r=run("good box and empty/failed mask remain linked",[[.9]],[[0]],{})
    assert r["gt_rows"][0]["state"]=="box_good_mask_bad" and not r["gt_rows"][0]["task_mask_success"]
    r=run("bad box does not exclude a correct mask",[[.1]],[[.9]],{1:101})
    assert r["gt_rows"][0]["state"]=="box_bad_mask_good"
    r=run("one prediction cannot count for two GTs",[[.8,.9]],[[.8,.85]],{2:101})
    assert r["gt_rows"][0]["prediction_id"] is None and r["gt_rows"][1]["prediction_id"]==101
    r=run("wrong category cannot silently become a success",[[.9]],[[.9]],{},pcats=[2])
    assert r["gt_rows"][0]["association_basis"]=="unassociated"
    r=run("ignored prediction cannot be borrowed",[[.9]],[[.6]],{},ignored=[101])
    assert r["gt_rows"][0]["prediction_id"] is None and r["prediction_rows"][0]["task_status"]=="ignored"
    r=run("out-of-budget prediction cannot be borrowed",[[.9]],[[.9]],{},eligible=[])
    assert r["gt_rows"][0]["prediction_id"] is None
    r=run("diagnostic association is not a Mask75 true positive",[[.6]],[[.65]],{})
    assert r["gt_rows"][0]["association_basis"]=="diagnostic_mask50" and r["prediction_rows"][0]["task_status"]=="false_positive_at_task_threshold"
    r=run("duplicate remains visible on prediction side",[[.9],[.8]],[[.9],[.8]],{1:101})
    assert r["prediction_rows"][1]["duplicate_related_gt_ids"]==[1]
    px=pixel_metrics([[1,1],[0,0]],[[1,0],[1,0]])
    assert (px["tp"],px["fp"],px["fn"],px["target_coverage"],px["prediction_purity"])==(1,1,1,.5,.5)
    tests.append("same-mask pixel accounting")
    print(json.dumps({"contract_tests_passed":len(tests)}),flush=True)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mu
    cache=a.root/"experiments/official_coco_task_alignment_20260916/runs/RUN_1bea3cce89f441ec885b737feaa26989"
    official={int(x["annotation_id"]):x for x in json.loads((cache/"official_gt_mask75.json").read_text()) if not x["ignore"]}
    inventory=list(csv.DictReader((cache/"gt_inventory.csv").open(encoding="utf-8")))
    targets=[r for r in inventory if r["mask_iou"] and float(r["mask_iou"])>=.75 and not official[int(r["annotation_id"])]["matched"]]
    image_ids=sorted({int(r["image_id"]) for r in targets});wanted=set(image_ids)
    coco=COCO(str(a.root/"assets/datasets/coco/annotations/instances_val2017.json"))
    preds=[];global_id=0
    for rid in ("RUN_c0797f4ee9644d459f825cfd76ea90d6","RUN_8fe82201101b4e9d941cb07a3496c110"):
        source=a.root/"experiments/mask_boundary_route_20260914/runs"/rid/"predictions_official_zero.json"
        for p in json.loads(source.read_text()):
            global_id+=1
            if p["image_id"] in wanted:
                preds.append({**p,"source_prediction_id":global_id})
    with contextlib.redirect_stdout(io.StringIO()) as output:
        result=coco.loadRes(preds);ev=COCOeval(coco,result,"segm");ev.params.imgIds=image_ids
        ev.evaluate()
    (a.out/"cocoeval_audit.log").write_text(output.getvalue(),encoding="utf-8")
    t=int(np.argmin(abs(ev.params.iouThrs-.75)));statuses={};gt_matches={}
    for item in ev.evalImgs:
        if item is None or item["aRng"]!=ev.params.areaRng[0]:continue
        for j,did in enumerate(item["dtIds"]):
            statuses[did]=dict(matched_gt=int(item["dtMatches"][t,j]),ignored=bool(item["dtIgnore"][t,j]))
        for j,gid in enumerate(item["gtIds"]):
            gt_matches[gid]=int(item["gtMatches"][t,j])
    by_image=defaultdict(list)
    for p in result.anns.values():by_image[p["image_id"]].append(p)
    rows=[];counts=Counter()
    for row in targets:
        gid=int(row["annotation_id"]);image_id=int(row["image_id"]);gt=coco.anns[gid]
        assert gt_matches[gid]==0,"Subset matching changed a previously unmatched GT"
        ps=[p for p in by_image[image_id] if p["category_id"]==gt["category_id"]]
        ious=mu.iou([p["segmentation"] for p in ps],[coco.annToRLE(gt)],[0])[:,0]
        witnesses=[]
        for p,value in zip(ps,ious):
            if value<.75:continue
            status=statuses.get(p["id"])
            reason=("outside_max_detections" if status is None else "ignored_prediction" if status["ignored"] else
                    "prediction_assigned_to_other_gt" if status["matched_gt"] and status["matched_gt"]!=gid else "unexplained")
            witnesses.append(dict(source_prediction_id=p["source_prediction_id"],score=p["score"],mask_iou=float(value),reason=reason,
                                  matched_gt=status["matched_gt"] if status else None))
        reason="no_reproduced_mask75_candidate" if not witnesses else "+".join(sorted({x["reason"] for x in witnesses}))
        counts[reason]+=1
        rows.append(dict(annotation_id=gid,image_id=image_id,old_state=row["state"],old_mask_iou=float(row["mask_iou"]),reason=reason,witnesses=witnesses))
    summary=dict(contract_tests_passed=tests,discrepancy_gt=len(targets),images=len(image_ids),reason_counts=dict(counts),instances=rows,
                 scope="Only mapping-contract validation and explanation of 26 previously reported discrepant GTs; no new model-performance claim")
    (a.out/"SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="instances"}),flush=True)


if __name__=="__main__":main()
