"""Full official-checkpoint COCO mask evaluation and fixed-slot failure inventory."""
import argparse
import contextlib
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def matching_size(edges):
    """Maximum cardinality bipartite matching, diagnostic only (not COCO matching)."""
    owners = {}
    def visit(gt, seen):
        for dt in edges[gt]:
            if dt in seen:
                continue
            seen.add(dt)
            if dt not in owners or visit(owners[dt], seen):
                owners[dt] = gt
                return True
        return False
    return sum(visit(g, set()) for g in range(len(edges)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    coco = COCO(str(args.root / "assets/datasets/coco/annotations/instances_val2017.json"))
    sources = ["RUN_c0797f4ee9644d459f825cfd76ea90d6", "RUN_8fe82201101b4e9d941cb07a3496c110"]
    predictions, image_ids, slots, provenance = [], [], {}, []
    for run in sources:
        directory = args.root / "experiments/mask_boundary_route_20260914/runs" / run
        ids = json.loads((directory / "image_ids.json").read_text())
        assert not set(ids) & set(image_ids), "Cache partitions overlap"
        image_ids.extend(ids)
        predictions.extend(json.loads((directory / "predictions_official_zero.json").read_text()))
        source_summary = json.loads((directory / "SUMMARY.json").read_text())
        assert source_summary["ultralytics"] == "8.4.100" and source_summary["branch"] == "one2one"
        provenance.append({"run_id": run, "counts": source_summary["counts"]["official_zero"], "parity": source_summary["parity"]})
        with (directory / "instance_records.csv").open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["variant"] != "official_zero":
                    continue
                aid = int(row["annotation_id"])
                assert aid not in slots
                slots[aid] = {k: (v if k in ("baseline_state", "state", "variant") else float(v)) for k, v in row.items()}
        print(json.dumps({"loaded": run, "images": len(ids), "slots_total": len(slots)}), flush=True)
    assert set(image_ids) == set(coco.imgs) and len(image_ids) == 5000
    by_image = defaultdict(list)
    for p in predictions:
        by_image[p["image_id"]].append(p)
    states, directions, opportunity = Counter(), Counter(), Counter()
    rows = []
    for n, image_id in enumerate(sorted(image_ids)):
        anns = [a for a in coco.imgToAnns[image_id] if not a.get("iscrowd", 0)]
        dt = by_image[image_id]
        gt_rles = [coco.annToRLE(a) for a in anns]
        ious = mu.iou([p["segmentation"] for p in dt], gt_rles, [0]*len(anns)) if dt and anns else np.zeros((len(dt), len(anns)))
        edges = []
        for j, a in enumerate(anns):
            slot = slots.get(a["id"])
            inds = [i for i,p in enumerate(dt) if p["category_id"] == a["category_id"]]
            best = max(inds, key=lambda i: ious[i,j]) if inds else None
            best_iou = float(ious[best,j]) if best is not None else 0.0
            edges.append([i for i in inds if ious[i,j] >= .75])
            state = slot["state"] if slot else "no_fixed_sameclass_box50_slot"
            direction = "no_slot" if slot is None else ("mask75_success" if slot["iou"]>=.75 else ("needs_own_pixels" if slot["recall"]<.75 else "fp_removal_admits_recovery"))
            states[state] += 1
            directions[(state, direction)] += 1
            opportunity["any_exported_sameclass_mask75"] += int(best_iou>=.75)
            opportunity["matched_slot_mask75"] += int(slot is not None and slot["iou"]>=.75)
            opportunity["slot_failed_but_other_mask75"] += int((slot is None or slot["iou"]<.75) and best_iou>=.75)
            row = dict(image_id=image_id, annotation_id=a["id"], category_id=a["category_id"],
                gt_area=a["area"], state=state, pixel_direction=direction,
                candidate_index=int(slot["candidate_index"]) if slot else -1,
                box_iou=slot["box_iou"] if slot else None,
                mask_iou=slot["iou"] if slot else None,
                coverage=slot["recall"] if slot else None,
                purity=slot["purity"] if slot else None,
                box_gt_support=slot["box_gt_support"] if slot else None,
                best_exported_mask_iou=best_iou,
                best_exported_position=int(best) if best is not None else -1)
            rows.append(row)
        opportunity["unique_candidate_maximum_mask75_matches"] += matching_size(edges)
        if (n+1)%500==0:
            print(json.dumps({"classification_images": n+1}), flush=True)
    with (out/"gt_inventory.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # Deterministic pilot; sample failure images without replacement, then same-class controls.
    rng=np.random.default_rng(20260916)
    failures=[r for r in rows if r["box_iou"] is not None and r["box_iou"]>=.75 and r["mask_iou"]<.75]
    successes=[r for r in rows if r["state"]=="box_good_mask_good"]
    rng.shuffle(failures)
    used_images=set(); sample=[]
    for f in failures:
        if f["image_id"] in used_images:
            continue
        candidates=[s for s in successes if s["category_id"]==f["category_id"] and s["image_id"] not in used_images and s["image_id"]!=f["image_id"]]
        if not candidates:
            continue
        control=min(candidates,key=lambda s:abs(np.log(max(s["gt_area"],1)/max(f["gt_area"],1))))
        pair=len(sample)//2
        sample.extend([{**f,"cohort":"failure","pair_id":pair},{**control,"cohort":"control","pair_id":pair}])
        used_images.update((f["image_id"],control["image_id"]))
        if len(sample)==64:
            break
    save(out/"probe_sample.json",sample)
    summary=dict(images=len(image_ids),ordinary_gt=len(rows),predictions=len(predictions),
        states=dict(states),pixel_directions=[dict(state=k[0],direction=k[1],count=v) for k,v in directions.items()],
        opportunity=dict(opportunity),provenance=provenance,
        limitations=["No full Box AP: source export lacks actual detection boxes", "Nonempty mask export only; original empty counts retained", "Fixed Box50 slots are not official segmentation matches", "Maximum bipartite matching is GT-assisted opportunity, not AP gain", "COCO val is exploratory"])
    save(out/"CLASSIFICATION.json",summary)
    print("Starting full COCOeval segm",flush=True)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        result=coco.loadRes(predictions)
        evaluator=COCOeval(coco,result,"segm")
        evaluator.params.imgIds=sorted(image_ids)
        evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
    (out/"cocoeval.log").write_text(log.getvalue(),encoding="utf-8")
    summary["segm_metrics"]=dict(zip(["AP","AP50","AP75","APS","APM","APL","AR1","AR10","AR100","ARS","ARM","ARL"],map(float,evaluator.stats)))
    # Official GT-side matching at IoU=.75, all area, maxDets100. Keep crowd/ignored flags.
    official_rows=[]
    t=int(np.argmin(abs(evaluator.params.iouThrs-.75)))
    for item in evaluator.evalImgs:
        if item is None or item["aRng"]!=evaluator.params.areaRng[0]:
            continue
        for j,aid in enumerate(item["gtIds"]):
            official_rows.append(dict(annotation_id=aid,image_id=item["image_id"],ignore=bool(item["gtIgnore"][j]),
                matched=bool(item["gtMatches"][t,j]),prediction_id=int(item["gtMatches"][t,j])))
    save(out/"official_gt_mask75.json",official_rows)
    ordinary=[x for x in official_rows if not x["ignore"]]
    summary["official_gt_mask75"]={"gt":len(ordinary),"matched":sum(x["matched"] for x in ordinary)}
    save(out/"SUMMARY.json",summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
