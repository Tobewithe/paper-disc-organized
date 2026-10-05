"""Streaming paired decoder comparisons with instance transitions and COCO AP.

Run through research runner; --output must be unique. Prediction indices are
kept explicitly (never reconstructed from rounded category/score tuples).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--package-root", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--branch", choices=["one2one", "one2many"], default="one2one")
    p.add_argument("--variants", default="official_zero,official_gated,official_global,official_erode,smooth_gated,input_area_gated,legacy_zero,legacy_gated")
    p.add_argument("--chunk", type=int, default=24)
    p.add_argument("--benchmark-repeats", type=int, default=10)
    return p.parse_args()


def main():
    args = arguments()
    sys.path.insert(0, args.package_root)
    import numpy as np
    import torch
    from pycocotools import mask as mu
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.utils import ops
    from mask_calibration import input_logits, export_masks, apply_threshold, erode_small, process_mask_calibrated

    torch.set_num_threads(4)
    if ultralytics.__version__ != "8.4.100":
        raise RuntimeError(f"Wrong package: {ultralytics.__version__}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "SUMMARY.json").exists():
        raise RuntimeError("Completed output exists; use a fresh run directory")
    coco = COCO(args.annotations)
    all_ids = sorted(coco.getImgIds())
    image_ids = all_ids[args.start:args.start + args.limit]
    paths = [str(Path(args.images) / coco.imgs[i]["file_name"]) for i in image_ids]
    assert all(Path(p).is_file() for p in paths)
    (out / "image_ids.json").write_text(json.dumps(image_ids), encoding="utf-8")
    # Ultralytics treats a Python list as an in-memory batch. A source list file
    # uses LoadImagesAndVideos and honors batch=1 for bounded GPU memory.
    source_list = out / "image_paths.txt"
    source_list.write_text("\n".join(paths) + "\n", encoding="utf-8")
    variants = args.variants.split(",")
    assert "official_zero" in variants and "official_gated" in variants
    predictions = {key: [] for key in variants}
    counts = {key: Counter() for key in variants}
    transitions = {key: Counter() for key in variants}
    pixel_sums = {key: defaultdict(lambda: Counter()) for key in variants}
    timings = []
    parity = Counter()
    processed = 0
    start_time = time.perf_counter()
    gt_seen = 0
    record_handle = (out / "instance_records.csv").open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(record_handle, fieldnames=[
        "image_id", "annotation_id", "category_id", "candidate_index", "box_iou",
        "gt_area", "box_gt_support", "baseline_state", "baseline_iou", "baseline_recall",
        "baseline_purity", "variant", "state", "iou", "recall", "purity",
        "removed_tp", "removed_fp", "improvement_margin"])
    writer.writeheader()

    def json_save(name, data):
        target = out / name
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temp, target)

    def encode_stack(binary):
        array = binary.detach().cpu().numpy()
        rles = mu.encode(np.asfortranarray(array.transpose(1, 2, 0)))
        for rle in rles:
            rle["counts"] = rle["counts"].decode("ascii")
        return rles

    def state(box_iou, mask_iou, support):
        if mask_iou >= .75:
            return "box_good_mask_good" if box_iou >= .75 else "box_bad_mask_good"
        if box_iou < .75:
            return "box_bad_mask_bad"
        return "box_good_supported_mask_bad" if support >= .95 else "box_good_support_limited_mask_bad"

    def box_overlaps(boxes, gt_boxes):
        lt = np.maximum(boxes[:, None, :2], gt_boxes[None, :, :2])
        rb = np.minimum(boxes[:, None, 2:], gt_boxes[None, :, 2:])
        inter = np.maximum(rb-lt, 0).prod(-1)
        return inter / np.maximum((boxes[:,2:]-boxes[:,:2]).clip(0).prod(-1)[:,None] +
                                  (gt_boxes[:,2:]-gt_boxes[:,:2]).clip(0).prod(-1)[None] - inter, 1e-12)

    def pixel_metrics(rle, gt_rle):
        pa, ga = float(mu.area(rle)), float(mu.area(gt_rle))
        j = float(mu.iou([rle], [gt_rle], [0])[0,0])
        tp = j * (pa + ga) / (1+j)
        return j, tp / max(ga,1), tp / max(pa,1), tp, pa-tp

    def diagnose(image_id, pred, boxes, rles, areas):
        nonlocal gt_seen
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False))
        gt_seen += len(anns)
        if not anns:
            return
        gt_boxes = np.array([a["bbox"] for a in anns], dtype=float).reshape(-1,4)
        gt_boxes[:,2:] += gt_boxes[:,:2]
        ious = box_overlaps(boxes, gt_boxes)
        available = set(range(len(anns)))
        matched = {}
        for i in np.argsort(-pred[:,4], kind="stable"):
            if not areas["official_zero"][i]:
                continue
            candidates = [g for g in sorted(available) if anns[g]["category_id"] == cat_ids[int(pred[i,5])] and ious[i,g] >= .5]
            if candidates:
                g = max(candidates, key=lambda g: ious[i,g])
                matched[g] = int(i)
                available.remove(g)
        for g,a in enumerate(anns):
            if g not in matched:
                for key in variants:
                    transitions[key][("no_box50_slot", "no_box50_slot")] += 1
                continue
            i = matched[g]
            truth = coco.annToRLE(a)
            gt_mask = mu.decode(truth)
            h,w = gt_mask.shape
            x1,y1,x2,y2 = boxes[i]
            support = float(gt_mask[max(0,int(np.ceil(y1))):min(h,max(0,int(np.ceil(y2)))),
                                         max(0,int(np.ceil(x1))):min(w,max(0,int(np.ceil(x2))))].sum()) / max(float(mu.area(truth)),1)
            base = pixel_metrics(rles["official_zero"][i], truth)
            s0 = state(ious[i,g], base[0], support)
            for key in variants:
                val = pixel_metrics(rles[key][i], truth)
                s1 = state(ious[i,g], val[0], support)
                transitions[key][(s0,s1)] += 1
                # Mechanism group uses actual mask recall, distinct from box support.
                cohort = "coverage_sufficient_failure" if ious[i,g]>=.75 and base[1]>=.95 and base[0]<.75 else "other"
                group = pixel_sums[key][cohort]
                group.update(n=1, iou=val[0], recall=val[1], purity=val[2],
                             repaired=int(base[0]<.75<=val[0]), harmed=int(val[0]<.75<=base[0]))
                u, v = base[3]-val[3], base[4]-val[4]
                # For subset masks, J' > J iff T*v - (G+F)*u > 0.
                margin = base[3]*v - (float(mu.area(truth))+base[4])*u
                writer.writerow(dict(image_id=image_id, annotation_id=a["id"], category_id=a["category_id"],
                    candidate_index=i, box_iou=ious[i,g], gt_area=float(mu.area(truth)), box_gt_support=support,
                    baseline_state=s0, baseline_iou=base[0], baseline_recall=base[1], baseline_purity=base[2],
                    variant=key, state=s1, iou=val[0], recall=val[1], purity=val[2],
                    removed_tp=u, removed_fp=v, improvement_margin=margin))

    model = YOLO(args.weights)
    model.model.model[-1].end2end = args.branch == "one2one"
    cat_lookup = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    cat_ids = {int(i): cat_lookup[name] for i,name in model.names.items()}
    parity["official_input_mask_mismatches"] = 0
    bench_done = False

    class StreamingPredictor(SegmentationPredictor):
        def construct_result(self, pred, img, orig_img, img_path, proto):
            nonlocal processed, bench_done
            image_id = int(Path(img_path).stem)
            shape, orig = tuple(img.shape[2:]), tuple(orig_img.shape[:2])
            boxes = ops.scale_boxes(shape, pred[:,:4].clone(), orig).cpu().numpy()
            pred_cpu = pred.detach().cpu().numpy()
            image_rles = {key: [] for key in variants}
            for first in range(0, len(pred), args.chunk):
                p = pred[first:first+args.chunk]
                logits = input_logits(proto, p[:,6:], p[:,:4], shape)
                binary = (logits>0).byte()
                # Only a few real images need this exact implementation parity check.
                if processed < 3:
                    ref = ops.process_mask(proto,p[:,6:],p[:,:4],shape,upsample=True)
                    diff = int((ref != binary).sum().item())
                    parity["official_input_mask_mismatches"] += diff
                    parity["checked_masks"] += len(p)
                    if diff:
                        raise RuntimeError(f"Official decoder parity failed: {diff} pixels")
                    native = ops.process_mask_native(proto,p[:,6:],p[:,:4],shape)
                    parity["native_input_mask_mismatches"] += int((native != binary).sum().item())
                base = export_masks(binary, orig)
                area = base.sum((1,2)).float()
                decoded = {"official_zero": base}
                for key in variants:
                    if key == "official_gated":
                        decoded[key] = export_masks(apply_threshold(logits,area), orig)
                    elif key == "official_global":
                        decoded[key] = export_masks(apply_threshold(logits,area,"global"), orig)
                    elif key == "smooth_gated":
                        decoded[key] = export_masks(apply_threshold(logits,area,"smooth"), orig)
                    elif key == "input_area_gated":
                        decoded[key] = export_masks(apply_threshold(logits,binary.sum((1,2)).float()), orig)
                    elif key == "official_erode":
                        decoded[key] = erode_small(base,area)
                if "legacy_zero" in variants or "legacy_gated" in variants:
                    original_logits = ops.scale_masks(logits[None], orig)[0]
                    legacy = (original_logits>0).byte()
                    if "legacy_zero" in variants:
                        decoded["legacy_zero"] = legacy
                    if "legacy_gated" in variants:
                        decoded["legacy_gated"] = apply_threshold(original_logits,legacy.sum((1,2)).float())
                for key in variants:
                    image_rles[key].extend(encode_stack(decoded[key]))
                if not bench_done and len(p)>=4:
                    for key in ("official_decoder", "calibrated_decoder"):
                        samples=[]
                        for repeat in range(args.benchmark_repeats+3):
                            torch.cuda.synchronize()
                            before=time.perf_counter()
                            if key=="official_decoder":
                                output = ops.process_mask(proto,p[:,6:],p[:,:4],shape,upsample=True)
                            else:
                                output = process_mask_calibrated(proto,p[:,6:],p[:,:4],shape,orig)
                            torch.cuda.synchronize()
                            if repeat>=3:
                                samples.append((time.perf_counter()-before)*1000)
                        timings.append(dict(operation=key, masks=len(p), input_shape=shape, original_shape=orig,
                                            median_ms=float(np.median(samples)), samples_ms=samples))
                    bench_done=True
            areas = {key: np.array([float(mu.area(r)) for r in image_rles[key]]) for key in variants}
            for key in variants:
                for i,rle in enumerate(image_rles[key]):
                    counts[key]["candidates"]+=1
                    if not areas[key][i]:
                        counts[key]["empty"]+=1
                        continue
                    predictions[key].append(dict(image_id=image_id,category_id=cat_ids[int(pred_cpu[i,5])],
                                                 score=float(pred_cpu[i,4]),segmentation=rle))
            diagnose(image_id,pred_cpu,boxes,image_rles,areas)
            processed+=1
            if processed%25==0 or processed==len(image_ids):
                record_handle.flush()
                progress=dict(stage="inference",images=processed,total=len(image_ids),elapsed_seconds=time.perf_counter()-start_time)
                json_save("progress.json",progress)
                print(json.dumps(progress),flush=True)
            # RLEs, not float tensors, survive each image. Results here serve only
            # the generator contract; all predictions above keep explicit slots.
            result_pred=pred[:,:6].clone()
            result_pred[:,:4]=torch.as_tensor(boxes,device=pred.device)
            return Results(orig_img,path=img_path,names=self.model.names,boxes=result_pred,masks=None)

    for _ in model.predict(source=str(source_list), imgsz=640, conf=.001, max_det=300, device=0,
                           batch=1, half=False, stream=True, verbose=False, save=False,
                           predictor=StreamingPredictor):
        pass
    record_handle.close()
    assert processed == len(image_ids)
    for key,items in predictions.items():
        # Save before COCOeval, whose loadRes mutates records.
        (out / f"predictions_{key}.json").write_text(json.dumps(items),encoding="utf-8")
    metrics={}
    for key in variants:
        print(f"COCOeval {key}",flush=True)
        json_save("progress.json",dict(stage="cocoeval",variant=key,images=processed,total=len(image_ids)))
        with contextlib.redirect_stdout(io.StringIO()) as logs:
            # Each variant is loaded exactly once from canonical segmentation fields.
            result=coco.loadRes(predictions[key])
            evaluator=COCOeval(coco,result,"segm")
            evaluator.params.imgIds=image_ids
            evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        metrics[key]=[float(x) for x in evaluator.stats]
        (out/f"cocoeval_{key}.log").write_text(logs.getvalue(),encoding="utf-8")
        predictions[key].clear()
        del result,evaluator
        gc.collect()
        json_save("metrics_partial.json",metrics)
    summary=dict(run_id=os.environ.get("RESEARCH_RUN_ID"),study_id="STUDY_8fb3468ebb704682a2225ebed0e16206",
        branch=args.branch,ultralytics=ultralytics.__version__,package_file=ultralytics.__file__,
        protocol_sha256=hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest(),
        images=len(image_ids),start=args.start,ordinary_gt=gt_seen,selection="fixed before run; no parameter search in this program",
        decoder="upsample logits -> crop -> threshold on input grid -> scale binary -> byte (8.4.100 scale_preds)",
        metrics=metrics,delta_vs_official={key:[m-b for m,b in zip(val,metrics["official_zero"])] for key,val in metrics.items()},
        counts=counts,parity=parity,timings=timings,
        transitions={key:[dict(before=a,after=b,count=n) for (a,b),n in table.items()] for key,table in transitions.items()},
        cohort_statistics={key:{group:{**dict(sums),"mean_iou":sums["iou"]/max(sums["n"],1),
            "mean_recall":sums["recall"]/max(sums["n"],1),"mean_purity":sums["purity"]/max(sums["n"],1)}
            for group,sums in table.items()} for key,table in pixel_sums.items()},
        elapsed_seconds=time.perf_counter()-start_time,
        limitations=["development set result, not independent confirmation" if args.start==0 else "COCO val previously inspected during route development",
            "single checkpoint; same forward for all variants", "predict preprocessing, official mask functions checked on the same tensors; not full model.val parity",
            "timing is isolated decoder microbenchmark on one shape, not end-to-end deployment latency"])
    json_save("SUMMARY.json",summary)
    json_save("progress.json",dict(stage="completed",images=processed,total=len(image_ids)))
    print(json.dumps({"metrics":metrics,"parity":dict(parity),"elapsed_seconds":summary["elapsed_seconds"]}),flush=True)


if __name__=="__main__":
    main()
