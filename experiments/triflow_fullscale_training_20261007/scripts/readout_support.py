"""COCO/paired helpers copied unchanged from the completed ACD evaluator.
Only EVALUATOR_VERSION differs; no legacy model loader or legacy main is used.
Origin: experiments/acd_proto_tail_unfreeze_20261006/scripts/evaluate.py
Origin SHA256: 82aa72bf536d570db1c763036bbf8b74a649d7bfd472f7cf8de2cee4114f5ada
Copied functions: gather_images,coco_ap,box_iou,bootstrap_summary,paired_metrics.
Their AP/crowd/mask-area/duplicate-diagnostic definitions remain unchanged.
"""
from __future__ import annotations
import contextlib
import hashlib
import io
import json
from pathlib import Path
from frozen_io import dump_json as dump, sha256 as sha
EVALUATOR_VERSION = "triflow_eval_v1"

def gather_images(images_list: Path, coco):
    paths = []
    ids = []
    by_filename = {image["file_name"]: iid for iid, image in coco.imgs.items()}
    for line in images_list.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        path = Path(line.strip())
        if not path.is_absolute():
            path = images_list.parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        iid = by_filename.get(path.name)
        if iid is None:
            raise ValueError(f"Image is absent from annotation file: {path}")
        if iid in ids:
            raise ValueError(f"Duplicate image identity: {iid}")
        ids.append(iid)
        paths.append(path)
    if not ids:
        raise ValueError("Image list is empty")
    return paths, ids


def coco_ap(name, coco, ids, out, COCO, COCOeval):
    arm = out / name
    output = arm / "COCO_METRICS.json"
    prediction_sha = sha(arm / "predictions.json")
    if output.is_file():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if cached.get("predictions_sha256") == prediction_sha and cached.get("image_ids") == sorted(ids):
            print(f"METRICS_CACHE_REUSE {name}", flush=True)
            return cached
    result = {"image_ids": sorted(ids), "predictions_sha256": prediction_sha,
              "backend": "pycocotools.COCOeval", "units": "fraction (multiply by 100 for AP points)"}
    names = ["AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge", "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge"]
    for iou_type in ("segm", "bbox"):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            predictions = json.loads((arm / "predictions.json").read_text(encoding="utf-8"))
            if iou_type == "segm":
                # COCO.loadRes prefers bbox when both keys are present. Drop
                # bbox for segmentation evaluation so detection area is the
                # mask area, as required for standard APsmall/medium/large.
                for prediction in predictions:
                    prediction.pop("bbox", None)
            else:
                for prediction in predictions:
                    prediction.pop("segmentation", None)
            if predictions:
                detected = coco.loadRes(predictions)
            else:
                detected = COCO()
                detected.dataset = {"images": list(coco.imgs.values()), "categories": list(coco.cats.values()), "annotations": []}
                detected.createIndex()
            evaluator = COCOeval(coco, detected, iou_type)
            evaluator.params.imgIds = sorted(ids)
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        (arm / f"COCO_{iou_type}.txt").write_text(stream.getvalue(), encoding="utf-8")
        result[iou_type] = {key: float(value) if value >= 0 else None for key, value in zip(names, evaluator.stats)}
        # Save the standard accumulated tensors so AP is independently inspectable.
        import numpy as np
        np.savez_compressed(arm / f"COCO_{iou_type}_ACCUMULATED.npz",
                            precision=evaluator.eval["precision"], recall=evaluator.eval["recall"],
                            scores=evaluator.eval["scores"], iou_thresholds=evaluator.params.iouThrs,
                            recall_thresholds=evaluator.params.recThrs, category_ids=evaluator.params.catIds,
                            max_detections=evaluator.params.maxDets, area_ranges=evaluator.params.areaRng)
        print(f"COCO_AP {name} {iou_type}: {json.dumps(result[iou_type])}", flush=True)
        del evaluator, detected, predictions
    dump(output, result)
    return result


def box_iou(a, b, np):
    if not len(b):
        return np.zeros(0)
    b = np.asarray(b, dtype=np.float64)
    intersection = np.maximum(0, np.minimum(a[2:], b[:, 2:])-np.maximum(a[:2], b[:, :2])).prod(1)
    union = np.maximum(0, a[2:]-a[:2]).prod() + np.maximum(0, b[:, 2:]-b[:, :2]).prod(1) - intersection
    return intersection / np.maximum(union, 1e-12)


def bootstrap_summary(per_image, np, samples, seed):
    # Columns: matches, delta_sum, success_count, damage_count, failure_count,
    # repair_count, baseline_iou_sum, method_iou_sum, success_delta_sum.
    sums = per_image.sum(0)
    definitions = {"mean_mask_iou_delta": (1, 0), "damage_rate_of_baseline_success": (3, 2),
                   "repair_rate_of_baseline_failure": (5, 4), "baseline_mean_mask_iou": (6, 0),
                   "method_mean_mask_iou": (7, 0), "baseline_success_mean_iou_delta": (8, 2)}
    rng = np.random.default_rng(seed)
    draws = []
    # Bounded memory: full-val 5000x5000 draw arrays would consume 200 MB.
    for offset in range(0, samples, 100):
        batch = min(100, samples-offset)
        indices = rng.integers(0, len(per_image), size=(batch, len(per_image)))
        draws.append(per_image[indices].sum(1))
    draws = np.concatenate(draws, axis=0)
    result = {"matched_detection_count": int(sums[0]), "baseline_success_count": int(sums[2]),
              "damage_count": int(sums[3]), "baseline_failure_count": int(sums[4]),
              "repair_count": int(sums[5]), "images": len(per_image),
              "images_with_matched_detections": int((per_image[:, 0] > 0).sum()),
              "bootstrap_samples": samples, "bootstrap_seed": seed,
              "bootstrap_unit": "supplied images, paired across arms; ratio of clustered sums"}
    for key, (numerator, denominator) in definitions.items():
        valid = draws[:, denominator] > 0
        distribution = draws[valid, numerator] / draws[valid, denominator]
        result[key] = {"value": float(sums[numerator]/sums[denominator]) if sums[denominator] else None,
                       "ci95": np.quantile(distribution, [.025, .975]).tolist() if len(distribution) else None,
                       "valid_bootstrap_draws": int(valid.sum())}
    valid = per_image[:, 0] > 0
    result["image_macro_mask_iou_delta"] = float(np.mean(per_image[valid, 1]/per_image[valid, 0])) if valid.any() else None
    return result


def paired_metrics(baseline, names, coco, ids, out, receipts, np, mask_utils, args):
    output = {}
    baseline_hash = receipts[baseline]["frozen_state_sha256"]
    for name in names:
        if name == baseline:
            continue
        pairdir = out / f"paired_{name}_vs_{baseline}"
        pairdir.mkdir(parents=True, exist_ok=True)
        identity = {"baseline_predictions": receipts[baseline]["predictions_sha256"],
                    "method_predictions": receipts[name]["predictions_sha256"],
                    "evaluator_version": EVALUATOR_VERSION, "bootstrap": args.bootstrap,
                    "bootstrap_seed": args.bootstrap_seed}
        summary_file = pairdir / "SUMMARY.json"
        if summary_file.is_file():
            prior = json.loads(summary_file.read_text(encoding="utf-8"))
            if prior.get("input_identity") == identity:
                print(f"PAIRED_CACHE_REUSE {name}", flush=True)
                output[name] = prior
                continue
        parity = {"frozen_state_exact_equal": baseline_hash == receipts[name]["frozen_state_sha256"],
                  "image_count": len(ids), "detection_count_mismatch_images": 0,
                  "class_mismatch_detections": 0, "box_max_abs_error": 0., "confidence_max_abs_error": 0.,
                  "detection_index_equal": True, "all_images_box_class_exact_equal": True,
                  "pairing": "same native postprocess detection index; no mask-dependent filtering",
                  "readout_view": "output detections; raw candidate geometry/five-state diagnosis not computed here"}
        unique_gt = set()
        counts = np.zeros((len(ids), 9), dtype=np.float64)
        with (pairdir / "INSTANCES.jsonl").open("w", encoding="utf-8") as instances, \
             (pairdir / "IMAGES.jsonl").open("w", encoding="utf-8") as images:
            for image_pos, iid in enumerate(ids):
                base = json.loads((out / baseline / "images" / f"{iid:012d}.json").read_text(encoding="utf-8"))["detections"]
                method = json.loads((out / name / "images" / f"{iid:012d}.json").read_text(encoding="utf-8"))["detections"]
                image_equal = len(base) == len(method)
                if not image_equal:
                    parity["detection_count_mismatch_images"] += 1
                if image_equal and base:
                    base_boxes = np.asarray([d["raw_input_box_xyxy"] for d in base])
                    method_boxes = np.asarray([d["raw_input_box_xyxy"] for d in method])
                    base_conf = np.asarray([d["raw_confidence"] for d in base])
                    method_conf = np.asarray([d["raw_confidence"] for d in method])
                    mismatches = sum(a["category_id"] != b["category_id"] for a, b in zip(base, method))
                    box_error = float(np.max(np.abs(method_boxes-base_boxes)))
                    conf_error = float(np.max(np.abs(method_conf-base_conf)))
                    parity["class_mismatch_detections"] += mismatches
                    parity["box_max_abs_error"] = max(parity["box_max_abs_error"], box_error)
                    parity["confidence_max_abs_error"] = max(parity["confidence_max_abs_error"], conf_error)
                    index_equal = all(a["detection_index"] == b["detection_index"] for a, b in zip(base, method))
                    parity["detection_index_equal"] &= index_equal
                    image_equal &= mismatches == 0 and box_error == 0 and conf_error == 0 and index_equal
                parity["all_images_box_class_exact_equal"] &= image_equal
                if not image_equal:
                    images.write(json.dumps({"image_id": iid, "paired_valid": False,
                                             "reason": "box/class/confidence/detection-index parity failed"})+"\n")
                    continue
                anns_by_class = {}
                gt_rles = {}
                for ann in coco.imgToAnns.get(iid, []):
                    if not ann.get("iscrowd", 0):
                        anns_by_class.setdefault(ann["category_id"], []).append(ann)
                for a, b in zip(base, method):
                    candidates = anns_by_class.get(a["category_id"], [])
                    if not candidates:
                        continue
                    gt_boxes = [[x["bbox"][0], x["bbox"][1], x["bbox"][0]+x["bbox"][2], x["bbox"][1]+x["bbox"][3]] for x in candidates]
                    overlaps = box_iou(np.asarray(a["box_xyxy"]), gt_boxes, np)
                    gt_index = int(np.argmax(overlaps))
                    if overlaps[gt_index] < .5:
                        continue
                    gt = candidates[gt_index]
                    gid = int(gt["id"])
                    if gid not in gt_rles:
                        gt_rles[gid] = coco.annToRLE(gt)
                    def overlap(detection):
                        rle = detection["segmentation"] | {"counts": detection["segmentation"]["counts"].encode("ascii")}
                        return float(mask_utils.iou([rle], [gt_rles[gid]], [0])[0, 0])
                    baseline_iou, method_iou = overlap(a), overlap(b)
                    success = baseline_iou >= .75
                    damage = success and method_iou < .75
                    repair = not success and method_iou >= .75
                    delta = method_iou-baseline_iou
                    counts[image_pos] += [1, delta, int(success), int(damage), int(not success), int(repair),
                                          baseline_iou, method_iou, delta if success else 0]
                    unique_gt.add((iid, gid))
                    row = {"image_id": iid, "detection_index": a["detection_index"], "annotation_id": gid,
                           "category_id": a["category_id"], "baseline_score": a["raw_confidence"],
                           "baseline_box_iou": float(overlaps[gt_index]), "gt_area": float(gt["area"]),
                           "baseline_mask_iou": baseline_iou, "method_mask_iou": method_iou,
                           "delta_mask_iou": delta, "baseline_success": success, "damage": damage, "repair": repair}
                    instances.write(json.dumps(row, allow_nan=False)+"\n")
                c = counts[image_pos]
                images.write(json.dumps({"image_id": iid, "paired_valid": True,
                                         "matched_detections": int(c[0]), "mask_iou_delta_sum": float(c[1]),
                                         "success": int(c[2]), "damage": int(c[3]),
                                         "failure": int(c[4]), "repair": int(c[5])})+"\n")
        summary = {"input_identity": identity, "parity": parity,
                   "paired_readout_valid": parity["frozen_state_exact_equal"] and parity["all_images_box_class_exact_equal"],
                   "definition": "all baseline detections conf>.001, same-class highest-box-IoU noncrowd GT>=.5; "
                                 "duplicate detection-to-GT matches allowed; success/damage/repair at maskIoU=.75",
                   "unique_matched_gt_instances": len(unique_gt),
                   "statistics": bootstrap_summary(counts, np, args.bootstrap, args.bootstrap_seed)}
        if not summary["paired_readout_valid"]:
            summary["limitation"] = "Invalid frozen parity; paired rows only cover parity-passing images and cannot support the gate"
        dump(summary_file, summary)
        output[name] = summary
        print(f"PAIRED {name} valid={summary['paired_readout_valid']} "
              f"damage={summary['statistics']['damage_count']} repair={summary['statistics']['repair_count']}", flush=True)
    return output

