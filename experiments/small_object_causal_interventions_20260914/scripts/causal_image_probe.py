"""GT-assisted causal image interventions for small-object raw-geometry failures.

GT is used only to perform controlled diagnostic edits and to measure outcomes.
This is not an inference method and the outputs are not AP claims.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.utils import ops

COCO80 = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def input_box(box, shape):
    h, w = shape
    scale = min(640 / h, 640 / w)
    nw, nh = round(w * scale), round(h * scale)
    left, top = round((640 - nw) / 2 - .1), round((640 - nh) / 2 - .1)
    x, y, bw, bh = box
    return np.array([x * scale + left, y * scale + top, (x + bw) * scale + left, (y + bh) * scale + top], np.float32)


def box_iou(boxes, target):
    lt = np.maximum(boxes[:, :2], target[:2])
    rb = np.minimum(boxes[:, 2:], target[2:])
    inter = np.maximum(rb - lt, 0).prod(1)
    area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(1)
    target_area = np.maximum(target[2:] - target[:2], 0).prod()
    return inter / np.maximum(area + target_area - inter, 1e-12)


def clean_ring(target, all_objects):
    area = max(float(target.sum()), 1.0)
    radius = max(3, int(round(np.sqrt(area) * .20)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    ring = (cv2.dilate(target, kernel) > 0) & (target == 0) & (all_objects == 0)
    return ring, radius


def shift_target_contrast(image, target, all_objects, direction, magnitude=28.0):
    """Shift target luminance away from/toward a clean local ring mean."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    fg = target > 0
    ring, _ = clean_ring(target, all_objects)
    if not fg.any() or not ring.any():
        return image.copy(), 0.0, 0.0
    fg_mean = float(lab[..., 0][fg].mean())
    bg_mean = float(lab[..., 0][ring].mean())
    before = abs(fg_mean - bg_mean)
    if direction == "increase":
        sign = 1.0 if fg_mean >= bg_mean else -1.0
        if before < 2.0:
            # Pick the direction with more available range when the sign is ambiguous.
            sign = 1.0 if fg_mean < 127.5 else -1.0
        delta = sign * magnitude
    elif direction == "decrease":
        delta = np.clip(bg_mean - fg_mean, -magnitude, magnitude)
    else:
        raise ValueError(direction)
    lab[..., 0][fg] = np.clip(lab[..., 0][fg] + delta, 0, 255)
    edited = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    after_lab = cv2.cvtColor(edited, cv2.COLOR_BGR2LAB).astype(np.float32)
    after = abs(float(after_lab[..., 0][fg].mean()) - float(after_lab[..., 0][ring].mean()))
    return edited, before, after


def remove_proximal_neighbors(image, coco, ann, target, all_objects):
    """Inpaint other annotated objects sufficiently close to the target."""
    _, radius = clean_ring(target, all_objects)
    proximity_radius = max(5, radius * 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * proximity_radius + 1, 2 * proximity_radius + 1))
    influence = cv2.dilate(target, kernel) > 0
    remove = np.zeros_like(target, np.uint8)
    same_pixels = 0
    neighbor_count = 0
    for oid in coco.getAnnIds(imgIds=[ann["image_id"]], iscrowd=None):
        other = coco.anns[oid]
        if oid == ann["id"] or other.get("iscrowd", 0):
            continue
        om = coco.annToMask(other).astype(np.uint8)
        proximal = (om > 0) & influence & (target == 0)
        if proximal.any():
            neighbor_count += 1
            remove[proximal] = 255
            if other["category_id"] == ann["category_id"]:
                same_pixels += int(proximal.sum())
    if not remove.any():
        return image.copy(), 0, 0, 0
    edited = cv2.inpaint(image, remove, 5, cv2.INPAINT_TELEA)
    return edited, int((remove > 0).sum()), same_pixels, neighbor_count


def measure(model, TraceCapture, image, ann, shape):
    label = COCO80.index(ann["category_id"])
    target = input_box(ann["bbox"], shape)
    with torch.inference_mode():
        model.predict(image, predictor=TraceCapture, imgsz=640, rect=False, conf=.001, iou=.7,
                      max_det=300, retina_masks=False, device=0, verbose=False, end2end=False)
    raw = model.predictor.dense[0]
    boxes = ops.xywh2xyxy(raw[:4].T).float().cpu().numpy()
    scores = raw[4:84].T.float().cpu().numpy()
    ious = box_iou(boxes, target)
    best_any = int(np.argmax(ious))
    # Geometry conditional on meaningful true-class evidence.
    true_valid = scores[:, label] >= .001
    best_true = int(np.argmax(np.where(true_valid, ious, -1.0))) if true_valid.any() else -1
    # Local P3 response in cells whose centers lie near the target support.
    side = 80
    yy, xx = np.meshgrid(np.arange(side) + .5, np.arange(side) + .5, indexing="ij")
    points = np.c_[xx.ravel() * 8, yy.ravel() * 8]
    local = ((points[:, 0] >= target[0] - 4) & (points[:, 0] <= target[2] + 4) &
             (points[:, 1] >= target[1] - 4) & (points[:, 1] <= target[3] + 4))
    ids = np.flatnonzero(local)
    local_id = int(ids[np.argmax(scores[ids, label])])
    return {
        "best_any_box_iou": float(ious[best_any]),
        "best_any_true_score": float(scores[best_any, label]),
        "best_true_box_iou": float(ious[best_true]) if best_true >= 0 else 0.0,
        "best_true_score": float(scores[best_true, label]) if best_true >= 0 else 0.0,
        "local_true_score": float(scores[local_id, label]),
        "local_box_iou": float(ious[local_id]),
        "local_top1": int(scores[local_id].argmax() == label),
        "raw_box50": int(ious[best_any] >= .5),
    }


def bootstrap(values, seed=0, reps=3000):
    x = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(reps)])
    lo, hi = np.quantile(means, [.025, .975])
    return float(x.mean()), float(lo), float(hi)


def main():
    out = Path(__file__).resolve().parents[1]
    root = out.parents[1]
    source = root / "experiments/small_raw_geometry_origin_20260914"
    shared = root / "shared/coco_clean_20260911"
    sys.path.insert(0, str(shared))
    from structure_candidate_trace import TraceCapture

    selected = read_csv(source / "selection.csv")
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(root / "assets/datasets/coco/annotations/instances_val2017.json"))
    model = YOLO(str(root / "assets/models/coco_clean_20260911/yolo26m-seg.pt"))
    model.model.eval().requires_grad_(False)
    model.model.model[-1].end2end = False
    images = shared / "local_readout_runtime_20260912/data/images/val2017"

    rows = []
    start = time.time()
    for index, item in enumerate(selected, 1):
        ann = coco.anns[int(item["annotation_id"])]
        info = coco.imgs[int(item["image_id"])]
        image = cv2.imread(str(images / info["file_name"]))
        target = coco.annToMask(ann).astype(np.uint8)
        all_objects = np.zeros_like(target, np.uint8)
        for oid in coco.getAnnIds(imgIds=[ann["image_id"]], iscrowd=None):
            other = coco.anns[oid]
            if not other.get("iscrowd", 0):
                all_objects |= coco.annToMask(other).astype(np.uint8)

        plus, contrast_before, contrast_plus = shift_target_contrast(image, target, all_objects, "increase")
        minus, _, contrast_minus = shift_target_contrast(image, target, all_objects, "decrease")
        no_neighbor, removed, same_removed, neighbor_count = remove_proximal_neighbors(
            image, coco, ann, target, all_objects)
        plus_no_neighbor, _, _ = shift_target_contrast(no_neighbor, target, all_objects, "increase")
        variants = {
            "original": image,
            "contrast_increase": plus,
            "contrast_decrease": minus,
            "neighbor_remove": no_neighbor,
            "contrast_plus_neighbor_remove": plus_no_neighbor,
        }
        for intervention, edited in variants.items():
            result = measure(model, TraceCapture, edited, ann, (info["height"], info["width"]))
            rows.append({**item, "intervention": intervention,
                         "contrast_before": contrast_before, "contrast_after_plus": contrast_plus,
                         "contrast_after_minus": contrast_minus, "removed_neighbor_pixels": removed,
                         "removed_sameclass_pixels": same_removed, "proximal_neighbor_count": neighbor_count,
                         **result})
        if index % 32 == 0 or index == len(selected):
            write_csv(out / "per_target_intervention.csv", rows)
            (out / "progress.json").write_text(json.dumps({"done": index, "total": len(selected),
                "elapsed_s": time.time() - start}, indent=2), encoding="utf-8")
            print(f"[{index}/{len(selected)}] {time.time()-start:.1f}s", flush=True)

    by_key = {(r["annotation_id"], r["intervention"]): r for r in rows}
    metrics = ["best_any_box_iou", "best_any_true_score", "best_true_box_iou", "best_true_score",
               "local_true_score", "local_box_iou", "local_top1", "raw_box50"]
    summary = []
    interventions = [k for k in variants if k != "original"]
    for cohort in sorted({r["cohort"] for r in rows}):
        ids = sorted({r["annotation_id"] for r in rows if r["cohort"] == cohort})
        for intervention in interventions:
            record = {"cohort": cohort, "intervention": intervention, "n": len(ids)}
            for metric in metrics:
                delta = [float(by_key[(aid, intervention)][metric]) - float(by_key[(aid, "original")][metric]) for aid in ids]
                mean, lo, hi = bootstrap(delta)
                record[metric + "_delta"] = mean
                record[metric + "_ci_low"] = lo
                record[metric + "_ci_high"] = hi
            if cohort == "raw_geometry_small":
                record["box50_recovered"] = sum(int(by_key[(aid, "original")]["raw_box50"]) == 0 and int(by_key[(aid, intervention)]["raw_box50"]) == 1 for aid in ids)
                record["box50_lost"] = sum(int(by_key[(aid, "original")]["raw_box50"]) == 1 and int(by_key[(aid, intervention)]["raw_box50"]) == 0 for aid in ids)
            summary.append(record)
    write_csv(out / "summary.csv", summary)
    (out / "COMPLETE.json").write_text(json.dumps({"status": "complete", "targets": len(selected),
        "inferences": len(rows), "ultralytics": "8.4.100", "end2end": False,
        "gt_role": "controlled image intervention and outcome measurement only",
        "elapsed_s": time.time() - start}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
