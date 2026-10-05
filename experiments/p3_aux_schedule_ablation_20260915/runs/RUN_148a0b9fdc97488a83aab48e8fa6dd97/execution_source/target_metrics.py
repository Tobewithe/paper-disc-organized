"""Rich paired evaluation for the frozen raw-geometry-small failure cohort."""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import ops


COCO80 = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
THRESHOLDS = (.50, .60, .70, .75)


class EvalTrace(SegmentationPredictor):
    def postprocess(self, preds, img, orig_imgs):
        raw = preds[0][0] if isinstance(preds[0], tuple) else preds[0]
        self.dense = raw.detach().clone()
        return super().postprocess(preds, img, orig_imgs)

    def construct_result(self, pred, img, orig_img, img_path, proto):
        result = super().construct_result(pred, img, orig_img, img_path, proto)
        self.capture = {"shape": tuple(orig_img.shape[:2]), "input_shape": tuple(img.shape[2:])}
        return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def box_iou(boxes: np.ndarray, target: np.ndarray) -> np.ndarray:
    lt = np.maximum(boxes[:, :2], target[:2])
    rb = np.minimum(boxes[:, 2:], target[2:])
    inter = np.maximum(rb - lt, 0).prod(1)
    area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(1)
    target_area = np.maximum(target[2:] - target[:2], 0).prod()
    return inter / np.maximum(area + target_area - inter, 1e-12)


def box_details(box: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    iou = float(box_iou(box[None], gt)[0])
    center = (box[:2] + box[2:]) / 2
    gt_center = (gt[:2] + gt[2:]) / 2
    gt_wh = np.maximum(gt[2:] - gt[:2], 1e-6)
    wh = np.maximum(box[2:] - box[:2], 1e-6)
    lt = np.maximum(box[:2], gt[:2]); rb = np.minimum(box[2:], gt[2:])
    inter = float(np.maximum(rb - lt, 0).prod())
    return {
        "box_iou": iou,
        "box_center_error_norm": float(np.linalg.norm(center - gt_center) / np.hypot(*gt_wh)),
        "box_log_width_error": float(abs(np.log(wh[0] / gt_wh[0]))),
        "box_log_height_error": float(abs(np.log(wh[1] / gt_wh[1]))),
        "box_gt_coverage": inter / float(gt_wh.prod()),
        "box_purity": inter / float(wh.prod()),
    }


def boundary(mask: np.ndarray) -> np.ndarray:
    value = mask.astype(np.uint8)
    eroded = cv2.erode(value, np.ones((3, 3), np.uint8), iterations=1)
    return value.astype(bool) & ~eroded.astype(bool)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int) -> float:
    pb, gb = boundary(pred), boundary(gt)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tolerance+1, 2*tolerance+1))
    gd = cv2.dilate(gb.astype(np.uint8), kernel).astype(bool)
    pd = cv2.dilate(pb.astype(np.uint8), kernel).astype(bool)
    precision = float((pb & gd).sum()) / max(int(pb.sum()), 1)
    recall = float((gb & pd).sum()) / max(int(gb.sum()), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def bootstrap(values: list[float], seed: int, reps: int = 3000) -> tuple[float, float, float]:
    x = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(reps)])
    low, high = np.quantile(means, [.025, .975])
    return float(x.mean()), float(low), float(high)


def image_context(coco: COCO, ann: dict, image: np.ndarray):
    image_id, category_id = int(ann["image_id"]), int(ann["category_id"])
    own = coco.annToMask(ann).astype(bool)
    same = np.zeros_like(own); other = np.zeros_like(own)
    ici = 0.0; same_count = 0
    x, y, w, h = ann["bbox"]; own_box = np.array([x, y, x+w, y+h], float)
    for other_ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False)):
        if int(other_ann["id"]) == int(ann["id"]):
            continue
        mask = coco.annToMask(other_ann).astype(bool)
        if int(other_ann["category_id"]) == category_id:
            same |= mask; same_count += 1
            ox, oy, ow, oh = other_ann["bbox"]
            ob = np.array([ox, oy, ox+ow, oy+oh], float)
            lt = np.maximum(own_box[:2], ob[:2]); rb = np.minimum(own_box[2:], ob[2:])
            ici += float(np.maximum(rb-lt, 0).prod()) / max(w*h, 1e-12)
        else:
            other |= mask
    ratio = min(640 / own.shape[0], 640 / own.shape[1])
    radius4 = max(1, int(math.ceil(4 / ratio)))
    own_boundary = boundary(own)
    if same.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius4+1, 2*radius4+1))
        near = cv2.dilate(same.astype(np.uint8), kernel).astype(bool)
        exposure4 = float((own_boundary & near).sum()) / max(int(own_boundary.sum()), 1)
        distance = cv2.distanceTransform((~same).astype(np.uint8), cv2.DIST_L2, 3)
        nearest_same_input_px = float(distance[own].min(initial=np.inf) * ratio)
    else:
        exposure4, nearest_same_input_px = 0.0, -1.0
    all_gt = own | same | other
    ring_radius = max(1, int(math.ceil(8 / ratio)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*ring_radius+1, 2*ring_radius+1))
    ring = cv2.dilate(own.astype(np.uint8), kernel).astype(bool) & ~all_gt
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    contrast = float(np.linalg.norm(lab[own].mean(0) - lab[ring].mean(0))) if own.any() and ring.any() else float("nan")
    return own, same, other, {
        "same_class_neighbor_count": same_count,
        "same_class_box_ici": ici,
        "same_class_boundary_exposure_e4": exposure4,
        "nearest_same_mask_distance_input_px": nearest_same_input_px,
        "target_ring_lab_contrast": contrast,
    }


def mask_details(pred: np.ndarray, own: np.ndarray, same: np.ndarray, other: np.ndarray) -> dict[str, float]:
    gt_n, pred_n = int(own.sum()), int(pred.sum())
    tp = int((pred & own).sum())
    same_fp = int((pred & same & ~own).sum())
    other_fp = int((pred & other & ~own & ~same).sum())
    background_fp = int((pred & ~(own | same | other)).sum())
    union = int((pred | own).sum())
    tolerance = max(1, int(round(.0075 * np.hypot(*own.shape))))
    return {
        "gt_pixels": gt_n,
        "predicted_pixels": pred_n,
        "true_positive_pixels": tp,
        "false_negative_pixels": gt_n - tp,
        "same_neighbor_fp_pixels": same_fp,
        "other_neighbor_fp_pixels": other_fp,
        "background_fp_pixels": background_fp,
        "mask_iou": tp / max(union, 1),
        "target_coverage": tp / max(gt_n, 1),
        "prediction_purity": tp / max(pred_n, 1),
        "false_positive_over_gt": (pred_n - tp) / max(gt_n, 1),
        "predicted_area_over_gt": pred_n / max(gt_n, 1),
        "same_neighbor_leak_pred": same_fp / max(pred_n, 1),
        "same_neighbor_leak_gt": same_fp / max(gt_n, 1),
        "other_neighbor_leak_pred": other_fp / max(pred_n, 1),
        "other_neighbor_leak_gt": other_fp / max(gt_n, 1),
        "background_leak_pred": background_fp / max(pred_n, 1),
        "background_leak_gt": background_fp / max(gt_n, 1),
        "boundary_f1": boundary_f1(pred, own, tolerance),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    chosen = read_csv(args.selection)
    if args.limit:
        chosen = chosen[:args.limit]
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(args.annotation))
    rows: list[dict[str, object]] = []; start = time.time()
    for arm, path in (("baseline", args.baseline), ("method", args.method)):
        model = YOLO(str(path)); model.model.eval().requires_grad_(False); model.model.model[-1].end2end = False
        for number, item in enumerate(chosen, 1):
            ann = coco.anns[int(item["annotation_id"])]; info = coco.imgs[int(item["image_id"])]
            image_path = args.images / info["file_name"]; image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
            own, same, other, context = image_context(coco, ann, image)
            x, y, w, h = ann["bbox"]; gtbox = np.array([x, y, x+w, y+h], float)
            with torch.inference_mode():
                result = model.predict(str(image_path), predictor=EvalTrace, imgsz=640, rect=False, conf=.001, iou=.7, max_det=300, retina_masks=True, device=0, verbose=False, end2end=False)[0]
            raw = model.predictor.dense[0]
            raw_boxes = ops.xywh2xyxy(raw[:4].T.clone())
            raw_boxes = ops.scale_boxes((640, 640), raw_boxes, model.predictor.capture["shape"]).float().cpu().numpy()
            raw_scores = raw[4:84].T.float().cpu().numpy(); true_index = COCO80.index(int(ann["category_id"]))
            raw_ious = box_iou(raw_boxes, gtbox); best_geometry = int(raw_ious.argmax()); best_score = int(raw_scores[:, true_index].argmax())
            raw_detail = box_details(raw_boxes[best_geometry], gtbox)
            row: dict[str, object] = {**item, "seed": args.seed, "arm": arm, **context,
                "raw_best_box_iou": float(raw_ious[best_geometry]),
                "raw_best_source_level": int(0 if best_geometry < 6400 else 1 if best_geometry < 8000 else 2),
                "raw_true_score_at_best_geometry": float(raw_scores[best_geometry, true_index]),
                "raw_top1_correct_at_best_geometry": int(raw_scores[best_geometry].argmax() == true_index),
                "raw_max_true_score": float(raw_scores[best_score, true_index]),
                "raw_box_iou_at_max_true_score": float(raw_ious[best_score]),
                "raw_center_error_norm": raw_detail["box_center_error_norm"],
                "raw_log_width_error": raw_detail["box_log_width_error"],
                "raw_log_height_error": raw_detail["box_log_height_error"],
                "raw_p3_best_box_iou": float(raw_ious[:6400].max(initial=0)),
                "raw_p4_best_box_iou": float(raw_ious[6400:8000].max(initial=0)),
                "raw_p5_best_box_iou": float(raw_ious[8000:].max(initial=0)),
            }
            for threshold in THRESHOLDS:
                row[f"raw_box{int(threshold*100)}"] = int(raw_ious[best_geometry] >= threshold)
            pred = np.zeros_like(own); final_exists = 0; final_confidence = 0.0
            final_box = np.zeros(4, float)
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes.xyxy.float().cpu().numpy(); classes = result.boxes.cls.int().cpu().numpy()
                ids = np.flatnonzero([0 <= c < 80 and COCO80[c] == int(ann["category_id"]) for c in classes])
                if len(ids):
                    values = box_iou(boxes[ids], gtbox); best = int(ids[int(values.argmax())]); final_exists = 1
                    final_box = boxes[best]; final_confidence = float(result.boxes.conf[best].item())
                    if result.masks is not None:
                        pred = result.masks.data[best].bool().cpu().numpy()
                        if pred.shape != own.shape:
                            pred = cv2.resize(pred.astype(np.uint8), (own.shape[1], own.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            final_box_detail = box_details(final_box, gtbox) if final_exists else {k: 0.0 for k in box_details(gtbox, gtbox)}
            spatial = mask_details(pred, own, same, other)
            row.update({"final_candidate_exists": final_exists, "final_confidence": final_confidence})
            row.update({f"final_{k}": v for k, v in final_box_detail.items()})
            row.update(spatial)
            for threshold in THRESHOLDS:
                row[f"final_box{int(threshold*100)}"] = int(final_box_detail["box_iou"] >= threshold)
                row[f"final_mask{int(threshold*100)}"] = int(spatial["mask_iou"] >= threshold)
            row["final_failure_state"] = ("no_same_class_candidate" if not final_exists else "poor_box" if final_box_detail["box_iou"] < .5 else "good_box_poor_mask" if spatial["mask_iou"] < .75 else "good_box_good_mask")
            rows.append(row)
            if number % 64 == 0:
                write_csv(args.out / "per_target.csv", rows)
                print(f"[{arm} {number}/{len(chosen)}] {time.time()-start:.1f}s", flush=True)
    write_csv(args.out / "per_target.csv", rows)

    numeric_metrics = [k for k, value in rows[0].items() if isinstance(value, (int, float)) and k not in {"pair_id", "image_id", "annotation_id", "category_id", "seed"}]
    by = {(str(r["cohort"]), int(r["annotation_id"]), str(r["arm"])): r for r in rows}
    summary: list[dict[str, object]] = []
    for cohort in sorted({str(r["cohort"]) for r in rows}):
        ids = sorted({int(r["annotation_id"]) for r in rows if r["cohort"] == cohort})
        for arm in ("baseline", "method"):
            rec: dict[str, object] = {"seed": args.seed, "cohort": cohort, "arm": arm, "n": len(ids)}
            for metric in numeric_metrics:
                values = [float(by[(cohort, aid, arm)][metric]) for aid in ids]
                if arm == "method":
                    values = [v - float(by[(cohort, aid, "baseline")][metric]) for aid, v in zip(ids, values)]
                mean, low, high = bootstrap(values, args.seed + 100)
                suffix = "_delta" if arm == "method" else "_mean"
                rec[metric + suffix] = mean; rec[metric + "_ci_low"] = low; rec[metric + "_ci_high"] = high
            summary.append(rec)
    write_csv(args.out / "summary.csv", summary)

    pairs = sorted({int(r["pair_id"]) for r in rows})
    pair_map = {(int(r["pair_id"]), str(r["cohort"]), str(r["arm"])): r for r in rows}
    interactions = []
    for metric in numeric_metrics:
        values = []
        for pair in pairs:
            keys = [(pair, cohort, arm) for cohort in ("raw_geometry_small", "matched_small_control") for arm in ("baseline", "method")]
            if not all(k in pair_map for k in keys):
                continue
            failure_delta = float(pair_map[(pair, "raw_geometry_small", "method")][metric]) - float(pair_map[(pair, "raw_geometry_small", "baseline")][metric])
            control_delta = float(pair_map[(pair, "matched_small_control", "method")][metric]) - float(pair_map[(pair, "matched_small_control", "baseline")][metric])
            values.append(failure_delta - control_delta)
        mean, low, high = bootstrap(values, args.seed + 200)
        interactions.append({"seed": args.seed, "metric": metric, "pairs": len(values), "failure_minus_control_interaction": mean, "ci_low": low, "ci_high": high})
    write_csv(args.out / "failure_control_interactions.csv", interactions)
    (args.out / "COMPLETE.json").write_text(json.dumps({"status": "complete", "seed": args.seed, "targets": len(chosen), "branch": "one-to-many+NMS", "elapsed_s": time.time()-start}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
