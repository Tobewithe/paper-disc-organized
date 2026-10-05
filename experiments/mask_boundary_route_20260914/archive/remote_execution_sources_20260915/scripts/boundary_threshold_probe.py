"""Frozen diagnostic for prediction-visible, instance-adaptive mask thresholds.

The model is unchanged. COCO annotations are used only after inference to
define cohorts and score an inference policy. The policy feature is computed
from raw mask logits and predicted boxes, so it is available without GT.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "D:/coco_wire/py")

from ultralytics import YOLO
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import ops

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_utils
    from pycocotools.cocoeval import COCOeval
except Exception as exc:  # pragma: no cover - remote environment gate
    raise RuntimeError("pycocotools is required for this diagnostic") from exc


ROOT = Path("D:/coco_wire")
IMAGE_DIR = ROOT / "data/images/val2017"
ANN_FILE = ROOT / "data/annotations/instances_val2017.json"
WEIGHTS = Path("D:/mdoeldata/pigcv-task05/models/yolo26m-seg.pt")
OUT_DIR = ROOT / "results/mask_boundary_route_20260914"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = (-0.50, 0.0, 0.25, 0.50, 0.75, 1.00)
MATCH_BOX_IOU = 0.50
TARGET_BOX_IOU = 0.75
TARGET_SUPPORT = 0.95
TARGET_MASK_IOU = 0.75


def box_iou_one(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    den = aa + bb - inter
    return inter / den if den > 0 else 0.0


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union) if union else 0.0


def decode_logits(proto: torch.Tensor, pred: torch.Tensor, img_shape: tuple[int, int], orig_shape: tuple[int, int]):
    """Return cropped logits at letterbox and original-image coordinates."""
    if pred.shape[0] == 0:
        return pred.new_zeros((0, *orig_shape)), pred.new_zeros((0, 4))
    if proto.ndim == 4:
        proto = proto[0]
    c, mh, mw = proto.shape
    logits = (pred[:, 6:] @ proto.float().reshape(c, -1)).reshape(-1, mh, mw)
    logits = F.interpolate(logits[None], img_shape, mode="bilinear")[0]
    logits = ops.crop_mask(logits, pred[:, :4])
    logits_orig = ops.scale_masks(
        logits[:, None], orig_shape, padding=True, mode="bilinear"
    )[:, 0]
    boxes_orig = ops.scale_boxes(img_shape, pred[:, :4].clone(), orig_shape).cpu().numpy()
    return logits_orig, boxes_orig


def mask_at_threshold(logits: torch.Tensor, threshold: float) -> np.ndarray:
    return logits.gt(float(threshold)).cpu().numpy()


def collect_prediction_captures(model: YOLO):
    captures = {}
    original = SegmentationPredictor.construct_result

    def wrapped(self, pred, img, orig_img, img_path, proto):
        key = Path(str(img_path)).name
        captures[key] = {
            "pred": pred.detach().float().cpu().clone(),
            "proto": proto.detach().float().cpu().clone(),
            "img_shape": tuple(int(x) for x in img.shape[2:]),
            "orig_shape": tuple(int(x) for x in orig_img.shape[:2]),
        }
        return original(self, pred, img, orig_img, img_path, proto)

    SegmentationPredictor.construct_result = wrapped
    try:
        stream = model.predict(
            source=str(IMAGE_DIR),
            imgsz=640,
            device=0,
            conf=0.001,
            max_det=300,
            verbose=False,
            save=False,
            stream=True,
        )
        # Consume one result at a time; retaining the full Results list keeps
        # every decoded mask on the GPU and is unnecessary for this probe.
        for _ in stream:
            pass
    finally:
        SegmentationPredictor.construct_result = original
    return captures


def load_gt(coco: COCO, image_name: str, name_to_cls: dict[str, int]):
    image_id = int(Path(image_name).stem)
    image_info = coco.loadImgs([image_id])[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False))
    out = []
    for ann in anns:
        cat = coco.loadCats([ann["category_id"]])[0]
        cls = name_to_cls.get(cat["name"])
        if cls is None:
            continue
        mask = coco.annToMask(ann).astype(bool)
        out.append(
            {
                "ann_id": int(ann["id"]),
                "cls": int(cls),
                "bbox": np.asarray(ann["bbox"], dtype=float),
                "xyxy": np.asarray(
                    [ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]],
                    dtype=float,
                ),
                "mask": mask,
                "image_id": image_id,
                "width": int(image_info["width"]),
                "height": int(image_info["height"]),
            }
        )
    return out


def records_for_image(capture, gts, image_name):
    pred = capture["pred"].numpy()
    logits, boxes = decode_logits(
        capture["proto"],
        capture["pred"],
        capture["img_shape"],
        capture["orig_shape"],
    )
    logits = logits.numpy()
    if pred.shape[0] == 0 or not gts:
        return []

    # A deterministic confidence-first one-to-one match keeps the assignment
    # fixed while threshold policies are compared.
    order = np.argsort(-pred[:, 4])
    used = set()
    matches = []
    for pi in order.tolist():
        cls = int(pred[pi, 5])
        candidates = [
            (box_iou_one(boxes[pi], gt["xyxy"]), gi)
            for gi, gt in enumerate(gts)
            if gi not in used and int(gt["cls"]) == cls
        ]
        if not candidates:
            continue
        biou, gi = max(candidates)
        if biou < MATCH_BOX_IOU:
            continue
        used.add(gi)
        gt = gts[gi]
        inside = np.ones_like(logits[pi], dtype=np.float32)
        # Use the same crop primitive as the official decoder for a
        # prediction-visible ambiguity estimate.
        inside_t = ops.crop_mask(
            torch.ones((1, *capture["img_shape"]), dtype=torch.float32),
            capture["pred"][pi : pi + 1, :4],
        )[0]
        inside = ops.scale_masks(
            inside_t[None, None], capture["orig_shape"], padding=True, mode="nearest"
        )[0, 0].numpy() > 0
        abs_logit = np.abs(logits[pi])
        denom = max(1, int(inside.sum()))
        features = {
            "ambiguity": float((abs_logit[inside] < 0.50).sum() / denom),
            "positive_fraction": float((logits[pi][inside] > 0).sum() / denom),
            "margin_q25": float(np.quantile(abs_logit[inside], 0.25)) if inside.any() else 0.0,
            "margin_q50": float(np.quantile(abs_logit[inside], 0.50)) if inside.any() else 0.0,
        }
        threshold_masks = {}
        ious = {}
        support = {}
        for t in THRESHOLDS:
            pmask = mask_at_threshold(torch.from_numpy(logits[pi : pi + 1]), t)[0]
            threshold_masks[t] = pmask
            ious[t] = mask_iou(pmask, gt["mask"])
            support[t] = float(np.logical_and(pmask, gt["mask"]).sum() / max(1, gt["mask"].sum()))
        base_iou = ious[0.0]
        row = {
            "image_name": image_name,
            "image_id": int(gt["image_id"]),
            "ann_id": int(gt["ann_id"]),
            "pred_index": int(pi),
            "cls": int(cls),
            "confidence": float(pred[pi, 4]),
            "box_iou": float(biou),
            "base_iou": float(base_iou),
            "base_support": float(support[0.0]),
            **features,
        }
        for t in THRESHOLDS:
            key = str(t).replace("-", "m").replace(".", "p")
            row[f"iou_{key}"] = float(ious[t])
            row[f"support_{key}"] = float(support[t])
        row["target"] = int(biou >= TARGET_BOX_IOU and support[0.0] >= TARGET_SUPPORT and base_iou < TARGET_MASK_IOU)
        row["mask_good"] = int(biou >= TARGET_BOX_IOU and base_iou >= TARGET_MASK_IOU)
        row["box_good"] = int(biou >= TARGET_BOX_IOU)
        matches.append(row)
    return matches


def threshold_key(t: float) -> str:
    return str(t).replace("-", "m").replace(".", "p")


def prediction_feature(logits: torch.Tensor, capture, index: int) -> dict[str, float]:
    """Compute inference-only boundary features for one decoded prediction."""
    inside_t = ops.crop_mask(
        torch.ones((1, *capture["img_shape"]), dtype=torch.float32),
        capture["pred"][index : index + 1, :4],
    )[0]
    inside = ops.scale_masks(
        inside_t[None, None], capture["orig_shape"], padding=True, mode="nearest"
    )[0, 0].numpy() > 0
    values = logits[index].numpy()
    abs_logit = np.abs(values)
    denom = max(1, int(inside.sum()))
    return {
        "ambiguity": float((abs_logit[inside] < 0.50).sum() / denom),
        "positive_fraction": float((values[inside] > 0).sum() / denom),
        "margin_q25": float(np.quantile(abs_logit[inside], 0.25)) if inside.any() else 0.0,
        "margin_q50": float(np.quantile(abs_logit[inside], 0.50)) if inside.any() else 0.0,
    }


def build_coco_predictions(coco, captures, names, policy):
    cat_by_name = {c["name"]: int(c["id"]) for c in coco.loadCats(coco.getCatIds())}
    selected = policy["selected"] if policy and "selected" in policy else None
    outputs = {"baseline": [], "adaptive": []}
    image_ids = []
    for image_name, capture in sorted(captures.items()):
        image_id = int(Path(image_name).stem)
        image_ids.append(image_id)
        logits, boxes = decode_logits(
            capture["proto"], capture["pred"], capture["img_shape"], capture["orig_shape"]
        )
        pred = capture["pred"].numpy()
        for i in range(pred.shape[0]):
            cls = int(pred[i, 5])
            cat_id = cat_by_name.get(str(names[cls]))
            if cat_id is None:
                continue
            features = prediction_feature(logits, capture, i)
            adaptive_t = 0.0
            if selected is not None:
                adaptive_t = selected["high"] if features[selected["feature"]] >= selected["cut"] else selected["low"]
            for label, threshold in (("baseline", 0.0), ("adaptive", adaptive_t)):
                binary = logits[i].gt(float(threshold)).numpy().astype(np.uint8)
                if not binary.any():
                    continue
                encoded = mask_utils.encode(np.asfortranarray(binary))
                encoded["counts"] = encoded["counts"].decode("ascii")
                encoded["size"] = [int(x) for x in encoded["size"]]
                outputs[label].append(
                    {
                        "image_id": image_id,
                        "category_id": cat_id,
                        "segmentation": encoded,
                        "score": float(pred[i, 4]),
                    }
                )
    return outputs, image_ids


def evaluate_coco_seg(coco, predictions, image_ids):
    if not predictions:
        return None
    detections = coco.loadRes(predictions)
    evaluator = COCOeval(coco, detections, "segm")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return [float(x) for x in evaluator.stats]


def mean_for(rows, t, field="base_iou"):
    if not rows:
        return float("nan")
    if field == "base_iou":
        vals = [r[field] for r in rows]
    else:
        vals = [r[f"iou_{threshold_key(t)}"] for r in rows]
    return float(np.mean(vals))


def choose_stump(train_rows):
    features = ["ambiguity", "positive_fraction", "margin_q25", "margin_q50"]
    candidates = []
    for feature in features:
        values = np.asarray([r[feature] for r in train_rows], dtype=float)
        if len(values) < 8 or not np.isfinite(values).all():
            continue
        qs = np.quantile(values, np.linspace(0.2, 0.8, 7))
        for q in np.unique(qs):
            for low in THRESHOLDS:
                for high in THRESHOLDS:
                    pred_rows = []
                    for r in train_rows:
                        t = high if r[feature] >= q else low
                        pred_rows.append(r[f"iou_{threshold_key(t)}"])
                    # Primary objective is target-cohort IoU; keep a small
                    # penalty for harming already-good masks.
                    target = np.asarray([r["target"] for r in train_rows], dtype=bool)
                    good = np.asarray([r["mask_good"] for r in train_rows], dtype=bool)
                    vals = np.asarray(pred_rows)
                    base = np.asarray([r["base_iou"] for r in train_rows])
                    score = 0.0
                    target_gain = 0.0
                    good_drop = 0.0
                    if target.any():
                        target_gain = float((vals[target] - base[target]).mean())
                        score += target_gain
                    if good.any():
                        good_drop = float((base[good] - vals[good]).mean())
                        score -= 0.25 * max(0.0, good_drop)
                    candidates.append(
                        {
                            "feature": feature,
                            "cut": float(q),
                            "low": float(low),
                            "high": float(high),
                            "score": float(score),
                            "target_gain": float(target_gain),
                            "good_drop": float(good_drop),
                        }
                    )
    if not candidates:
        return None
    unconstrained = max(candidates, key=lambda x: x["score"])
    constrained_pool = [x for x in candidates if x["good_drop"] <= 0.001]
    constrained = max(constrained_pool, key=lambda x: x["target_gain"]) if constrained_pool else None
    selected = constrained or unconstrained
    return {
        "selected": selected,
        "unconstrained": unconstrained,
        "constrained": constrained,
        "constraint": "mean mask-good IoU drop <= 0.001",
    }


def summarize(rows, label, policy=None):
    if policy is None:
        out = {"label": label, "n": len(rows)}
        for t in THRESHOLDS:
            vals = [r[f"iou_{threshold_key(t)}"] for r in rows]
            out[f"mean_iou_t{threshold_key(t)}"] = float(np.mean(vals)) if vals else None
            out[f"success_t{threshold_key(t)}"] = float(np.mean(np.asarray(vals) >= TARGET_MASK_IOU)) if vals else None
        return out
    if "selected" in policy:
        policy = policy["selected"]
    vals = []
    for r in rows:
        t = policy["high"] if r[policy["feature"]] >= policy["cut"] else policy["low"]
        vals.append(r[f"iou_{threshold_key(t)}"])
    return {
        "label": label,
        "n": len(rows),
        "mean_iou_adaptive": float(np.mean(vals)) if vals else None,
        "success_adaptive": float(np.mean(np.asarray(vals) >= TARGET_MASK_IOU)) if vals else None,
        "mean_iou_base": float(np.mean([r["base_iou"] for r in rows])) if rows else None,
    }


def main():
    coco = COCO(str(ANN_FILE))
    model = YOLO(str(WEIGHTS))
    names = model.names
    name_to_cls = {str(name): int(cls) for cls, name in names.items()}
    captures = collect_prediction_captures(model)
    rows = []
    for image_name, capture in sorted(captures.items()):
        gts = load_gt(coco, image_name, name_to_cls)
        rows.extend(records_for_image(capture, gts, image_name))

    rows.sort(key=lambda r: (r["image_id"], r["ann_id"]))
    csv_path = OUT_DIR / "matched_records.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    image_ids = sorted({r["image_id"] for r in rows})
    split = max(1, len(image_ids) // 2)
    train_ids = set(image_ids[:split])
    test_ids = set(image_ids[split:])
    train_rows = [r for r in rows if r["image_id"] in train_ids]
    test_rows = [r for r in rows if r["image_id"] in test_ids]
    target = [r for r in rows if r["target"]]
    target_train = [r for r in train_rows if r["target"]]
    target_test = [r for r in test_rows if r["target"]]
    good_test = [r for r in test_rows if r["mask_good"]]
    policy = choose_stump(train_rows)

    coco_predictions, eval_image_ids = build_coco_predictions(coco, captures, names, policy)
    coco_eval = {
        label: evaluate_coco_seg(coco, preds, eval_image_ids)
        for label, preds in coco_predictions.items()
    }
    for label, preds in coco_predictions.items():
        with (OUT_DIR / f"predictions_{label}.json").open("w", encoding="utf-8") as f:
            json.dump(
                preds,
                f,
                ensure_ascii=False,
                default=lambda value: value.tolist() if hasattr(value, "tolist") else int(value),
            )

    summary = {
        "protocol": {
            "images": len(captures),
            "matched_records": len(rows),
            "train_images": len(train_ids),
            "test_images": len(test_ids),
            "official_version": __import__("ultralytics").__version__,
            "weight": str(WEIGHTS),
            "thresholds": list(THRESHOLDS),
            "target": "box_iou>=0.75, baseline support>=0.95, baseline mask_iou<0.75",
        },
        "cohort_counts": {
            "all_matched": len(rows),
            "box_good_mask_bad_support_sufficient": len(target),
            "mask_good_box_good": sum(r["mask_good"] for r in rows),
            "test_target": len(target_test),
        },
        "all_fixed_thresholds": summarize(rows, "all_matched"),
        "target_fixed_thresholds": summarize(target, "target"),
        "test_target_fixed_thresholds": summarize(target_test, "test_target"),
        "test_mask_good_fixed_thresholds": summarize(good_test, "test_mask_good"),
        "oracle_target_test": {
            "n": len(target_test),
            "mean_iou": float(np.mean([max(r[f"iou_{threshold_key(t)}"] for t in THRESHOLDS) for r in target_test])) if target_test else None,
        },
        "policy": policy,
        "policy_test_target": summarize(target_test, "test_target", policy) if policy else None,
        "policy_test_mask_good": summarize(good_test, "test_mask_good", policy) if policy else None,
        "coco_eval_segmentation": coco_eval,
        "prediction_counts": {label: len(preds) for label, preds in coco_predictions.items()},
    }
    with (OUT_DIR / "SUMMARY.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
