"""Test whether raw-P3 recovery is specific to the spatially exposed neighbour.

This is a GT-assisted diagnostic, not an inference method and not an AP claim.
The neighbour is selected from input geometry before any model outcome is seen.
Target pixels are required to remain bit-identical in every intervention.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
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
P3_COUNT = 80 * 80
ARMS = ("original", "neighbor_flat", "background_flat", "distant_instance_flat",
        "neighbor_blur", "background_blur")
METRICS = ("raw_p3_best_box_iou", "raw_best_box_iou", "raw_p3_center_error_norm",
           "raw_true_score_at_p3_best", "raw_box50", "raw_p3_box50")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def input_box(box: list[float], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    scale = min(640 / h, 640 / w)
    nw, nh = round(w * scale), round(h * scale)
    left, top = round((640 - nw) / 2 - .1), round((640 - nh) / 2 - .1)
    x, y, bw, bh = box
    return np.asarray([x * scale + left, y * scale + top,
                       (x + bw) * scale + left, (y + bh) * scale + top], np.float32)


def box_iou(boxes: np.ndarray, target: np.ndarray) -> np.ndarray:
    lt = np.maximum(boxes[:, :2], target[:2])
    rb = np.minimum(boxes[:, 2:], target[2:])
    inter = np.maximum(rb - lt, 0).prod(1)
    area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(1)
    target_area = np.maximum(target[2:] - target[:2], 0).prod()
    return inter / np.maximum(area + target_area - inter, 1e-12)


def center_error(box: np.ndarray, target: np.ndarray) -> float:
    center = (box[:2] + box[2:]) / 2
    target_center = (target[:2] + target[2:]) / 2
    target_wh = np.maximum(target[2:] - target[:2], 1e-6)
    return float(np.linalg.norm(center - target_center) / np.hypot(*target_wh))


def clean_ring(target: np.ndarray, all_objects: np.ndarray) -> tuple[np.ndarray, int]:
    radius = max(3, int(round(math.sqrt(max(float(target.sum()), 1.0)) * .20)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    ring = (cv2.dilate(target.astype(np.uint8), kernel) > 0) & (target == 0) & (all_objects == 0)
    return ring, radius


def choose_exposed_neighbor(coco: COCO, ann: dict, target: np.ndarray,
                            influence: np.ndarray) -> tuple[dict | None, np.ndarray]:
    candidates: list[tuple[int, int, dict, np.ndarray]] = []
    for other in coco.loadAnns(coco.getAnnIds(imgIds=[ann["image_id"]], iscrowd=False)):
        if int(other["id"]) == int(ann["id"]):
            continue
        other_mask = coco.annToMask(other).astype(bool)
        exposed = other_mask & influence & ~target.astype(bool)
        pixels = int(exposed.sum())
        if pixels:
            # Selection is fully determined by input geometry; model output is absent.
            candidates.append((pixels, -int(other["id"]), other, exposed))
    if not candidates:
        return None, np.zeros_like(target, bool)
    _, _, selected, exposed = max(candidates, key=lambda item: (item[0], item[1]))
    return selected, exposed


def translated_background_mask(source: np.ndarray, background: np.ndarray,
                               seed: int, tries: int = 2500) -> np.ndarray | None:
    """Translate the exact intervention shape onto pure background."""
    ys, xs = np.nonzero(source)
    if not len(xs):
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    rel_y, rel_x = ys - y0, xs - x0
    patch_h, patch_w = y1 - y0 + 1, x1 - x0 + 1
    height, width = source.shape
    if patch_h > height or patch_w > width:
        return None
    rng = np.random.default_rng(seed)
    order_y = rng.integers(0, height - patch_h + 1, size=tries)
    order_x = rng.integers(0, width - patch_w + 1, size=tries)
    for top, left in zip(order_y, order_x):
        yy, xx = rel_y + top, rel_x + left
        if background[yy, xx].all():
            out = np.zeros_like(source, bool)
            out[yy, xx] = True
            return out
    return None


def compact_subset(mask: np.ndarray, count: int) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < count or count <= 0:
        return None
    cy, cx = ys.mean(), xs.mean()
    ids = np.argsort((ys - cy) ** 2 + (xs - cx) ** 2, kind="stable")[:count]
    out = np.zeros_like(mask, bool)
    out[ys[ids], xs[ids]] = True
    return out


def choose_distant_instance(coco: COCO, ann: dict, target: np.ndarray,
                            influence: np.ndarray, count: int) -> tuple[dict | None, np.ndarray | None]:
    target_y, target_x = np.nonzero(target)
    tc = np.asarray([target_x.mean(), target_y.mean()])
    candidates: list[tuple[float, int, dict, np.ndarray]] = []
    for other in coco.loadAnns(coco.getAnnIds(imgIds=[ann["image_id"]], iscrowd=False)):
        if int(other["id"]) == int(ann["id"]):
            continue
        mask = coco.annToMask(other).astype(bool) & ~target.astype(bool)
        if int(mask.sum()) < count or (mask & influence).any():
            continue
        ys, xs = np.nonzero(mask)
        distance = float(np.linalg.norm(np.asarray([xs.mean(), ys.mean()]) - tc))
        candidates.append((distance, -int(other["id"]), other, mask))
    if not candidates:
        return None, None
    _, _, selected, mask = max(candidates, key=lambda item: (item[0], item[1]))
    return selected, compact_subset(mask, count)


def fill_color(image: np.ndarray, target: np.ndarray, all_objects: np.ndarray) -> np.ndarray:
    ring, _ = clean_ring(target, all_objects)
    if ring.any():
        return np.median(image[ring], axis=0).astype(np.uint8)
    background = all_objects == 0
    if background.any():
        return np.median(image[background], axis=0).astype(np.uint8)
    return np.median(image.reshape(-1, 3), axis=0).astype(np.uint8)


def apply_flat(image: np.ndarray, mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    edited = image.copy()
    edited[mask] = color
    return edited


def apply_blur(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    edited = image.copy()
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=5.0, sigmaY=5.0)
    edited[mask] = blurred[mask]
    return edited


def edit_energy(original: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(np.abs(original[mask].astype(np.float32) - edited[mask].astype(np.float32)).mean())


def measure(model: YOLO, trace_capture, image: np.ndarray, ann: dict,
            shape: tuple[int, int]) -> dict[str, float | int]:
    label = COCO80.index(int(ann["category_id"]))
    target = input_box(ann["bbox"], shape)
    with torch.inference_mode():
        model.predict(image, predictor=trace_capture, imgsz=640, rect=False, conf=.001, iou=.7,
                      max_det=300, retina_masks=False, device=0, verbose=False, end2end=False)
    raw = model.predictor.dense[0]
    boxes = ops.xywh2xyxy(raw[:4].T).float().cpu().numpy()
    scores = raw[4:84].T.float().cpu().numpy()
    ious = box_iou(boxes, target)
    best = int(np.argmax(ious))
    p3_best = int(np.argmax(ious[:P3_COUNT]))
    return {
        "raw_p3_best_box_iou": float(ious[p3_best]),
        "raw_best_box_iou": float(ious[best]),
        "raw_p3_center_error_norm": center_error(boxes[p3_best], target),
        "raw_true_score_at_p3_best": float(scores[p3_best, label]),
        "raw_box50": int(ious[best] >= .5),
        "raw_p3_box50": int(ious[p3_best] >= .5),
        "raw_best_source_level": int(0 if best < 6400 else 1 if best < 8000 else 2),
    }


def bootstrap(values: list[float], seed: int, reps: int = 5000) -> tuple[float, float, float]:
    x = np.asarray(values, float)
    if not len(x):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(x, len(x), replace=True).mean() for _ in range(reps)])
    low, high = np.quantile(means, [.025, .975])
    return float(x.mean()), float(low), float(high)


def analyse(rows: list[dict[str, object]], out: Path) -> dict[str, object]:
    index = {(str(row["annotation_id"]), str(row["arm"])): row for row in rows}
    summary: list[dict[str, object]] = []
    contrasts: list[dict[str, object]] = []
    for cohort in sorted({str(row["cohort"]) for row in rows}):
        cohort_rows = [row for row in rows if row["cohort"] == cohort and row["arm"] == "original"]
        for arm in ARMS[1:]:
            ids = [str(row["annotation_id"]) for row in cohort_rows
                   if (str(row["annotation_id"]), arm) in index]
            record: dict[str, object] = {"cohort": cohort, "arm": arm, "n": len(ids)}
            for metric in METRICS:
                delta = [float(index[(aid, arm)][metric]) - float(index[(aid, "original")][metric]) for aid in ids]
                mean, low, high = bootstrap(delta, 1700 + len(summary))
                record.update({metric + "_delta": mean, metric + "_ci_low": low,
                               metric + "_ci_high": high})
            summary.append(record)

        for treatment, control in (("neighbor_flat", "background_flat"),
                                   ("neighbor_blur", "background_blur"),
                                   ("neighbor_flat", "distant_instance_flat")):
            ids = [str(row["annotation_id"]) for row in cohort_rows
                   if (str(row["annotation_id"]), treatment) in index
                   and (str(row["annotation_id"]), control) in index]
            for metric in METRICS:
                values = [float(index[(aid, treatment)][metric]) - float(index[(aid, control)][metric])
                          for aid in ids]
                mean, low, high = bootstrap(values, 2900 + len(contrasts))
                contrasts.append({"cohort": cohort, "treatment": treatment, "control": control,
                                  "metric": metric, "n": len(ids), "paired_net": mean,
                                  "ci_low": low, "ci_high": high})

    write_csv(out / "arm_deltas.csv", summary)
    write_csv(out / "paired_contrasts.csv", contrasts)

    failures = [row for row in rows if row["cohort"] == "raw_geometry_small" and row["arm"] == "original"]
    specific: list[dict[str, object]] = []
    for row in failures:
        aid = str(row["annotation_id"])
        required = [(aid, arm) in index for arm in ("neighbor_flat", "background_flat")]
        if not all(required):
            continue
        original = float(index[(aid, "original")]["raw_p3_best_box_iou"])
        neighbor = float(index[(aid, "neighbor_flat")]["raw_p3_best_box_iou"])
        background = float(index[(aid, "background_flat")]["raw_p3_best_box_iou"])
        distant = (float(index[(aid, "distant_instance_flat")]["raw_p3_best_box_iou"])
                   if (aid, "distant_instance_flat") in index else float("nan"))
        is_specific = (original < .5 and neighbor >= .5 and neighbor - original >= .1
                       and background < .5 and neighbor - background >= .05)
        specific.append({"annotation_id": aid, "pair_id": row["pair_id"],
                         "original_p3_iou": original, "neighbor_p3_iou": neighbor,
                         "background_p3_iou": background, "distant_p3_iou": distant,
                         "neighbor_specific_recovery": int(is_specific),
                         "selected_neighbor_id": index[(aid, "neighbor_flat")]["selected_neighbor_id"],
                         "selected_neighbor_same_class": index[(aid, "neighbor_flat")]["selected_neighbor_same_class"],
                         "edited_pixels": index[(aid, "neighbor_flat")]["edited_pixels"]})
    write_csv(out / "neighbor_specific_instances.csv", specific)

    primary = next((row for row in contrasts
                    if row["cohort"] == "raw_geometry_small"
                    and row["treatment"] == "neighbor_flat"
                    and row["control"] == "background_flat"
                    and row["metric"] == "raw_p3_best_box_iou"), None)
    return {
        "targets": len({str(row["annotation_id"]) for row in rows}),
        "inferences": len(rows),
        "failure_with_selected_neighbor": sum(1 for row in failures
                                               if (str(row["annotation_id"]), "neighbor_flat") in index),
        "neighbor_specific_recoveries": sum(int(row["neighbor_specific_recovery"]) for row in specific),
        "neighbor_specific_evaluable": len(specific),
        "primary_contrast": primary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = out.parents[1]
    shared = root / "shared/coco_clean_20260911"
    sys.path.insert(0, str(shared))
    from structure_candidate_trace import TraceCapture

    selection_path = root / "experiments/small_raw_geometry_origin_20260914/selection.csv"
    annotation_path = root / "assets/datasets/coco/annotations/instances_val2017.json"
    image_root = shared / "local_readout_runtime_20260912/data/images/val2017"
    weight_path = root / "assets/models/coco_clean_20260911/yolo26m-seg.pt"
    selected = read_csv(selection_path)
    if args.limit:
        selected = selected[:args.limit]
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(annotation_path))
    model = YOLO(str(weight_path))
    model.model.eval().requires_grad_(False)
    model.model.model[-1].end2end = False
    torch.set_num_threads(4)

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    started = time.time()
    for number, item in enumerate(selected, 1):
        ann = coco.anns[int(item["annotation_id"])]
        info = coco.imgs[int(item["image_id"])]
        image_path = image_root / info["file_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        target = coco.annToMask(ann).astype(bool)
        all_objects = np.zeros_like(target, bool)
        for other in coco.loadAnns(coco.getAnnIds(imgIds=[ann["image_id"]], iscrowd=False)):
            all_objects |= coco.annToMask(other).astype(bool)
        _, ring_radius = clean_ring(target, all_objects)
        proximity_radius = max(5, 2 * ring_radius)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (2 * proximity_radius + 1, 2 * proximity_radius + 1))
        influence = cv2.dilate(target.astype(np.uint8), kernel).astype(bool)
        neighbor, neighbor_mask = choose_exposed_neighbor(coco, ann, target, influence)

        variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "original": (image, np.zeros_like(target, bool))
        }
        metadata: dict[str, object] = {
            "selected_neighbor_id": "", "selected_neighbor_category_id": "",
            "selected_neighbor_same_class": "", "eligible_background_control": 0,
            "eligible_distant_control": 0,
        }
        if neighbor is not None and int(neighbor_mask.sum()) >= 8:
            count = int(neighbor_mask.sum())
            background = ~all_objects
            background_mask = translated_background_mask(
                neighbor_mask, background, 20260915 + int(ann["id"]))
            distant, distant_mask = choose_distant_instance(coco, ann, target, influence, count)
            color = fill_color(image, target, all_objects)
            variants["neighbor_flat"] = (apply_flat(image, neighbor_mask, color), neighbor_mask)
            variants["neighbor_blur"] = (apply_blur(image, neighbor_mask), neighbor_mask)
            metadata.update({
                "selected_neighbor_id": int(neighbor["id"]),
                "selected_neighbor_category_id": int(neighbor["category_id"]),
                "selected_neighbor_same_class": int(neighbor["category_id"] == ann["category_id"]),
            })
            if background_mask is not None:
                variants["background_flat"] = (apply_flat(image, background_mask, color), background_mask)
                variants["background_blur"] = (apply_blur(image, background_mask), background_mask)
                metadata["eligible_background_control"] = 1
            if distant is not None and distant_mask is not None:
                variants["distant_instance_flat"] = (apply_flat(image, distant_mask, color), distant_mask)
                metadata["eligible_distant_control"] = 1
                metadata["distant_instance_id"] = int(distant["id"])
                metadata["distant_instance_category_id"] = int(distant["category_id"])
        else:
            skipped.append({**item, "reason": "no_exposed_neighbor_or_fewer_than_8_pixels",
                            "exposed_pixels": int(neighbor_mask.sum())})

        for arm in ARMS:
            if arm not in variants:
                continue
            edited, edit_mask = variants[arm]
            if arm != "original" and not np.array_equal(edited[target], image[target]):
                raise RuntimeError(f"Target pixels changed for annotation {ann['id']} arm {arm}")
            result = measure(model, TraceCapture, edited, ann, (info["height"], info["width"]))
            rows.append({**item, "arm": arm, **metadata,
                         "edited_pixels": int(edit_mask.sum()),
                         "target_pixels_changed": int(np.any(edited[target] != image[target])),
                         "edit_mean_absolute_channel_delta": edit_energy(image, edited, edit_mask),
                         "proximity_radius_original_px": proximity_radius,
                         **result})

        if number % 24 == 0 or number == len(selected):
            write_csv(out / "per_target.csv", rows)
            write_csv(out / "skipped.csv", skipped)
            progress = {"done": number, "total": len(selected), "inferences": len(rows),
                        "elapsed_s": time.time() - started}
            (out / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print(json.dumps(progress), flush=True)

    result = analyse(rows, out)
    result.update({"status": "complete", "elapsed_s": time.time() - started,
                   "ultralytics": "8.4.100", "checkpoint": str(weight_path),
                   "runtime_branch": "one-to-many plus NMS (end2end=False)",
                   "gt_role": "diagnostic intervention and outcome measurement only"})
    (out / "COMPLETE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
