"""Outcome-first diagnostic: can local high-resolution re-inference recover distinct COCO failures?

This is an oracle diagnostic, not a deployable method. Ground-truth boxes define ROI crops and
ground-truth masks are used only for target selection and measurement. The model never receives a
GT mask. Each selected image contributes one target so paired statistics are image-independent.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from ultralytics import YOLO


COCO80 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def truthy(value: str) -> bool:
    return str(value).lower() == "true"


def select_balanced(rows: list[dict], n: int, used_images: set[int], rng: random.Random) -> list[dict]:
    by_size: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if int(r["image_id"]) not in used_images:
            by_size[r.get("area_bin", "unknown")].append(r)
    for pool in by_size.values():
        rng.shuffle(pool)
    chosen: list[dict] = []
    order = ["small", "medium", "large"]
    while len(chosen) < n:
        progressed = False
        for size in order:
            pool = by_size.get(size, [])
            while pool and int(pool[-1]["image_id"]) in used_images:
                pool.pop()
            if pool and len(chosen) < n:
                item = pool.pop()
                used_images.add(int(item["image_id"]))
                chosen.append(item)
                progressed = True
        if not progressed:
            break
    return chosen


def build_selection(instance_rows: list[dict], residual_rows: list[dict], per_cohort: int, seed: int) -> list[dict]:
    residual = {int(r["annotation_id"]): r["residual_state"] for r in residual_rows}
    pools: dict[str, list[dict]] = defaultdict(list)
    for r in instance_rows:
        aid = int(r["annotation_id"])
        scope = r["failure_scope"]
        if scope == "no_final_slot":
            pools["no_final_slot"].append(r)
        elif scope == "box_limited_mask_bad" and aid in residual:
            pools[f"joint_{residual[aid]}"].append(r)
        elif scope == "box_good_support_sufficient_mask_bad":
            pools["boxgood_supported_maskbad"].append(r)
        elif scope == "box_good_support_sufficient_mask_good" and r["task_mask75"] == "hit":
            pools["success_control"].append(r)

    cohort_order = [
        "no_final_slot",
        "joint_rescued_by_rectangle",
        "joint_requires_target_pixel_recovery",
        "joint_sufficient_true_pixels_but_residual_false_pixels",
        "boxgood_supported_maskbad",
        "success_control",
    ]
    rng = random.Random(seed)
    used_images: set[int] = set()
    selected: list[dict] = []
    for cohort in cohort_order:
        chosen = select_balanced(pools[cohort], per_cohort, used_images, rng)
        if len(chosen) < per_cohort:
            raise RuntimeError(f"Only {len(chosen)} unique-image targets available for {cohort}")
        for r in chosen:
            selected.append(
                {
                    "cohort": cohort,
                    "image_id": int(r["image_id"]),
                    "annotation_id": int(r["annotation_id"]),
                    "category_id": int(r["category_id"]),
                    "area_bin": r["area_bin"],
                    "density_e4": r.get("density_e4", ""),
                    "historical_box_iou": r.get("box_iou", ""),
                    "historical_mask_iou": r.get("mask_iou", ""),
                    "historical_task_mask75": r.get("task_mask75", ""),
                    "historical_prediction_slot": int(r.get("prediction_slot", -1)),
                }
            )
    return selected


def square_crop(box_xywh: list[float], width: int, height: int, context: float) -> tuple[int, int, int, int]:
    x, y, w, h = box_xywh
    cx, cy = x + w / 2, y + h / 2
    side = max(8.0, max(w, h) * context)
    x0, y0 = int(np.floor(cx - side / 2)), int(np.floor(cy - side / 2))
    x1, y1 = int(np.ceil(cx + side / 2)), int(np.ceil(cy + side / 2))
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / max(aa + bb - inter, 1e-9))


def measure(result, category_id: int, target_box: np.ndarray, target_mask: np.ndarray,
            same_other: np.ndarray, all_other: np.ndarray, offset: tuple[int, int],
            forced_slot: int | None = None) -> dict:
    x0, y0 = offset
    if result.boxes is None or len(result.boxes) == 0 or result.masks is None:
        return {"candidate": 0, "box_iou": 0.0, "mask_iou": 0.0, "coverage": 0.0,
                "same_neighbor_fp": 0.0, "other_instance_fp": 0.0, "background_fp": 0.0,
                "score": 0.0}
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float64)
    boxes[:, [0, 2]] += x0
    boxes[:, [1, 3]] += y0
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = result.boxes.conf.detach().cpu().numpy()
    wanted = np.array([0 <= c < len(COCO80) and COCO80[c] == category_id for c in classes], dtype=bool)
    if forced_slot is not None:
        if forced_slot < 0 or forced_slot >= len(boxes) or not wanted[forced_slot]:
            return {"candidate": 0, "box_iou": 0.0, "mask_iou": 0.0, "coverage": 0.0,
                    "same_neighbor_fp": 0.0, "other_instance_fp": 0.0, "background_fp": 0.0,
                    "score": 0.0}
        wanted[:] = False
        wanted[forced_slot] = True
    if not wanted.any():
        return {"candidate": 0, "box_iou": 0.0, "mask_iou": 0.0, "coverage": 0.0,
                "same_neighbor_fp": 0.0, "other_instance_fp": 0.0, "background_fp": 0.0,
                "score": 0.0}
    indices = np.flatnonzero(wanted)
    ious = np.array([box_iou(boxes[i], target_box) for i in indices])
    best = int(indices[int(ious.argmax())])
    pred_local = result.masks.data[best].detach().cpu().numpy() > 0.5
    h, w = target_mask.shape
    ph, pw = pred_local.shape
    if (ph, pw) != (h, w) and (x0, y0) == (0, 0):
        pred_local = cv2.resize(pred_local.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    if (x0, y0) == (0, 0) and pred_local.shape == target_mask.shape:
        pred = pred_local
    else:
        pred = np.zeros_like(target_mask, dtype=bool)
        y1, x1 = min(h, y0 + pred_local.shape[0]), min(w, x0 + pred_local.shape[1])
        pred[y0:y1, x0:x1] = pred_local[: y1 - y0, : x1 - x0]
    gt_area = int(target_mask.sum())
    inter = int((pred & target_mask).sum())
    pred_area = int(pred.sum())
    union = gt_area + pred_area - inter
    fp = pred & ~target_mask
    same_fp = int((fp & same_other).sum())
    other_fp = int((fp & all_other & ~same_other).sum())
    bg_fp = int((fp & ~all_other).sum())
    return {
        "candidate": 1,
        "box_iou": float(ious.max()),
        "mask_iou": inter / max(union, 1),
        "coverage": inter / max(gt_area, 1),
        "same_neighbor_fp": same_fp / max(gt_area, 1),
        "other_instance_fp": other_fp / max(gt_area, 1),
        "background_fp": bg_fp / max(gt_area, 1),
        "score": float(scores[best]),
    }


def infer(model: YOLO, image: np.ndarray, imgsz: int):
    return model.predict(
        source=image, imgsz=imgsz, conf=0.001, iou=0.7, max_det=300, device=0,
        retina_masks=True, verbose=False, end2end=False,
    )[0]


def bootstrap_ci(values: np.ndarray, seed: int = 0, reps: int = 2000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_target: dict[tuple[int, str], dict] = {}
    for r in rows:
        groups[(r["cohort"], r["arm"])].append(r)
        by_target[(int(r["annotation_id"]), r["arm"])] = r
    out: list[dict] = []
    for (cohort, arm), items in sorted(groups.items()):
        arr = lambda k: np.asarray([float(x[k]) for x in items], dtype=np.float64)
        base = [by_target[(int(x["annotation_id"]), "full640")] for x in items]
        delta = arr("mask_iou") - np.asarray([float(x["mask_iou"]) for x in base])
        lo, hi = bootstrap_ci(delta)
        old75 = np.asarray([float(x["mask_iou"]) >= 0.75 for x in base])
        new75 = arr("mask_iou") >= 0.75
        out.append({
            "cohort": cohort, "arm": arm, "n": len(items),
            "candidate_rate": arr("candidate").mean(),
            "box50_rate": (arr("box_iou") >= 0.50).mean(),
            "box75_rate": (arr("box_iou") >= 0.75).mean(),
            "mask75_rate": new75.mean(),
            "mask_iou_mean": arr("mask_iou").mean(),
            "delta_mask_iou_vs_full640": delta.mean(),
            "delta_mask_iou_ci_low": lo, "delta_mask_iou_ci_high": hi,
            "recovered75": int((~old75 & new75).sum()), "lost75": int((old75 & ~new75).sum()),
            "coverage_mean": arr("coverage").mean(),
            "same_neighbor_fp_mean": arr("same_neighbor_fp").mean(),
            "other_instance_fp_mean": arr("other_instance_fp").mean(),
            "background_fp_mean": arr("background_fp").mean(),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cohort", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="Debug cap after selection; 0 keeps all")
    args = ap.parse_args()

    outdir = Path(__file__).resolve().parents[1]
    root = outdir.parents[1]
    runtime = root / "shared/coco_clean_20260911/local_readout_runtime_20260912"
    ann_path = root / "assets/datasets/coco/annotations/instances_val2017.json"
    image_root = runtime / "data/images/val2017"
    weight = root / "assets/models/coco_clean_20260911/yolo26m-seg.pt"
    instances = root / "experiments/coco_failure_dimensions_20260913/instances_improved.csv"
    residuals = root / "experiments/failure_dimension_analysis_20260913/joint_failure_residuals.csv"

    selection = build_selection(read_csv(instances), read_csv(residuals), args.per_cohort, args.seed)
    if args.limit:
        selection = selection[: args.limit]
    write_csv(outdir / "selection.csv", selection)
    coco = COCO(str(ann_path))
    model = YOLO(str(weight))
    model.model.model[-1].end2end = False
    rows: list[dict] = []
    arms = [("full640", None, 640), ("full1280", None, 1280), ("roi_context4", 4.0, 640), ("roi_context2", 2.0, 640)]
    t0 = time.time()
    for index, item in enumerate(selection, 1):
        ann = coco.anns[int(item["annotation_id"])]
        info = coco.imgs[int(item["image_id"])]
        image = cv2.imread(str(image_root / info["file_name"]))
        if image is None:
            raise FileNotFoundError(image_root / info["file_name"])
        h, w = image.shape[:2]
        target = coco.annToMask(ann).astype(bool)
        same_other = np.zeros((h, w), dtype=bool)
        all_other = np.zeros((h, w), dtype=bool)
        for oid in coco.getAnnIds(imgIds=[int(item["image_id"])], iscrowd=None):
            if oid == int(item["annotation_id"]):
                continue
            other_ann = coco.anns[oid]
            if other_ann.get("iscrowd", 0):
                continue
            om = coco.annToMask(other_ann).astype(bool)
            all_other |= om
            if int(other_ann["category_id"]) == int(item["category_id"]):
                same_other |= om
        x, y, bw, bh = map(float, ann["bbox"])
        target_box = np.array([x, y, x + bw, y + bh], dtype=np.float64)
        for arm, context, imgsz in arms:
            if context is None:
                offset = (0, 0)
                source = image
                crop_box = (0, 0, w, h)
            else:
                crop_box = square_crop(ann["bbox"], w, h, context)
                x0, y0, x1, y1 = crop_box
                offset = (x0, y0)
                source = image[y0:y1, x0:x1]
            result = infer(model, source, imgsz)
            metrics = measure(
                result, int(item["category_id"]), target_box, target, same_other, all_other, offset,
                forced_slot=None,
            )
            rows.append({
                **item, "arm": arm, "imgsz": imgsz, "crop_context": context or 0,
                "crop_x0": crop_box[0], "crop_y0": crop_box[1], "crop_x1": crop_box[2], "crop_y1": crop_box[3],
                "target_long_side_at_input": max(bw, bh) * imgsz / max(source.shape[0], source.shape[1]),
                **metrics,
            })
        if index % 8 == 0 or index == len(selection):
            write_csv(outdir / "per_target.csv", rows)
            (outdir / "progress.json").write_text(json.dumps({"done": index, "total": len(selection), "elapsed_s": time.time()-t0}, indent=2), encoding="utf-8")
            print(f"[{index}/{len(selection)}] elapsed={time.time()-t0:.1f}s", flush=True)
    summary = summarize(rows)
    write_csv(outdir / "summary.csv", summary)
    (outdir / "COMPLETE.json").write_text(json.dumps({
        "status": "complete", "targets": len(selection), "arms": [a[0] for a in arms],
        "model": str(weight), "ultralytics": "8.4.100", "head_end2end": False,
        "oracle_roi": True, "elapsed_s": time.time()-t0,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
