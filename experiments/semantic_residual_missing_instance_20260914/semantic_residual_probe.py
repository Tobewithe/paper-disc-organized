"""Test whether YOLO26's discarded semantic branch retains evidence for missed instances.

The diagnostic compares GT instances with no final same-class Box50 slot against successful and
mask-only-failure controls. GT is used for cohort selection and measurement only. The tested signal
is available from the model at inference: class semantic logits not covered by retained class boxes.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from sklearn.metrics import roc_auc_score
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


def choose(rows: list[dict], per_cohort: int, seed: int) -> list[dict]:
    definitions = {
        "no_final_slot": lambda r: r["failure_scope"] == "no_final_slot",
        "boxgood_maskbad": lambda r: r["failure_scope"] == "box_good_support_sufficient_mask_bad",
        "success_control": lambda r: r["failure_scope"] == "box_good_support_sufficient_mask_good" and r["task_mask75"] == "hit",
    }
    rng = random.Random(seed)
    used: set[int] = set()
    selected: list[dict] = []
    for cohort, predicate in definitions.items():
        by_size: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if predicate(r) and int(r["image_id"]) not in used:
                by_size[r["area_bin"]].append(r)
        for pool in by_size.values():
            rng.shuffle(pool)
        picked: list[dict] = []
        while len(picked) < per_cohort:
            changed = False
            for size in ("small", "medium", "large"):
                pool = by_size[size]
                while pool and int(pool[-1]["image_id"]) in used:
                    pool.pop()
                if pool and len(picked) < per_cohort:
                    r = pool.pop()
                    used.add(int(r["image_id"]))
                    picked.append(r)
                    changed = True
            if not changed:
                break
        if len(picked) < per_cohort:
            raise RuntimeError(f"Insufficient unique images for {cohort}: {len(picked)}")
        for r in picked:
            selected.append({
                "cohort": cohort, "image_id": int(r["image_id"]),
                "annotation_id": int(r["annotation_id"]), "category_id": int(r["category_id"]),
                "area_bin": r["area_bin"], "density_e4": r["density_e4"],
            })
    return selected


def letterbox_mask(mask: np.ndarray, size: int = 640) -> np.ndarray:
    h, w = mask.shape
    scale = min(size / h, size / w)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    left, top = round(dw - 0.1), round(dh - 0.1)
    canvas = np.zeros((size, size), dtype=np.float32)
    canvas[top : top + nh, left : left + nw] = resized
    return cv2.resize(canvas, (80, 80), interpolation=cv2.INTER_AREA)


def original_box_to_grid(box: np.ndarray, shape: tuple[int, int], size: int = 640) -> tuple[int, int, int, int]:
    h, w = shape
    scale = min(size / h, size / w)
    nw, nh = round(w * scale), round(h * scale)
    left, top = round((size - nw) / 2 - 0.1), round((size - nh) / 2 - 0.1)
    q = box.copy().astype(np.float64)
    q[[0, 2]] = q[[0, 2]] * scale + left
    q[[1, 3]] = q[[1, 3]] * scale + top
    q /= 8.0
    return max(0, int(np.floor(q[0]))), max(0, int(np.floor(q[1]))), min(80, int(np.ceil(q[2]))), min(80, int(np.ceil(q[3])))


def bootstrap_mean(values: list[float], seed: int = 0, reps: int = 2000) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(x, len(x), replace=True).mean() for _ in range(reps)])
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cohort", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(__file__).resolve().parent
    root = out.parents[1]
    shared = root / "shared/coco_clean_20260911"
    sys.path.insert(0, str(shared))
    from structure_candidate_trace import TraceCapture

    ann_path = root / "assets/datasets/coco/annotations/instances_val2017.json"
    image_root = shared / "local_readout_runtime_20260912/data/images/val2017"
    weight = root / "assets/models/coco_clean_20260911/yolo26m-seg.pt"
    table = root / "diagnostics/coco_failure_dimensions_20260913/instances_improved.csv"
    selected = choose(read_csv(table), args.per_cohort, args.seed)
    if args.limit:
        selected = selected[: args.limit]
    write_csv(out / "selection.csv", selected)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ann_path))
    model = YOLO(str(weight))
    model.model.eval().requires_grad_(False)
    head = model.model.model[-1]
    head.end2end = False
    # BasePredictor fuses the PyTorch model. Preserve the trained semantic branch that Segment26.fuse
    # normally removes because the experiment explicitly measures that otherwise-discarded output.
    head.proto.fuse = lambda: None
    captured: dict[str, torch.Tensor] = {}

    def proto_pre_hook(module, hook_args):
        features = hook_args[0]
        feat = features[0]
        for i, refine in enumerate(module.feat_refine):
            up = refine(features[i + 1])
            up = torch.nn.functional.interpolate(up, size=feat.shape[2:], mode="nearest")
            feat = feat + up
        captured["semantic"] = module.semseg(feat).detach().float().cpu()

    hook = head.proto.register_forward_pre_hook(proto_pre_hook)
    rows: list[dict] = []
    started = time.time()
    for index, item in enumerate(selected, 1):
        ann = coco.anns[item["annotation_id"]]
        info = coco.imgs[item["image_id"]]
        path = image_root / info["file_name"]
        captured.clear()
        with torch.inference_mode():
            result = model.predict(
                str(path), predictor=TraceCapture, imgsz=640, rect=False, conf=0.001, iou=0.7,
                max_det=300, retina_masks=False, device=0, verbose=False, end2end=False,
            )[0]
        if "semantic" not in captured:
            raise RuntimeError("Semantic pre-hook did not run")
        logits = captured["semantic"][0]
        semantic_all = logits.sigmoid().numpy()
        class_index = COCO80.index(item["category_id"])
        score = semantic_all[class_index]
        target_mask = coco.annToMask(ann).astype(bool)
        target_soft = letterbox_mask(target_mask)
        target_cells = target_soft >= 0.20
        other_same = np.zeros_like(target_mask)
        for oid in coco.getAnnIds(imgIds=[item["image_id"]], catIds=[item["category_id"]], iscrowd=False):
            if oid != item["annotation_id"]:
                other_same |= coco.annToMask(coco.anns[oid]).astype(bool)
        other_same_cells = letterbox_mask(other_same) >= 0.20
        background = ~(target_cells | other_same_cells)
        labels = np.r_[np.ones(target_cells.sum(), dtype=np.uint8), np.zeros(background.sum(), dtype=np.uint8)]
        values = np.r_[score[target_cells], score[background]]
        auc = float(roc_auc_score(labels, values)) if target_cells.any() and background.any() else float("nan")

        covered = np.zeros((80, 80), dtype=bool)
        residual_all = semantic_all.copy()
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, cls in zip(boxes, classes):
                if 0 <= cls < len(COCO80) and COCO80[cls] == item["category_id"]:
                    x0, y0, x1, y1 = original_box_to_grid(box, target_mask.shape)
                    covered[y0:y1, x0:x1] = True
                if 0 <= cls < len(COCO80):
                    x0, y0, x1, y1 = original_box_to_grid(box, target_mask.shape)
                    residual_all[cls, y0:y1, x0:x1] = 0.0
        weights = target_soft / max(float(target_soft.sum()), 1e-9)
        residual_target_mass = float((score * weights * ~covered).sum())
        uncovered_fraction = float((weights * ~covered).sum())
        peak = np.unravel_index(int(np.argmax(score)), score.shape)
        ys, xs = np.where(target_cells)
        if len(xs):
            peak_distance = float(np.sqrt(((xs - peak[1]) ** 2 + (ys - peak[0]) ** 2).min()))
        else:
            peak_distance = float("nan")
        threshold = float(np.quantile(score, 0.99))
        top = score >= threshold
        valid_cells = letterbox_mask(np.ones_like(target_mask, dtype=bool)) >= 0.99
        attention = residual_all.max(axis=0)
        attention[~valid_cells] = 0.0
        local_max = attention >= cv2.dilate(attention, np.ones((5, 5), np.uint8)) - 1e-12
        candidates = np.argwhere(local_max & (attention > 0))
        candidates = sorted(candidates.tolist(), key=lambda q: float(attention[q[0], q[1]]), reverse=True)
        peaks: list[tuple[int, int]] = []
        for py, px in candidates:
            if all((py-qy)**2 + (px-qx)**2 >= 36 for qy, qx in peaks):
                peaks.append((py, px))
            if len(peaks) == 10:
                break
        hit_region = cv2.dilate(target_cells.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        hit_rank = next((rank for rank, (py, px) in enumerate(peaks, 1) if hit_region[py, px]), 0)
        rows.append({
            **item, "target_cells": int(target_cells.sum()), "semantic_auc_target_vs_background": auc,
            "semantic_target_mean": float((score * weights).sum()),
            "semantic_target_max": float(score[target_cells].max()) if target_cells.any() else 0.0,
            "semantic_background_mean": float(score[background].mean()),
            "uncovered_target_fraction": uncovered_fraction,
            "uncovered_semantic_target_mass": residual_target_mass,
            "top1pct_target_precision": float((top & target_cells).sum() / max(top.sum(), 1)),
            "semantic_peak_distance_cells": peak_distance,
            "retained_sameclass_boxes": int(sum(COCO80[int(c)] == item["category_id"] for c in (result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes is not None else []))),
            "residual_attention_target_max": float(attention[hit_region].max()) if hit_region.any() else 0.0,
            "residual_proposal_hit_rank": hit_rank,
            "residual_proposal_hit_at1": int(0 < hit_rank <= 1),
            "residual_proposal_hit_at3": int(0 < hit_rank <= 3),
            "residual_proposal_hit_at5": int(0 < hit_rank <= 5),
            "residual_proposal_hit_at10": int(0 < hit_rank <= 10),
        })
        if index % 24 == 0 or index == len(selected):
            write_csv(out / "per_target.csv", rows)
            (out / "progress.json").write_text(json.dumps({"done": index, "total": len(selected), "elapsed_s": time.time()-started}, indent=2), encoding="utf-8")
            print(f"[{index}/{len(selected)}] elapsed={time.time()-started:.1f}s", flush=True)
    hook.remove()
    metrics = [
        "semantic_auc_target_vs_background", "semantic_target_mean", "semantic_target_max",
        "uncovered_target_fraction", "uncovered_semantic_target_mass", "top1pct_target_precision",
        "semantic_peak_distance_cells", "retained_sameclass_boxes",
        "residual_attention_target_max", "residual_proposal_hit_at1", "residual_proposal_hit_at3",
        "residual_proposal_hit_at5", "residual_proposal_hit_at10",
    ]
    summary: list[dict] = []
    for cohort in sorted({r["cohort"] for r in rows}):
        rr = [r for r in rows if r["cohort"] == cohort]
        row = {"cohort": cohort, "n": len(rr)}
        for metric in metrics:
            mean, lo, hi = bootstrap_mean([float(r[metric]) for r in rr], seed=args.seed)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        summary.append(row)
    write_csv(out / "summary.csv", summary)
    (out / "COMPLETE.json").write_text(json.dumps({
        "status": "complete", "targets": len(rows), "per_cohort": args.per_cohort,
        "model": str(weight), "ultralytics": "8.4.100", "end2end": False,
        "semantic_branch_used_at_inference": True, "gt_used_for_measurement_only": True,
        "elapsed_s": time.time()-started,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
