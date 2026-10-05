"""Locate which head input scale carries the causal recovery signal.

This uses the GT-assisted combined image intervention as a donor. Feature swaps are
diagnostic, preserve all trained weights, and are never presented as an inference method.
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
import torch.nn.functional as F
from pycocotools.coco import COCO
from ultralytics import YOLO
from ultralytics.utils import ops

from causal_image_probe import (COCO80, box_iou, clean_ring, input_box,
                                read_csv, remove_proximal_neighbors,
                                shift_target_contrast, write_csv, bootstrap)


def input_region(mask: np.ndarray) -> torch.Tensor:
    h, w = mask.shape
    gain = min(640 / h, 640 / w)
    rw, rh = round(w * gain), round(h * gain)
    left, top = round((640 - rw) / 2 - .1), round((640 - rh) / 2 - .1)
    result = np.zeros((640, 640), np.uint8)
    result[top:top + rh, left:left + rw] = cv2.resize(mask.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(result)[None, None].bool().cuda()


def metrics(raw: torch.Tensor, ann: dict, shape):
    label = COCO80.index(ann["category_id"])
    target = input_box(ann["bbox"], shape)
    boxes = ops.xywh2xyxy(raw[0, :4].T).float().cpu().numpy()
    scores = raw[0, 4:84].T.float().cpu().numpy()
    ious = box_iou(boxes, target)
    best = int(np.argmax(ious))
    valid = scores[:, label] >= .001
    best_true = int(np.argmax(np.where(valid, ious, -1))) if valid.any() else -1
    return {"best_any_box_iou": float(ious[best]), "best_any_true_score": float(scores[best, label]),
            "best_true_box_iou": float(ious[best_true]) if best_true >= 0 else 0.0,
            "best_true_score": float(scores[best_true, label]) if best_true >= 0 else 0.0,
            "raw_box50": int(ious[best] >= .5)}


def main():
    out = Path(__file__).resolve().parents[1]
    root = out.parents[1]
    shared = root / "shared/coco_clean_20260911"
    sys.path.insert(0, str(shared))
    from structure_candidate_trace import TraceCapture
    selected = read_csv(root / "experiments/small_raw_geometry_origin_20260914/selection.csv")
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(root / "assets/datasets/coco/annotations/instances_val2017.json"))
    images = shared / "local_readout_runtime_20260912/data/images/val2017"
    model = YOLO(str(root / "assets/models/coco_clean_20260911/yolo26m-seg.pt"))
    model.model.eval().requires_grad_(False); model.model.model[-1].end2end = False

    # Initialize predictor, then hook the exact active AutoBackend head.
    first = selected[0]; fi = coco.imgs[int(first["image_id"])]
    dummy = cv2.imread(str(images / fi["file_name"]))
    with torch.inference_mode():
        model.predict(dummy, predictor=TraceCapture, imgsz=640, rect=False, conf=.001, iou=.7,
                      max_det=300, retina_masks=False, device=0, verbose=False, end2end=False)
    head = model.predictor.model.model.model[-1]
    captured = {}
    hook = head.register_forward_pre_hook(lambda module, args: captured.update(x=tuple(v.detach().clone() for v in args[0])))

    def natural(image):
        captured.clear()
        with torch.inference_mode():
            model.predict(image, predictor=TraceCapture, imgsz=640, rect=False, conf=.001, iou=.7,
                          max_det=300, retina_masks=False, device=0, verbose=False, end2end=False)
        return model.predictor.dense.detach().clone(), tuple(v.detach().clone() for v in captured["x"])

    rows = []; start = time.time()
    arms = ["original", "edited", "p3_full", "p4_full", "p5_full", "p3_local", "p4_local", "p5_local", "all_local"]
    for number, item in enumerate(selected, 1):
        ann = coco.anns[int(item["annotation_id"])]; info = coco.imgs[int(item["image_id"])]
        image = cv2.imread(str(images / info["file_name"]))
        target = coco.annToMask(ann).astype(np.uint8)
        all_objects = np.zeros_like(target, np.uint8)
        for oid in coco.getAnnIds(imgIds=[ann["image_id"]], iscrowd=None):
            if not coco.anns[oid].get("iscrowd", 0): all_objects |= coco.annToMask(coco.anns[oid]).astype(np.uint8)
        no_neighbor, removed, _, _ = remove_proximal_neighbors(image, coco, ann, target, all_objects)
        edited, _, _ = shift_target_contrast(no_neighbor, target, all_objects, "increase")
        _, ring_radius = clean_ring(target, all_objects)
        radius = max(5, ring_radius * 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        local_original = cv2.dilate(target, kernel) > 0
        local640 = input_region(local_original)
        raw0, x0 = natural(image); raw1, x1 = natural(edited)
        with torch.inference_mode():
            replay0 = head._inference(head.forward_head(list(x0), **head.one2many))
            replay1 = head._inference(head.forward_head(list(x1), **head.one2many))
        parity0 = float((raw0 - replay0).abs().max()); parity1 = float((raw1 - replay1).abs().max())
        if parity0 > 1e-6 or parity1 > 1e-6: raise RuntimeError((parity0, parity1))
        raw_by_arm = {"original": raw0, "edited": raw1}
        with torch.inference_mode():
            for level in range(3):
                xf = list(x0); xf[level] = x1[level]
                raw_by_arm[f"p{level+3}_full"] = head._inference(head.forward_head(xf, **head.one2many))
                mask = F.adaptive_max_pool2d(local640.float(), x0[level].shape[-2:]).bool()
                xl = list(x0); xl[level] = torch.where(mask, x1[level], x0[level])
                raw_by_arm[f"p{level+3}_local"] = head._inference(head.forward_head(xl, **head.one2many))
            xa = []
            for level in range(3):
                mask = F.adaptive_max_pool2d(local640.float(), x0[level].shape[-2:]).bool()
                xa.append(torch.where(mask, x1[level], x0[level]))
            raw_by_arm["all_local"] = head._inference(head.forward_head(xa, **head.one2many))
        for arm in arms:
            rows.append({**item, "arm": arm, "removed_neighbor_pixels": removed,
                         "replay_parity": max(parity0, parity1), **metrics(raw_by_arm[arm], ann, (info["height"], info["width"]))})
        if number % 32 == 0 or number == len(selected):
            write_csv(out / "feature_localization.csv", rows)
            print(f"[{number}/{len(selected)}] {time.time()-start:.1f}s", flush=True)
    hook.remove()

    by = {(r["annotation_id"], r["arm"]): r for r in rows}
    summary = []
    for cohort in sorted({r["cohort"] for r in rows}):
        ids = sorted({r["annotation_id"] for r in rows if r["cohort"] == cohort})
        for arm in arms[1:]:
            rec = {"cohort": cohort, "arm": arm, "n": len(ids)}
            for metric in ["best_any_box_iou", "best_true_box_iou", "best_any_true_score", "raw_box50"]:
                d = [float(by[(aid, arm)][metric]) - float(by[(aid, "original")][metric]) for aid in ids]
                mean, lo, hi = bootstrap(d)
                rec[metric + "_delta"] = mean; rec[metric + "_ci_low"] = lo; rec[metric + "_ci_high"] = hi
            if cohort == "raw_geometry_small":
                rec["box50_recovered"] = sum(int(by[(aid, "original")]["raw_box50"]) == 0 and int(by[(aid, arm)]["raw_box50"]) == 1 for aid in ids)
            summary.append(rec)
    write_csv(out / "feature_localization_summary.csv", summary)
    (out / "FEATURE_LOCALIZATION_COMPLETE.json").write_text(json.dumps({"status":"complete", "targets":len(selected),
        "arms":arms, "gt_role":"donor intervention and local feature-region definition only",
        "elapsed_s":time.time()-start}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
