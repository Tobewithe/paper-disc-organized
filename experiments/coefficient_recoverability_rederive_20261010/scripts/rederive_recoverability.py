"""Definition-complete re-derivation of fixed-prototype recoverability (STUDY_COEFFICIENT_RECOVERABILITY_REDERIVE_20261010).

Protocol: PROTOCOL.md (frozen before execution). The oracle is exactly the recorded 7D
convention: argmin_d BCE_GTbox(P(c0+d))/area + 0.003*||d||^2/2, LBFGS 120 iterations,
double precision, started from d=0. No support truncation, no variant search.

Inputs:
  --cache        directory with images/<iid>.npz (proto 32x112x160, coefficient, box_input in
                 448x640 input-pixel space, input_shape, original_shape, identity arrays) and
                 MANIFEST.json (identity cross-check)
  --annotations  COCO val2017 instances JSON (GT masks via polygon rasterization, cv2)
  --out          Run directory
  --limit        smoke: only the first N images

Outputs: PER_CANDIDATE.jsonl, PER_GT.jsonl, SUMMARY.json, COMPLETE.json, ENVIRONMENT.json,
DATA_RECEIPT.json, PROGRESS.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

LAMBDA = 0.003
ITERATIONS = 120
SCALE = 0.25  # decode grid per input pixel (160/640, 112/448)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Annotations:
    def __init__(self, path: Path):
        with path.open() as f:
            data = json.load(f)
        self.images = {i["id"]: i for i in data["images"]}
        self.by_image = defaultdict(list)
        for ann in data["annotations"]:
            self.by_image[ann["image_id"]].append(ann)

    def mask(self, image_id: int, ann: dict) -> np.ndarray:
        image = self.images[image_id]
        mask = np.zeros((image["height"], image["width"]), dtype=np.uint8)
        for polygon in ann["segmentation"]:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
        return mask


def geometry(input_shape, original_shape):
    ih, iw = int(input_shape[0]), int(input_shape[1])
    oh, ow = int(original_shape[0]), int(original_shape[1])
    gain = min(ih / oh, iw / ow)
    nh, nw = round(oh * gain), round(ow * gain)
    top = round((ih - nh) / 2 - 0.1)
    left = round((iw - nw) / 2 - 0.1)
    return gain, left, top, (ih, iw)


def gt_on_grid(gt: np.ndarray, geom, grid):
    gain, left, top, (ih, iw) = geom
    oh, ow = gt.shape
    nh, nw = round(oh * gain), round(ow * gain)
    tensor = torch.from_numpy(gt.astype(np.float32))[None, None]
    scaled = F.interpolate(tensor, (nh, nw), mode="nearest")
    padded = F.pad(scaled, (left, iw - nw - left, top, ih - nh - top))
    return F.interpolate(padded, grid, mode="nearest")[0, 0]


def solve(c0, p, y, area, penalty=LAMBDA, iterations=ITERATIONS):
    p, y, c0 = p.double().cuda(), y.double().cuda(), c0.double().cuda()

    def objective(delta):
        logits = p @ (c0 + delta)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="sum") / area
        return bce + penalty * delta.square().sum() / 2

    delta = torch.zeros_like(c0, requires_grad=True)
    optimizer = torch.optim.LBFGS([delta], lr=1, max_iter=iterations,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                                  tolerance_change=1e-12)

    def closure():
        optimizer.zero_grad()
        loss = objective(delta)
        loss.backward()
        return loss

    optimizer.step(closure)
    value = objective(delta)
    gradient = torch.autograd.grad(value, delta)[0]
    return delta.detach().float().cpu(), float(value.detach()), float(gradient.norm()), int(optimizer.state[delta]["n_iter"])


def decode_iou(proto, c, box_input, gt_grid):
    z = (c.cuda() @ proto.cuda().flatten(1)).reshape(proto.shape[1], proto.shape[2])
    x1, y1, x2, y2 = (box_input * SCALE).tolist()
    rr = torch.arange(proto.shape[2], device="cuda")[None, :]
    cc = torch.arange(proto.shape[1], device="cuda")[:, None]
    support = (rr >= x1) & (rr < x2) & (cc >= y1) & (cc < y2)
    pred, truth = z > 0, gt_grid.cuda() > 0.5
    inter = int((pred & truth & support).sum())
    union = int(((pred | truth) & support).sum())
    return inter / max(union, 1)


def main(a):
    torch.manual_seed(0)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.cache / "MANIFEST.json").read_text())
    manifest_rows = {(r["image_id"], r["annotation_id"]): r for r in manifest["rows"]}
    annotations = Annotations(a.annotations)
    files = sorted((a.cache / "images").glob("*.npz"))
    if a.limit:
        files = files[: a.limit]

    write_json(a.out / "DATA_RECEIPT.json", {
        "cache": {"path": str(a.cache), "images": len(files),
                  "manifest_sha256": file_hash(a.cache / "MANIFEST.json")},
        "annotations": {"path": str(a.annotations), "sha256": file_hash(a.annotations)},
        "oracle_definition": {"lambda": LAMBDA, "iterations": ITERATIONS, "roi": "GT box on decode grid",
                              "solver": "LBFGS double from zero", "decode": "predicted box, logit > 0"},
    })
    write_json(a.out / "ENVIRONMENT.json", {
        "python": sys.executable, "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(0), "tf32": False,
        "opencv": cv2.__version__,
    })

    candidate_rows, gt_rows = [], []
    seen_gt, seen_events = set(), 0
    start = time.monotonic()
    for position, path in enumerate(files, 1):
        image_id = int(path.stem)
        data = np.load(path, allow_pickle=True)
        proto = torch.from_numpy(np.asarray(data["proto"])).float()
        grid = (proto.shape[1], proto.shape[2])
        geom = geometry(data["input_shape"], data["original_shape"])
        ann_ids = np.asarray(data["annotation_id"])
        raw_ids = np.asarray(data["raw_id"])
        boxes = np.asarray(data["box_input"])
        coeffs = np.asarray(data["coefficient"])
        levels = np.asarray(data["pyramid_level"])
        areas = np.asarray(data["area"])
        gt_cache = {}

        for i in range(len(ann_ids)):
            ann_id = int(ann_ids[i])
            if ann_id not in gt_cache:
                ann = next(x for x in annotations.by_image[image_id] if x["id"] == ann_id)
                gt_cache[ann_id] = (annotations.mask(image_id, ann), ann)
            gt, ann = gt_cache[ann_id]
            gt_grid = gt_on_grid(gt, geom, grid)
            c0 = torch.from_numpy(coeffs[i]).float()
            gain, left, top, _ = geom
            scale = grid[1] / geom[3][1]
            bx, by, bw, bh = ann["bbox"]
            box_grid = [(bx * gain + left) * scale, (by * gain + top) * scale,
                        ((bx + bw) * gain + left) * scale, ((by + bh) * gain + top) * scale]
            rr = torch.arange(grid[1])[None, :]
            cc = torch.arange(grid[0])[:, None]
            support = (rr >= box_grid[0]) & (rr < box_grid[2]) & (cc >= box_grid[1]) & (cc < box_grid[3])
            p = proto[:, support].T.contiguous()
            y = gt_grid[support]
            if len(y) == 0:
                continue
            delta, objective_value, stationarity, iterations = solve(c0, p, y, float(len(y)))
            oracle_iou = decode_iou(proto, c0 + delta, torch.from_numpy(boxes[i]).float(), gt_grid)
            baseline_iou = decode_iou(proto, c0, torch.from_numpy(boxes[i]).float(), gt_grid)
            area = float(areas[i])
            candidate_rows.append(dict(
                image_id=image_id, annotation_id=ann_id, raw_id=int(raw_ids[i]),
                pyramid_level=int(levels[i]), area=area,
                area_group=("small" if area < 1024 else "medium" if area < 9216 else "large"),
                baseline_iou=baseline_iou, oracle_iou=oracle_iou,
                delta_norm=float(delta.norm()), objective=objective_value,
                stationary_norm=stationarity, iterations=iterations))
            seen_gt.add((image_id, ann_id))
            seen_events += 1
        if position % 100 == 0 or position == len(files):
            state = dict(images=position, total=len(files), candidates=seen_events,
                         elapsed=round(time.monotonic() - start, 1))
            print(json.dumps(state), flush=True)
            write_json(a.out / "PROGRESS.json", state)

    by_gt = defaultdict(list)
    for row in candidate_rows:
        by_gt[(row["image_id"], row["annotation_id"])].append(row)
    for (image_id, ann_id), rows in by_gt.items():
        oracle_best = max(r["oracle_iou"] for r in rows)
        area = rows[0]["area"]
        gt_rows.append(dict(image_id=image_id, annotation_id=ann_id, raw_count=len(rows),
                            baseline_best=max(r["baseline_iou"] for r in rows),
                            oracle_best=oracle_best, recoverable=bool(oracle_best >= 0.75),
                            area=area,
                            area_group=("small" if area < 1024 else "medium" if area < 9216 else "large")))
    recoverable = sum(1 for g in gt_rows if g["recoverable"])
    strata = {}
    for group in ("small", "medium", "large"):
        sub = [g for g in gt_rows if g["area_group"] == group]
        strata[group] = dict(gt=len(sub), recoverable=sum(1 for g in sub if g["recoverable"]))
    summary = dict(
        definition={"lambda": LAMBDA, "iterations": ITERATIONS, "roi": "GT box", "solver": "LBFGS double from zero"},
        images=len(files), candidates=len(candidate_rows), gt=len(gt_rows),
        recoverable=recoverable, recoverable_share=recoverable / max(len(gt_rows), 1),
        strata=strata,
        recorded={"gt": 6459, "recoverable": 4263, "share": 0.6600,
                  "strata": {"small": [3888, 2285], "medium": [1742, 1237], "large": [829, 741]}},
        difference={"gt": len(gt_rows) - 6459, "recoverable": recoverable - 4263},
        manifest_check={"gt": len(seen_gt), "events": seen_events,
                        "manifest_gt": manifest["strict_gt_total"], "manifest_events": manifest["selected_raw_events"]},
    )
    write_json(a.out / "SUMMARY.json", summary)
    write_json(a.out / "COMPLETE.json", {"status": "complete", "summary": summary})
    with (a.out / "PER_CANDIDATE.jsonl").open("w", encoding="utf-8") as sink:
        for row in candidate_rows:
            sink.write(json.dumps(row, allow_nan=False) + "\n")
    with (a.out / "PER_GT.jsonl").open("w", encoding="utf-8") as sink:
        for row in gt_rows:
            sink.write(json.dumps(row, allow_nan=False) + "\n")
    print("REDERIVE_COMPLETE " + json.dumps({k: summary[k] for k in
          ("gt", "candidates", "recoverable", "recoverable_share", "difference", "manifest_check")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
