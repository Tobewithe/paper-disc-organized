"""Measure raw/Top-300 candidate coverage from a provenance-complete trace."""
from __future__ import annotations

import csv
import json
import sqlite3
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


DB = Path(r"C:\Dpan\codexproject\pigcv_research\data\analysis\pigcv_analysis.db")
MANIFEST = Path(r"C:\Dpan\document\model_datasets\datasets\piglife\derived\task05_v1\pig_coco_test_task05_v1.json")
TRACE = Path(__file__).resolve().parents[1] / "experiments" / "yolo26_raw_trace_piglife_full_20260905_v1"
OUT = Path(__file__).resolve().parents[1] / "experiments" / "yolo26_candidate_gate_20260905"
SCREEN = (216, 384)
MAX_DET = 300


def iou_boxes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[..., 0] * wh[..., 1]
    aa = np.prod(np.maximum(0.0, a[:, 2:] - a[:, :2]), axis=1)
    bb = np.prod(np.maximum(0.0, b[:, 2:] - b[:, :2]), axis=1)
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def decode_gt(annotation: dict, height: int, width: int):
    from pycocotools import mask as mask_utils

    seg = annotation["segmentation"]
    if isinstance(seg, list):
        return mask_utils.merge(mask_utils.frPyObjects(seg, height, width), intersect=False)
    rle = dict(seg)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("ascii")
    return rle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PigLife_public_test")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--output-prefix", default="piglife_raw_trace_candidate_coverage")
    args = parser.parse_args()
    from ultralytics.utils import ops
    from pycocotools import mask as mask_utils

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    images = {int(x["id"]): x for x in manifest["images"]}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in manifest["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        failures = [dict(r) for r in conn.execute(
            "SELECT dataset,image_id,annotation_id,primary_class,total_failure FROM gt_instances WHERE dataset=? AND total_failure=1",
            (args.dataset,),
        )]
    failures_by_image: dict[int, list[dict]] = defaultdict(list)
    for row in failures:
        failures_by_image[int(row["image_id"])].append(row)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    for image_id, targets in sorted(failures_by_image.items()):
        image_info = images[image_id]
        h, w = int(image_info["height"]), int(image_info["width"])
        anns = {int(a["id"]): a for a in anns_by_image[image_id]}
        npz = np.load(args.trace / "raw_candidates" / f"image_{image_id}_raw_candidates.npz")
        boxes = np.asarray(npz["boxes_xyxy"], dtype=np.float32)
        scores = np.asarray(npz["scores"], dtype=np.float32)
        coeff = torch.from_numpy(np.asarray(npz["mask_coefficients"], dtype=np.float32)).to(device)
        proto = torch.from_numpy(np.asarray(npz["prototype"], dtype=np.float32)).to(device)
        top_idx = np.asarray(npz["top_indices"], dtype=np.int64)
        top_set = set(top_idx.tolist())
        target_boxes = np.asarray([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in (anns[r["annotation_id"]] for r in targets)], dtype=np.float32)
        overlap = iou_boxes(boxes, target_boxes)
        screen_indices = set(top_idx.tolist())
        for j in range(len(targets)):
            inter = np.prod(np.maximum(0.0, np.minimum(boxes[:, 2:], target_boxes[j, 2:]) - np.maximum(boxes[:, :2], target_boxes[j, :2])), axis=1)
            area = max(1.0, np.prod(target_boxes[j, 2:] - target_boxes[j, :2]))
            screen_indices.update(np.flatnonzero(inter / area >= 0.05).tolist())
            screen_indices.update(np.argsort(-overlap[:, j])[:120].tolist())
        idx = np.asarray(sorted(screen_indices), dtype=np.int64)
        gt_small = []
        for row in targets:
            rle = decode_gt(anns[row["annotation_id"]], h, w)
            full = mask_utils.decode(rle).astype(np.uint8)
            gt_small.append(cv2.resize(full, (SCREEN[1], SCREEN[0]), interpolation=cv2.INTER_NEAREST))
        gt = torch.from_numpy(np.stack(gt_small)).to(device).flatten(1).float()
        gt_area = gt.sum(1).clamp_min(1.0)
        best = np.zeros((len(targets),), dtype=np.float32)
        best_top = np.zeros((len(targets),), dtype=np.float32)
        for start in range(0, len(idx), 128):
            part = idx[start:start + 128]
            b = torch.from_numpy(boxes[part]).to(device)
            sb = ops.scale_boxes((h, w), b.clone(), SCREEN, padding=False)
            masks = ops.process_mask_native(proto, coeff[part], sb, SCREEN).flatten(1).float()
            inter = masks @ gt.T
            union = masks.sum(1, keepdim=True) + gt_area[None, :] - inter
            vals = (inter / union.clamp_min(1.0)).detach().cpu().numpy()
            best = np.maximum(best, vals.max(axis=0))
            for j, candidate in enumerate(part.tolist()):
                if candidate in top_set:
                    best_top = np.maximum(best_top, vals[j])
        for j, target in enumerate(targets):
            rows.append({
                "dataset": target["dataset"], "image_id": int(target["image_id"]), "annotation_id": int(target["annotation_id"]),
                "primary_class": target["primary_class"], "screen_candidates": int(len(idx)),
                "raw_best_mask_iou": float(best[j]), "top300_best_mask_iou": float(best_top[j]),
                "raw_good": int(best[j] >= 0.50), "top300_good": int(best_top[j] >= 0.50),
            })
    out_csv = OUT / f"{args.output_prefix}.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "trace": str(args.trace), "dataset": args.dataset, "failed_gt": len(rows),
        "by_class": {},
        "interpretation_boundary": "Same-forward raw/Top-300 mask-IoU upper bounds on the complete PigLife test manifest; this is candidate availability evidence, not a causal intervention result.",
    }
    for cls in sorted({r["primary_class"] for r in rows}):
        subset = [r for r in rows if r["primary_class"] == cls]
        summary["by_class"][cls] = {"failed_gt": len(subset), "raw_good": sum(r["raw_good"] for r in subset), "top300_good": sum(r["top300_good"] for r in subset)}
    (OUT / f"{args.output_prefix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
