"""Full COCO val2017 final-output relation taxonomy.

This is a post-hoc diagnostic over frozen final predictions.  It does not
perform model inference and it does not claim to identify upstream stages
(raw candidate generation, ranking, confidence filtering) when those
intermediate candidates are not present in the cache.

Primary relation (mutually exclusive, per ordinary GT):
  C/I/L/S: one GT <-> one same-class prediction, split by coverage/purity.
  O: one GT <-> many same-class predictions.
  M: many GT <-> one same-class prediction.
  X: many GT <-> many same-class predictions.
  MISS: no same-class core edge.

Secondary fields are deliberately orthogonal:
  relation_subtype, error_surface, density/size, weak and wrong-class flags.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from pycocotools import mask as mask_utils


ROOT = Path(__file__).resolve().parent
DEFAULT_ANN = ROOT.parents[1] / "datasets" / "coco" / "annotations" / "instances_val2017.json"
DEFAULT_SHARDS = ROOT / "diagnostics" / "frozen_readouts_fullval_20260912_v2" / "images"
DEFAULT_DENSITY = ROOT / "diagnostics" / "mask_geometry_failure_census_20260912" / "instances_classified.csv"


def bbox_xywh_to_xyxy(b):
    x, y, w, h = map(float, b)
    return x, y, x + max(0.0, w), y + max(0.0, h)


def bbox_overlap(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def decode_segmentation(seg) -> np.ndarray:
    m = mask_utils.decode(seg)
    if m.ndim == 3:
        m = np.any(m, axis=2)
    return np.asarray(m, dtype=bool)


def ann_mask(ann, h: int, w: int) -> np.ndarray:
    seg = ann.get("segmentation")
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict):
        rle = seg
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, h, w)
    else:
        return np.zeros((h, w), dtype=bool)
    return decode_segmentation(rle)


def load_density(path: Path) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                aid = int(r["annotation_id"])
            except Exception:
                continue
            out[aid] = r
    return out


def connected_components(gt_n: int, pred_n: int, edges: List[Tuple[int, int]]) -> List[Tuple[List[int], List[int]]]:
    g_adj = [[] for _ in range(gt_n)]
    p_adj = [[] for _ in range(pred_n)]
    for gi, pi in edges:
        g_adj[gi].append(pi)
        p_adj[pi].append(gi)
    seen_g = set()
    seen_p = set()
    comps = []
    for start in range(gt_n):
        if start in seen_g or not g_adj[start]:
            continue
        gs, ps = [], []
        stack = [("g", start)]
        seen_g.add(start)
        while stack:
            typ, idx = stack.pop()
            if typ == "g":
                gs.append(idx)
                for pi in g_adj[idx]:
                    if pi not in seen_p:
                        seen_p.add(pi)
                        stack.append(("p", pi))
            else:
                ps.append(idx)
                for gi in p_adj[idx]:
                    if gi not in seen_g:
                        seen_g.add(gi)
                        stack.append(("g", gi))
        comps.append((sorted(gs), sorted(ps)))
    return comps


def quality_relation(cov: float, purity: float) -> str:
    if cov >= 0.75 and purity >= 0.75:
        return "C"
    if cov < 0.75 and purity >= 0.75:
        return "I"
    if cov >= 0.75 and purity < 0.75:
        return "L"
    return "S"


def dominant_surface(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    same_other: np.ndarray,
    diff_other: np.ndarray,
    all_gt: np.ndarray,
) -> Tuple[str, dict]:
    own = max(int(gt_mask.sum()), 1)
    pred_n = max(int(pred_mask.sum()), 1)
    fn = int(np.count_nonzero(gt_mask & ~pred_mask))
    same = int(np.count_nonzero(pred_mask & same_other))
    diff = int(np.count_nonzero(pred_mask & diff_other))
    bg = int(np.count_nonzero(pred_mask & ~all_gt))
    vals = {
        "target_fn": fn / own,
        "same_neighbor": same / pred_n,
        "other_neighbor": diff / pred_n,
        "background": bg / pred_n,
    }
    order = sorted(vals.items(), key=lambda x: x[1], reverse=True)
    top, second = order[0], order[1]
    # Preserve the full fractions; the label is only a descriptive dominant
    # surface and is not presented as a causal attribution.
    if top[1] == 0:
        label = "none"
    elif second[1] >= 0.8 * top[1] and second[1] >= 0.10:
        label = "mixed"
    else:
        label = top[0]
    return label, {
        "target_fn_fraction": vals["target_fn"],
        "same_neighbor_fraction": vals["same_neighbor"],
        "other_neighbor_fraction": vals["other_neighbor"],
        "background_fraction": vals["background"],
        "pred_pixels": pred_n,
        "gt_pixels": own,
    }


def relation_subtype(gs: List[int], ps: List[int], pair_iou: Dict[Tuple[int, int], float],
                     pair_cov: Dict[Tuple[int, int], float], gt_masks, pred_masks) -> str:
    ng, npred = len(gs), len(ps)
    if ng == 1 and npred == 1:
        return "one_to_one"
    if ng == 1 and npred > 1:
        gi = gs[0]
        covs = [pair_cov.get((gi, pi), 0.0) for pi in ps]
        max_cov = max(covs) if covs else 0.0
        union = np.zeros_like(gt_masks[gi], dtype=bool)
        for pi in ps:
            union |= pred_masks[pi]
        union_cov = float(np.count_nonzero(union & gt_masks[gi])) / max(int(gt_masks[gi].sum()), 1)
        pair_vals = [
            pair_iou.get((a, b), 0.0)
            for i, a in enumerate(ps)
            for b in ps[i + 1:]
        ]
        max_pair = max(pair_vals) if pair_vals else 0.0
        if union_cov - max_cov >= 0.10:
            return "split_like"
        if max_pair >= 0.80:
            return "duplicate_like"
        return "multi_output_complex"
    if ng > 1 and npred == 1:
        return "merge_like"
    return "many_to_many_mixed"


def density_fields(ann_id: int, density: Dict[int, dict]) -> dict:
    r = density.get(ann_id, {})
    return {
        "ici": r.get("ici", ""),
        "ici_high": r.get("ici_high", ""),
        "mask_density_e4": r.get("mask_density", ""),
        "same_boundary_exposure4": r.get("same_boundary_exposure4", ""),
        "different_boundary_exposure4": r.get("different_boundary_exposure4", ""),
        "area_bin": r.get("area_bin", ""),
    }


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=Path, default=DEFAULT_ANN)
    ap.add_argument("--shards", type=Path, default=DEFAULT_SHARDS)
    ap.add_argument("--density", type=Path, default=DEFAULT_DENSITY)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug limit by image count; 0 means all")
    ap.add_argument("--score-min", type=float, default=0.0, help="drop final predictions below this score before relation graph construction")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.ann.read_text(encoding="utf-8"))
    images = {int(im["id"]): (int(im["height"]), int(im["width"])) for im in data["images"]}
    anns_by_image = defaultdict(list)
    cats = {}
    for c in data["categories"]:
        cats[int(c["id"])] = c.get("name", str(c["id"]))
    for ann in data["annotations"]:
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        if int(ann.get("ignore", 0)) == 1:
            continue
        anns_by_image[int(ann["image_id"])].append(ann)
    density = load_density(args.density)

    shard_paths = sorted(args.shards.glob("*.json.gz"), key=lambda p: int(p.name.split(".")[0]))
    if args.limit:
        shard_paths = shard_paths[:args.limit]
    gt_rows = []
    pred_rows = []
    summary = Counter()
    per_group = defaultdict(Counter)
    subtype_counts = Counter()
    surface_counts = Counter()
    image_count = 0
    pred_total = 0
    ordinary_gt_total = 0
    weak_total = 0
    wrong_class_total = 0

    for si, shard in enumerate(shard_paths, 1):
        image_id = int(shard.name.split(".")[0])
        if image_id not in images:
            continue
        h, w = images[image_id]
        anns = anns_by_image[image_id]
        gt_masks = [ann_mask(a, h, w) for a in anns]
        gt_bbox = [bbox_xywh_to_xyxy(a["bbox"]) for a in anns]
        gt_cat = [int(a["category_id"]) for a in anns]
        gt_area = [int(m.sum()) for m in gt_masks]
        ordinary_gt_total += len(anns)

        with gzip.open(shard, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        raw_preds = payload.get("original", [])
        raw_boxes = payload.get("original_boxes", [])
        preds = []
        boxes = []
        for idx, p in enumerate(raw_preds):
            b = raw_boxes[idx] if idx < len(raw_boxes) else None
            score = float(p.get("score", b.get("score", 0.0) if b else 0.0))
            if score >= args.score_min:
                preds.append(p)
                boxes.append(b)
        if len(boxes) != len(preds):
            boxes = [None] * len(preds)
        pred_total += len(preds)
        pred_masks = []
        pred_bbox = []
        pred_cat = []
        pred_score = []
        for p, b in zip(preds, boxes):
            seg = p.get("segmentation")
            try:
                pm = decode_segmentation(seg)
            except Exception:
                pm = np.zeros((h, w), dtype=bool)
            if pm.shape != (h, w):
                # This should not happen for the frozen COCO cache; retain a
                # safe diagnostic record rather than silently broadcasting.
                fixed = np.zeros((h, w), dtype=bool)
                hh, ww = min(h, pm.shape[0]), min(w, pm.shape[1])
                fixed[:hh, :ww] = pm[:hh, :ww]
                pm = fixed
            pred_masks.append(pm)
            if b and b.get("bbox") is not None:
                pred_bbox.append(bbox_xywh_to_xyxy(b["bbox"]))
            else:
                ys, xs = np.where(pm)
                pred_bbox.append((float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
                                 if len(xs) else (0.0, 0.0, 0.0, 0.0))
            pred_cat.append(int(p.get("category_id", b.get("category_id", -1) if b else -1)))
            pred_score.append(float(p.get("score", b.get("score", 0.0) if b else 0.0)))

        # Pairwise geometry.  Bbox filtering keeps this tractable; all masks
        # are still decoded once and every bbox-overlapping pair is evaluated.
        pair_iou, pair_cov, pair_purity = {}, {}, {}
        edges = []
        weak_same = defaultdict(list)
        wrong_core = defaultdict(list)
        any_bbox = defaultdict(list)
        for pi, pm in enumerate(pred_masks):
            parea = int(pm.sum())
            if parea == 0:
                continue
            for gi, gm in enumerate(gt_masks):
                if not bbox_overlap(pred_bbox[pi], gt_bbox[gi]):
                    continue
                any_bbox[gi].append(pi)
                inter = int(np.count_nonzero(pm & gm))
                if inter == 0:
                    continue
                union = gt_area[gi] + parea - inter
                iou = inter / max(union, 1)
                cov = inter / max(gt_area[gi], 1)
                purity = inter / max(parea, 1)
                pair_iou[(gi, pi)] = iou
                pair_cov[(gi, pi)] = cov
                pair_purity[(gi, pi)] = purity
                if pred_cat[pi] == gt_cat[gi]:
                    if iou >= 0.50 or cov >= 0.50:
                        edges.append((gi, pi))
                    elif purity >= 0.50 and cov >= 0.10:
                        weak_same[gi].append(pi)
                else:
                    if iou >= 0.50 or cov >= 0.50:
                        wrong_core[gi].append(pi)

        comps = connected_components(len(anns), len(preds), edges)
        gt_comp = {}
        pred_comp = {}
        for ci, (gs, ps) in enumerate(comps):
            for gi in gs:
                gt_comp[gi] = (ci, gs, ps)
            for pi in ps:
                pred_comp[pi] = (ci, gs, ps)

        # Classify GT rows.
        for gi, ann in enumerate(anns):
            if gi not in gt_comp:
                relation = "MISS"
                gs, ps = [gi], []
                subtype = "no_same_class_core_edge"
                same_weak = bool(weak_same.get(gi))
                wrong = bool(wrong_core.get(gi))
                if same_weak:
                    weak_total += 1
                if wrong:
                    wrong_class_total += 1
                surface = "unresolved_without_core_prediction"
                metrics = {
                    "target_fn_fraction": 1.0,
                    "same_neighbor_fraction": 0.0,
                    "other_neighbor_fraction": 0.0,
                    "background_fraction": 0.0,
                    "pred_pixels": 0,
                    "gt_pixels": gt_area[gi],
                }
                ci = ""
            else:
                ci, gs, ps = gt_comp[gi]
                ng, npred = len(gs), len(ps)
                if ng == 1 and npred == 1:
                    pi = ps[0]
                    cov = pair_cov[(gi, pi)]
                    purity = pair_purity[(gi, pi)]
                    relation = quality_relation(cov, purity)
                    subtype = "one_to_one"
                elif ng == 1 and npred > 1:
                    relation = "O"
                    subtype = relation_subtype(gs, ps, pair_iou, pair_cov, gt_masks, pred_masks)
                elif ng > 1 and npred == 1:
                    relation = "M"
                    subtype = "merge_like"
                else:
                    relation = "X"
                    subtype = "many_to_many_mixed"
                union_pred = np.zeros((h, w), dtype=bool)
                for pi in ps:
                    union_pred |= pred_masks[pi]
                own = gt_masks[gi]
                same_other = np.zeros((h, w), dtype=bool)
                diff_other = np.zeros((h, w), dtype=bool)
                for gj, gm in enumerate(gt_masks):
                    if gj == gi:
                        continue
                    if gt_cat[gj] == gt_cat[gi]:
                        same_other |= gm
                    else:
                        diff_other |= gm
                all_gt = np.zeros((h, w), dtype=bool)
                for gm in gt_masks:
                    all_gt |= gm
                surface, metrics = dominant_surface(own, union_pred, same_other, diff_other, all_gt)
                same_weak = False
                wrong = False
                ci = ci
            d = density_fields(int(ann["id"]), density)
            group = d["mask_density_e4"] or "undefined"
            row = {
                "image_id": image_id,
                "annotation_id": int(ann["id"]),
                "category_id": gt_cat[gi],
                "category_name": cats.get(gt_cat[gi], str(gt_cat[gi])),
                "coco_area": float(ann.get("area", gt_area[gi])),
                "valid_gt_pixels": gt_area[gi],
                "relation": relation,
                "relation_subtype": subtype,
                "component_id": ci,
                "component_gt_count": len(gs),
                "component_pred_count": len(ps),
                "same_class_weak_candidate": same_weak,
                "wrong_class_core_candidate": wrong,
                "bbox_overlapping_prediction_count": len(any_bbox.get(gi, [])),
                "error_surface": surface,
                **metrics,
                **d,
                "candidate_stage": "UNRESOLVED_FINAL_CACHE_ONLY",
            }
            gt_rows.append(row)
            summary[relation] += 1
            subtype_counts[subtype] += 1
            surface_counts[surface] += 1
            per_group[group][relation] += 1

        # Prediction-level accounting.
        for pi in range(len(preds)):
            if pi in pred_comp:
                _, gs, ps = pred_comp[pi]
                if len(gs) == 1 and len(ps) == 1:
                    rel = "linked_one_to_one"
                elif len(gs) == 1:
                    rel = "O"
                elif len(ps) == 1:
                    rel = "M"
                else:
                    rel = "X"
                subtype = relation_subtype(gs, ps, pair_iou, pair_cov, gt_masks, pred_masks)
            else:
                rel = "FP_no_same_class_core"
                subtype = "wrong_class_core" if any(
                    pred_cat[pi] != gt_cat[gi] and (gi, pi) in pair_iou and
                    (pair_iou[(gi, pi)] >= 0.50 or pair_cov[(gi, pi)] >= 0.50)
                    for gi in range(len(anns))
                ) else "no_core_relation"
            pred_rows.append({
                "image_id": image_id,
                "prediction_index": pi,
                "category_id": pred_cat[pi],
                "category_name": cats.get(pred_cat[pi], str(pred_cat[pi])),
                "score": pred_score[pi],
                "relation": rel,
                "relation_subtype": subtype,
                "candidate_stage": "UNRESOLVED_FINAL_CACHE_ONLY",
            })

        image_count += 1
        if si % 100 == 0:
            print(f"[{si}/{len(shard_paths)}] images={image_count} gt={ordinary_gt_total} pred={pred_total}", flush=True)

    gt_fields = [
        "image_id", "annotation_id", "category_id", "category_name", "coco_area",
        "valid_gt_pixels", "relation", "relation_subtype", "component_id",
        "component_gt_count", "component_pred_count", "same_class_weak_candidate",
        "wrong_class_core_candidate", "bbox_overlapping_prediction_count",
        "error_surface", "target_fn_fraction", "same_neighbor_fraction",
        "other_neighbor_fraction", "background_fraction", "pred_pixels", "gt_pixels",
        "ici", "ici_high", "mask_density_e4", "same_boundary_exposure4",
        "different_boundary_exposure4", "area_bin", "candidate_stage",
    ]
    pred_fields = [
        "image_id", "prediction_index", "category_id", "category_name", "score",
        "relation", "relation_subtype", "candidate_stage",
    ]
    write_csv(args.out / "gt_relation_taxonomy.csv", gt_rows, gt_fields)
    write_csv(args.out / "prediction_relation_taxonomy.csv", pred_rows, pred_fields)

    group_report = {
        group: dict(sorted(counts.items()))
        for group, counts in sorted(per_group.items())
    }
    report = {
        "protocol": {
            "dataset": "MS COCO val2017",
            "images_processed": image_count,
            "ordinary_gt_excludes_iscrowd_and_ignore": True,
            "core_edge": "same category and (mask IoU >= 0.50 OR GT coverage >= 0.50)",
            "weak_edge": "same category and purity >= 0.50 and coverage >= 0.10, excluding core",
            "one_to_one_threshold": "coverage and purity both use 0.75",
            "score_min": args.score_min,
            "purity": "intersection / prediction pixels",
            "stage_limit": "final-output cache cannot distinguish RAW/TOPK/CONF; all stage labels remain UNRESOLVED_FINAL_CACHE_ONLY",
            "relation_scope": "O/M/X describe final connected-component topology, not causal mechanism",
        },
        "counts": {
            "images": image_count,
            "ordinary_gt": ordinary_gt_total,
            "final_predictions": pred_total,
            "gt_relation": dict(sorted(summary.items())),
            "prediction_relation": dict(sorted(Counter(r["relation"] for r in pred_rows).items())),
            "weak_same_class_miss_flags": weak_total,
            "wrong_class_core_miss_flags": wrong_class_total,
        },
        "relation_subtype": dict(sorted(subtype_counts.items())),
        "error_surface": dict(sorted(surface_counts.items())),
        "by_mask_density_group": group_report,
        "limitations": [
            "This is a descriptive classification of final outputs; it is not an upstream causal attribution.",
            "A MISS means no final same-class core edge, not that the network never generated a raw candidate.",
            "Crowd annotations are excluded from ordinary GT relation totals and should be reported separately.",
            "Background error is normalized by predicted pixels and should not be interpreted as stronger causality solely from its area.",
        ],
    }
    (args.out / "SUMMARY.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "PROTOCOL.json").write_text(json.dumps(report["protocol"], ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({
        "status": "complete",
        "images": image_count,
        "ordinary_gt": ordinary_gt_total,
        "predictions": pred_total,
        "gt_csv": str(args.out / "gt_relation_taxonomy.csv"),
        "prediction_csv": str(args.out / "prediction_relation_taxonomy.csv"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
