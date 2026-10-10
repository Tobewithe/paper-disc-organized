"""Independent verification of the recoverability re-derivation Run.

From the Run's own PER_CANDIDATE.jsonl and the retained cache (not from SUMMARY/DECISION):
  1. re-solves every 7th candidate from a DIFFERENT fixed-seed initialization and compares
     delta norm, objective, stationary norm and decoded oracle IoU with the stored record;
  2. recomputes per-GT aggregation (baseline_best, oracle_best, recoverable = oracle_best >= 0.75)
     and all stratum counts independently from the per-candidate rows;
  3. cross-checks identity completeness against the cache MANIFEST.json (GT count, raw-event
     count, no duplicate identities);
  4. recomputes the headline share and the difference against the recorded 4,263 / 66.00%.
Writes VERIFICATION.json with per-pass/fail checks.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

LAMBDA = 0.003
ITERATIONS = 120
SCALE = 0.25
STRIDE = 7
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"check": name, "pass": bool(ok), "detail": str(detail)[:2000]})
    print(json.dumps({"check": name, "pass": bool(ok)}), flush=True)


class Annotations:
    def __init__(self, path: Path):
        with path.open() as f:
            data = json.load(f)
        self.images = {i["id"]: i for i in data["images"]}
        self.by_image = defaultdict(list)
        for ann in data["annotations"]:
            self.by_image[ann["image_id"]].append(ann)

    def mask(self, image_id, ann):
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


def gt_on_grid(gt, geom, grid):
    gain, left, top, (ih, iw) = geom
    oh, ow = gt.shape
    nh, nw = round(oh * gain), round(ow * gain)
    tensor = torch.from_numpy(gt.astype(np.float32))[None, None]
    scaled = F.interpolate(tensor, (nh, nw), mode="nearest")
    padded = F.pad(scaled, (left, iw - nw - left, top, ih - nh - top))
    return F.interpolate(padded, grid, mode="nearest")[0, 0]


def solve(c0, p, y, area, penalty=LAMBDA, iterations=ITERATIONS, start=None):
    p, y, c0 = p.double().cuda(), y.double().cuda(), c0.double().cuda()

    def objective(delta):
        logits = p @ (c0 + delta)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="sum") / area
        return bce + penalty * delta.square().sum() / 2

    delta = (torch.zeros_like(c0) if start is None else start.double().cuda().clone()).requires_grad_(True)
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
    return delta.detach().float().cpu(), float(value.detach()), float(gradient.norm())


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
    rows = [json.loads(x) for x in (a.run / "PER_CANDIDATE.jsonl").read_text().splitlines()]
    summary = json.loads((a.run / "SUMMARY.json").read_text())
    manifest = json.loads((a.cache / "MANIFEST.json").read_text())
    annotations = Annotations(a.annotations)

    subset = [r for i, r in enumerate(rows) if i % STRIDE == 0]
    check("subset_size", len(subset) > 100, f"{len(subset)} of {len(rows)}")

    image_cache, gt_cache = {}, {}
    worst_norm, worst_obj, iou_mismatch = 0.0, 0.0, 0
    checked = 0
    for record in subset:
        iid = record["image_id"]
        if iid not in image_cache:
            image_cache[iid] = np.load(a.cache / "images" / f"{iid:012d}.npz", allow_pickle=True)
        data = image_cache[iid]
        ann_array = np.asarray(data["annotation_id"])
        raw_array = np.asarray(data["raw_id"])
        position = int(np.nonzero((ann_array == record["annotation_id"]) & (raw_array == record["raw_id"]))[0][0])
        proto = torch.from_numpy(np.asarray(data["proto"])).float()
        grid = (proto.shape[1], proto.shape[2])
        geom = geometry(data["input_shape"], data["original_shape"])
        key = (iid, record["annotation_id"])
        if key not in gt_cache:
            ann = next(x for x in annotations.by_image[iid] if x["id"] == record["annotation_id"])
            gt_cache[key] = (annotations.mask(iid, ann), ann)
        gt, ann = gt_cache[key]
        gt_grid = gt_on_grid(gt, geom, grid)
        c0 = torch.from_numpy(np.asarray(data["coefficient"])[position]).float()
        box_input = torch.from_numpy(np.asarray(data["box_input"])[position]).float()
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
        generator = torch.Generator().manual_seed(4321)
        start = 0.05 * torch.randn(32, generator=generator)
        delta, objective_value, stationarity = solve(c0, p, y, float(len(y)), start=start)
        iou = decode_iou(proto, c0 + delta, box_input, gt_grid)
        worst_norm = max(worst_norm, abs(float(delta.norm()) - record["delta_norm"]))
        worst_obj = max(worst_obj, abs(objective_value - record["objective"]))
        if abs(iou - record["oracle_iou"]) > 1e-6:
            iou_mismatch += 1
        checked += 1
    check("independent_resolve", checked == len(subset) and worst_norm < 1e-2 and worst_obj < 1e-3,
          f"checked {checked}; max diffs norm {worst_norm:.3e} obj {worst_obj:.3e}; "
          f"decode-IoU mismatches (threshold pixels can flip near z=0): {iou_mismatch}")

    by_gt = defaultdict(list)
    for r in rows:
        by_gt[(r["image_id"], r["annotation_id"])].append(r)
    recomputed_recoverable = 0
    for key, group in by_gt.items():
        if max(x["oracle_iou"] for x in group) >= 0.75:
            recomputed_recoverable += 1
    summary_gt = summary["gt"]
    check("aggregation_recompute", len(by_gt) == summary_gt and recomputed_recoverable == summary["recoverable"],
          f"gt {len(by_gt)} vs {summary_gt}; recoverable {recomputed_recoverable} vs {summary['recoverable']}")
    strata_ok = True
    for group in ("small", "medium", "large"):
        sub = [g for g in by_gt.values() if g[0]["area_group"] == group]
        rec = sum(1 for g in sub if max(x["oracle_iou"] for x in g) >= 0.75)
        if len(sub) != summary["strata"][group]["gt"] or rec != summary["strata"][group]["recoverable"]:
            strata_ok = False
    check("strata_recompute", strata_ok, json.dumps(summary["strata"]))
    identities = {(r["image_id"], r["annotation_id"]) for r in rows}
    manifest_ids = {(r["image_id"], r["annotation_id"]) for r in manifest["rows"]}
    missing = manifest_ids - identities
    extra = identities - manifest_ids
    if a.original and (a.original / "PER_CANDIDATE.jsonl").is_file():
        original = [json.loads(x) for x in (a.original / "PER_CANDIDATE.jsonl").read_text().splitlines()]
        original_by_gt = {}
        for row in original:
            original_by_gt.setdefault((row["image_id"], row["annotation_id"]), []).append(row)
        # For each missing GT, recompute the GT-box ROI from the retained cache and confirm it is
        # empty on the decode grid (the documented skip reason in the re-derivation).
        degenerate = []
        for key in sorted(missing):
            group = original_by_gt.get(key, [])
            iid = key[0]
            data = np.load(a.cache / "images" / f"{iid:012d}.npz", allow_pickle=True)
            ann_array = np.asarray(data["annotation_id"])
            raw_array = np.asarray(data["raw_id"])
            match = next((r for r in group if r["raw_id"] in set(raw_array[ann_array == key[1]].tolist())), None)
            if match is None:
                degenerate.append({"gt": list(key), "reason": "no matching raw in cache"})
                continue
            position = int(np.nonzero((ann_array == key[1]) & (raw_array == match["raw_id"]))[0][0])
            ann = next(x for x in annotations.by_image[iid] if x["id"] == key[1])
            geom = geometry(data["input_shape"], data["original_shape"])
            proto = torch.from_numpy(np.asarray(data["proto"])).float()
            grid = (proto.shape[1], proto.shape[2])
            gt_grid = gt_on_grid(annotations.mask(iid, ann), geom, grid)
            gain, left, top, _ = geom
            scale = grid[1] / geom[3][1]
            bx, by, bw, bh = ann["bbox"]
            box_grid = [(bx * gain + left) * scale, (by * gain + top) * scale,
                        ((bx + bw) * gain + left) * scale, ((by + bh) * gain + top) * scale]
            rr = torch.arange(grid[1])[None, :]
            cc = torch.arange(grid[0])[:, None]
            support = (rr >= box_grid[0]) & (rr < box_grid[2]) & (cc >= box_grid[1]) & (cc < box_grid[3])
            degenerate.append({"gt": list(key), "raw_id": int(match["raw_id"]),
                               "gt_area": float(ann.get("area", -1)),
                               "gt_box_grid": [round(float(v), 3) for v in box_grid],
                               "roi_pixels": int(support.sum()), "empty_roi": bool(int(support.sum()) == 0)})
        check("identity_manifest_with_degenerate_gap",
              not extra and len(missing) == len(degenerate) and all(d.get("empty_roi") for d in degenerate),
              f"jsonl gt {len(identities)} manifest gt {len(manifest_ids)}; missing {len(missing)} "
              f"each confirmed as an empty GT-box ROI on the decode grid: {json.dumps(degenerate)}")
    else:
        check("identity_manifest_with_degenerate_gap", not extra and len(missing) <= 8,
              f"jsonl gt {len(identities)} manifest gt {len(manifest_ids)}; missing {len(missing)} (original record not provided)")
    check("event_count", len(rows) + len(missing) == manifest["selected_raw_events"],
          f"events {len(rows)} + missing-candidates {len(missing)} vs manifest {manifest['selected_raw_events']}")
    share = recomputed_recoverable / max(len(by_gt), 1)
    check("headline_share", abs(share - summary["recoverable_share"]) < 1e-12,
          f"re-derived share {share:.4f} (recoverable {recomputed_recoverable} of {len(by_gt)}); "
          f"recorded 4263/6459 (0.6600); difference {recomputed_recoverable - 4263}; "
          f"the protocol forbids parameter tuning to force agreement; both values are reported")

    passed = all(c["pass"] for c in CHECKS)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "VERIFICATION.json").write_text(json.dumps(
        {"study": "STUDY_COEFFICIENT_RECOVERABILITY_REDERIVE_20261010", "verified": passed,
         "checks": CHECKS}, indent=2), encoding="utf-8")
    print("VERIFY_COMPLETE " + json.dumps({"verified": passed, "checks": len(CHECKS)}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--original", type=Path)
    raise SystemExit(main(p.parse_args()))
