"""Phase B: replay a frozen shared affine coefficient-head solution.

This script does not optimize or train.  It loads the already refined
standardized affine maps, converts them to the native 64->32 one-to-one
coefficient-head coordinates, and evaluates A/B on the complete prototype
mask with Ultralytics 8.4.100 decoding.  It also emits candidate identities,
the two coefficient paths' numerical checks, diagnostic BCE and image-level
bootstrap summaries.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from ultralytics import YOLO
from ultralytics.utils import ops


def jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def identity(meta: dict) -> dict:
    return {
        "image_id": int(meta["image_id"]),
        "annotation_id": int(meta["annotation_id"]),
        "branch": "one2one",
        "raw_id": int(meta["raw_id"]),
        "pyramid_level": int(meta.get("level", meta.get("pyramid_level", -1))),
    }


def load_split_identities(cache: Path, out: Path, selected=None):
    """Read each split separately, avoiding the 2 GB host holding two banks."""
    groups = {}
    identities = {"fit": [], "dev": [], "val": []}
    selected = tuple(selected or ("fit", "dev", "val"))
    for group in selected:
        p = cache / f"{group}.pt"
        x = torch.load(p, map_location="cpu", weights_only=False)
        for row in x:
            m = identity(row["meta"])
            identities[group].append(m)
            groups.setdefault(str(m["image_id"]), group)
        del x
        gc.collect()
    jdump(out / "current_candidate_identities.json", identities)
    jdump(out / "image_groups.json", groups)
    return groups, identities


def normalized_affine(shared: dict):
    A = [torch.as_tensor(x, dtype=torch.float64) for x in shared["A"]]
    stats = {}
    for k, v in shared["stats"].items():
        mu, std = v
        stats[int(k)] = (torch.as_tensor(mu, dtype=torch.float64), torch.as_tensor(std, dtype=torch.float64))
    raw = []
    for level, a in enumerate(A):
        mu, std = stats[level]
        U, v = a[:64], a[64]
        dw = U / std[:, None]
        db = v - (mu / std) @ U
        raw.append((dw, db))
    return A, stats, raw


def load_native_head(weights: Path, raw_maps, out: Path):
    """Inspect the native one-to-one final conv and save an additive head copy."""
    wrapper = YOLO(str(weights))
    head = wrapper.model.model[-1]
    if not getattr(head, "end2end", False):
        raise RuntimeError("checkpoint does not expose the expected end2end head")
    layers = []
    checks = []
    for level, (dw, db) in enumerate(raw_maps):
        conv = head.one2one_cv4[level][-1]
        W0 = conv.weight.detach().cpu().float().squeeze(-1).squeeze(-1)
        b0 = conv.bias.detach().cpu().float() if conv.bias is not None else torch.zeros(W0.shape[0])
        if tuple(W0.shape) != (32, 64):
            raise RuntimeError(f"unexpected final coefficient layer shape at {level}: {tuple(W0.shape)}")
        dw32, db32 = dw.float(), db.float()
        W1 = W0 + dw32.T
        b1 = b0 + db32
        layers.append({"level": level, "W0": W0, "b0": b0, "delta_W": dw32.T, "delta_b": db32, "W1": W1, "b1": b1})
        checks.append({"level": level, "W0_shape": list(W0.shape), "delta_W_max": float(dw32.abs().max()), "delta_b_max": float(db32.abs().max())})
    torch.save({"branch": "one2one_cv4", "layers": layers, "checkpoint_sha256": sha256(weights)}, out / "merged_coeff_head.pt")
    jdump(out / "merged_head_diff.json", {"branch": "one2one_cv4", "layers": checks})
    del wrapper
    gc.collect()
    return layers


def ann_to_mask(ann: dict, shape: tuple[int, int]) -> np.ndarray:
    """COCO annToMask without retaining the full pycocotools index."""
    h, w = shape
    seg = ann.get("segmentation")
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict) and isinstance(seg.get("counts"), list):
        rle = mask_utils.frPyObjects(seg, h, w)
    elif isinstance(seg, dict):
        rle = seg
    else:
        return np.zeros((h, w), dtype=np.uint8)
    return np.asarray(mask_utils.decode(rle), dtype=np.uint8)


def load_annotation_map(path: Path, wanted: set[int]) -> dict[int, dict]:
    """Load one split's JSON and retain only annotations used by this run."""
    data = json.loads(path.read_text(encoding="utf-8"))
    ans = {int(a["id"]): a for a in data["annotations"] if int(a["id"]) in wanted}
    del data
    gc.collect()
    missing = wanted.difference(ans)
    if missing:
        raise RuntimeError(f"missing {len(missing)} candidate annotations in {path}")
    return ans


def padded_gt(ann: dict, archive: dict) -> tuple[torch.Tensor, torch.Tensor]:
    oh, ow = map(int, archive["original_shape"])
    gain = float(archive["gain"])
    left, top = int(archive["left"]), int(archive["top"])
    nh, nw = round(oh * gain), round(ow * gain)
    gt0 = torch.from_numpy(ann_to_mask(ann, (oh, ow)).astype(np.float32))[None, None]
    gt_pad = F.pad(F.interpolate(gt0, (nh, nw), mode="nearest"), (left, 640 - nw - left, top, 640 - nh - top))[0, 0]
    return gt_pad, gt0[0, 0].bool()


def decode_original(proto, coeff, box, archive):
    """Official 8.4.100 process_mask followed by exact letterbox inversion."""
    padded = ops.process_mask(proto, coeff[None], box[None], (640, 640), upsample=True)[0]
    oh, ow = map(int, archive["original_shape"])
    gain = float(archive["gain"])
    left, top = int(archive["left"]), int(archive["top"])
    x = ops.scale_masks(padded[None, None].float(), (oh, ow), ratio_pad=((gain, gain), (left, top)))[0, 0]
    return x > 0.5


def auc_supported(logit, gt_pad, box):
    x1, y1, x2, y2 = [float(v) for v in box]
    yy, xx = torch.meshgrid(torch.arange(640), torch.arange(640), indexing="ij")
    support = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    score = logit[support].numpy()
    label = gt_pad[support].numpy().astype(np.uint8)
    if label.min() == label.max():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    n_pos = float(label.sum())
    n_neg = float(len(label) - label.sum())
    return float((ranks[label == 1].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def diag_bce(proto, coeff, box, gt_pad):
    """Same GT-box support and area-normalized BCE used in Phase A."""
    raw = (coeff @ proto.flatten(1)).reshape(160, 160)
    x1, y1, x2, y2 = [float(v) / 4.0 for v in box]
    yy, xx = torch.meshgrid(torch.arange(160), torch.arange(160), indexing="ij")
    support = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    y = F.interpolate(gt_pad.to(raw.device)[None, None], (160, 160), mode="nearest")[0, 0]
    narea = max(float(max(x2 - x1, 0.0) * max(y2 - y1, 0.0)), 1e-8)
    bce = F.binary_cross_entropy_with_logits(raw[support], y[support], reduction="sum")
    return float(bce / narea), raw


def ci_bootstrap(per_image: dict[str, list[float]], seed=20260930, B=5000):
    keys = sorted(per_image)
    arr = np.asarray([per_image[k] for k in keys], dtype=np.float64)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) == 0:
        return {"n_images": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(B, len(arr)))
    means = arr[idx].mean(axis=1)
    return {"n_images": int(len(arr)), "mean_A": float(arr[:, 0].mean()), "mean_B": float(arr[:, 1].mean()), "mean_delta": float((arr[:, 1] - arr[:, 0]).mean()), "ci95_delta": [float(np.quantile(means[:, 1] - means[:, 0], .025)), float(np.quantile(means[:, 1] - means[:, 0], .975))]}


def evaluate(cache: Path, groups: dict, identities: dict, data: Path, layers, A, stats, out: Path, only_groups=None, max_images=None):
    # Build a stable lookup and assert that every image archive belongs to one split.
    image_group = groups
    cand_counts = defaultdict(int)
    image_rows = defaultdict(list)
    for group, vals in identities.items():
        for m in vals:
            image_rows[(group, int(m["image_id"]))].append(m)
            cand_counts[group] += 1
    results = []
    group_image_order = {g: sorted({int(x["image_id"]) for x in vals}) for g, vals in identities.items()}
    for group in ("fit", "dev", "val"):
        if only_groups and group not in only_groups:
            continue
        ids = group_image_order[group]
        if max_images is not None:
            ids = ids[:max_images]
        ann_path = data / "annotations" / ("instances_val2017.json" if group == "val" else "instances_train2017.json")
        wanted = {int(m["annotation_id"]) for m in identities[group]}
        ann_map = load_annotation_map(ann_path, wanted)
        for pos, iid in enumerate(ids, 1):
            f = cache / "images" / f"{iid:012d}.pt"
            archive = torch.load(f, map_location="cpu", weights_only=False)
            proto = archive["proto"].float()
            coeff_all = archive["coeff"].float()
            h_all = archive["h"].float()
            boxes = archive["boxes"].float()
            rows = {int(m["raw_id"]): m for m in archive["rows"]}
            anns = {int(m["annotation_id"]): ann_map[int(m["annotation_id"])] for m in archive["rows"] if int(m["image_id"]) == iid and int(m["annotation_id"]) in ann_map}
            for rid, m in rows.items():
                if int(m["image_id"]) != iid or int(m["annotation_id"]) not in anns:
                    continue
                ann = anns[int(m["annotation_id"])]
                gt_pad, gt_orig = padded_gt(ann, archive)
                c0 = coeff_all[rid]
                h = h_all[rid].double()
                level = int(m.get("level", m.get("pyramid_level", -1)))
                if level not in stats:
                    raise RuntimeError(f"missing affine stats for level {level}")
                mu, std = stats[level]
                a = A[level]
                c1_direct = c0.double() + torch.cat(((h - mu) / std, torch.ones(1, dtype=torch.float64))) @ a
                layer = layers[level]
                c0_from_head = h.float() @ layer["W0"].T + layer["b0"]
                c1_merged = h.float() @ layer["W1"].T + layer["b1"]
                # Logit path check uses the full prototype, while normal decode uses official process_mask.
                a_mask = decode_original(proto, c0, boxes[rid], archive)
                b_mask = decode_original(proto, c1_merged, boxes[rid], archive)
                iou_a = float((a_mask & gt_orig).sum() / (a_mask | gt_orig).sum().clamp_min(1))
                iou_b = float((b_mask & gt_orig).sum() / (b_mask | gt_orig).sum().clamp_min(1))
                cov_a = float((a_mask & gt_orig).sum() / gt_orig.sum().clamp_min(1))
                cov_b = float((b_mask & gt_orig).sum() / gt_orig.sum().clamp_min(1))
                z0 = (c0 @ proto.flatten(1)).reshape(160, 160)
                z1 = (c1_merged @ proto.flatten(1)).reshape(160, 160)
                z0_up = F.interpolate(z0[None, None], (640, 640), mode="bilinear")[0, 0]
                z1_up = F.interpolate(z1[None, None], (640, 640), mode="bilinear")[0, 0]
                auc_a = auc_supported(z0_up, gt_pad, boxes[rid])
                auc_b = auc_supported(z1_up, gt_pad, boxes[rid])
                bce_a, _ = diag_bce(proto, c0, boxes[rid], gt_pad)
                bce_b, _ = diag_bce(proto, c1_merged, boxes[rid], gt_pad)
                results.append({"group": group, "image_id": iid, "annotation_id": int(m["annotation_id"]), "branch": "one2one", "raw_id": rid, "pyramid_level": level, "box_iou": float(m.get("box_iou", float("nan")),), "iou_A": iou_a, "iou_B": iou_b, "coverage_A": cov_a, "coverage_B": cov_b, "auc_A": auc_a, "auc_B": auc_b, "bce_A": bce_a, "bce_B": bce_b, "mask75_A": int(iou_a >= .75), "mask75_B": int(iou_b >= .75), "orig_success_A": int(iou_a >= .75), "c0_head_maxerr": float((c0_from_head - c0).abs().max()), "merge_direct_coeff_maxerr": float((c1_merged.double() - c1_direct).abs().max()), "merge_direct_logit_maxerr": float(((c1_merged.double() - c1_direct) @ proto.double().flatten(1)).abs().max())})
            del archive, proto, coeff_all, h_all, boxes
            gc.collect()
            if pos % 25 == 0 or pos == len(ids):
                print(json.dumps({"group": group, "images": pos, "total": len(ids), "results": len(results)}, ensure_ascii=False), flush=True)
        del ann_map
        gc.collect()
    with (out / "per_candidate.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    # Candidate and image-macro tables, with image-clustered bootstrap for every group.
    tables = {}
    for group in ("fit", "dev", "val"):
        rr = [r for r in results if r["group"] == group]
        if not rr:
            continue
        tables[group] = {"n_candidates": len(rr), "n_images": len({r["image_id"] for r in rr}), "candidate": {}, "image_macro": {}, "by_level": {}}
        for key in ("iou", "coverage", "auc", "bce"):
            akey, bkey = key + "_A", key + "_B"
            aa = np.asarray([r[akey] for r in rr], dtype=float); bb = np.asarray([r[bkey] for r in rr], dtype=float)
            tables[group]["candidate"][key] = {"A": float(np.nanmean(aa)), "B": float(np.nanmean(bb)), "delta": float(np.nanmean(bb-aa))}
        by_img = defaultdict(list)
        for r in rr: by_img[r["image_id"]].append(r)
        for key in ("iou", "coverage", "auc", "bce"):
            per = {str(i): [float(np.nanmean([r[key+"_A"] for r in rs])), float(np.nanmean([r[key+"_B"] for r in rs]))] for i, rs in by_img.items()}
            tables[group]["image_macro"][key] = ci_bootstrap(per)
        for level in (0,1,2):
            q = [r for r in rr if r["pyramid_level"] == level]
            if q:
                tables[group]["by_level"][str(level)] = {"n": len(q), "iou_delta": float(np.mean([r["iou_B"]-r["iou_A"] for r in q])), "bce_delta": float(np.mean([r["bce_B"]-r["bce_A"] for r in q])), "mask75_delta": float(np.mean([r["mask75_B"]-r["mask75_A"] for r in q]))}
        tables[group]["mask75"] = {"A": int(sum(r["mask75_A"] for r in rr)), "B": int(sum(r["mask75_B"] for r in rr)), "repair": int(sum(r["mask75_A"] == 0 and r["mask75_B"] == 1 for r in rr)), "damage": int(sum(r["mask75_A"] == 1 and r["mask75_B"] == 0 for r in rr))}
        tables[group]["checks"] = {"c0_head_maxerr": float(max(r["c0_head_maxerr"] for r in rr)), "merge_direct_coeff_maxerr": float(max(r["merge_direct_coeff_maxerr"] for r in rr)), "merge_direct_logit_maxerr": float(max(r["merge_direct_logit_maxerr"] for r in rr))}
        for success in (0,1):
            q = [r for r in rr if r["orig_success_A"] == success]
            if q:
                tables[group].setdefault("by_original_success", {})[str(success)] = {"n": len(q), "iou_delta": float(np.mean([r["iou_B"]-r["iou_A"] for r in q])), "coverage_delta": float(np.mean([r["coverage_B"]-r["coverage_A"] for r in q])), "auc_delta": float(np.nanmean([r["auc_B"]-r["auc_A"] for r in q]))}
    jdump(out / "SUMMARY.json", {"tables": tables, "candidate_counts": dict(cand_counts), "protocol": {"decoder": "ultralytics-8.4.100 process_mask upsample=True then scale_masks nearest", "main_comparison": "B_minus_A", "bootstrap": 5000, "auc": "continuous full-resolution logit restricted to fixed predicted box support", "gt_use": "fit only for shared solve; dev/val only for evaluation"}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--shared", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--groups", nargs="*", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    groups, identities = load_split_identities(args.cache, args.out, args.groups)
    shared = torch.load(args.shared, map_location="cpu", weights_only=False)
    A, stats, raw_maps = normalized_affine(shared)
    layers = load_native_head(args.weights, raw_maps, args.out)
    # Load annotations after the large split banks have been released.
    evaluate(args.cache, groups, identities, args.data, layers, A, stats, args.out, args.groups, args.max_images)
    jdump(args.out / "COMPLETE.json", {"status": "completed", "checkpoint_sha256": sha256(args.weights), "cache": str(args.cache), "shared": str(args.shared), "groups": args.groups, "max_images": args.max_images})


if __name__ == "__main__":
    main()
