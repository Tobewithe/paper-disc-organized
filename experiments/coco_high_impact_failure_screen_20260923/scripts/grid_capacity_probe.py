"""Construct GT-assisted native-grid masks under fixed raw predicted boxes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO


def iou(pred, target):
    intersection = (pred & target).sum().item()
    union = (pred | target).sum().item()
    return intersection / union if union else 1.0


def summarize(rows):
    n = len(rows)
    return {
        "n": n,
        "baseline_mean_iou": sum(r["baseline_iou"] for r in rows) / n,
        "mean_input_construct_iou": sum(r["input_construct_iou"] for r in rows) / n,
        "mean_grid_best_iou": sum(r["grid_best_iou"] for r in rows) / n,
        "input_construct_mask75": sum(r["input_construct_iou"] >= .75 for r in rows),
        "grid_best_mask75": sum(r["grid_best_iou"] >= .75 for r in rows),
        "grid_nearest_mask75": sum(r["grid_nearest_iou"] >= .75 for r in rows),
        "grid_area_mask75": sum(r["grid_area_iou"] >= .75 for r in rows),
        "grid_best_at_least_baseline": sum(r["grid_best_iou"] >= r["baseline_iou"] for r in rows),
        "mean_grid_h": sum(r["grid_h"] for r in rows) / n,
        "mean_grid_w": sum(r["grid_w"] for r in rows) / n,
    }


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    coco = COCO(str(args.annotations))
    groups = defaultdict(list)
    with args.joined.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            ann = coco.anns[row["annotation_id"]]
            fill = ann["area"] / (ann["bbox"][2] * ann["bbox"][3])
            small = row["area_group"] == "small"
            low = fill < .5
            failed = row["geometry_state"] == "box_good_mask_unavailable"
            good = row["geometry_state"] == "joint_good"
            if failed and small and low:
                groups["small_low_fail"].append(row)
            elif failed and small and fill >= .75:
                groups["small_high_fail"].append(row)
            elif failed and row["area_group"] == "medium" and low:
                groups["medium_low_fail"].append(row)
            elif good and small and low:
                groups["small_low_good"].append(row)
    rng = random.Random(20260923)
    sample = []
    for name in ("small_low_fail", "small_high_fail", "medium_low_fail", "small_low_good"):
        source = sorted(groups[name], key=lambda row: row["annotation_id"])
        assert args.all or len(source) >= args.per_group, (name, len(source))
        selected = source if args.all else rng.sample(source, args.per_group)
        sample += [(name, row) for row in selected]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for number, (group, row) in enumerate(sample, 1):
        representative = row["best_mask_with_box75"]
        assert representative is not None
        image_id, raw_id = row["image_id"], representative["raw_id"]
        with np.load(args.archive / "images" / f"{image_id:012d}.npz") as archive:
            proto = torch.as_tensor(archive["proto"], device=device)
            coefficient = torch.as_tensor(archive["coefficients"][:, raw_id], device=device)
            box = torch.as_tensor(archive["boxes_input"][raw_id], device=device)
            ih, iw = map(int, archive["input_shape"])
            oh, ow = map(int, archive["original_shape"])
        gt = torch.as_tensor(coco.annToMask(coco.anns[row["annotation_id"]]), device=device).bool()
        assert gt.shape == (oh, ow)
        gain = min(ih / oh, iw / ow)
        nh, nw = round(oh * gain), round(ow * gain)
        top = round((ih - nh) / 2 - .1)
        left = round((iw - nw) / 2 - .1)
        target_input = F.pad(
            F.interpolate(gt.float()[None, None], (nh, nw), mode="bilinear", align_corners=False)[0, 0],
            (left, iw - nw - left, top, ih - nh - top),
        )
        yy, xx = torch.meshgrid(torch.arange(ih, device=device), torch.arange(iw, device=device), indexing="ij")
        support = (xx >= box[0]) & (xx < box[2]) & (yy >= box[1]) & (yy < box[3])

        def decode(logits):
            binary = ((logits > 0) & support).float()
            return F.interpolate(
                binary[None, None, top:top + nh, left:left + nw], (oh, ow),
                mode="bilinear", align_corners=False,
            )[0, 0].byte().bool()

        with torch.no_grad():
            original_low = torch.einsum("c,chw->hw", coefficient, proto)
            original_logits = F.interpolate(original_low[None, None], (ih, iw), mode="bilinear", align_corners=False)[0, 0]
            original_iou = iou(decode(original_logits), gt)
            batch_gemm_fallback = False
            if abs(original_iou - representative["mask_iou"]) >= 2e-6:
                with np.load(args.archive / "images" / f"{image_id:012d}.npz") as archive:
                    all_coefficients = torch.as_tensor(archive["coefficients"], device=device)
                original_low = (all_coefficients.T @ proto.flatten(1)).reshape(-1, *proto.shape[-2:])[raw_id]
                original_logits = F.interpolate(original_low[None, None], (ih, iw), mode="bilinear", align_corners=False)[0, 0]
                original_iou = iou(decode(original_logits), gt)
                batch_gemm_fallback = True
            assert abs(original_iou - representative["mask_iou"]) < 2e-6, (
                row["annotation_id"], original_iou, representative["mask_iou"]
            )
            grid_shape = proto.shape[-2:]
            score = {"input_construct_iou": iou(decode(2 * target_input - 1), gt)}
            for mode in ("nearest", "area"):
                sampled = F.interpolate(target_input[None, None], grid_shape, mode=mode)
                logits = F.interpolate(2 * sampled - 1, (ih, iw), mode="bilinear", align_corners=False)[0, 0]
                score[f"grid_{mode}_iou"] = iou(decode(logits), gt)
        results.append({
            "group": group, "image_id": image_id, "annotation_id": row["annotation_id"], "raw_id": raw_id,
            "baseline_iou": original_iou, "batch_gemm_fallback": batch_gemm_fallback, **score,
            "grid_best_iou": max(score["grid_nearest_iou"], score["grid_area_iou"]),
            "grid_h": int(grid_shape[0]), "grid_w": int(grid_shape[1]),
            "fill": float(coco.anns[row["annotation_id"]]["area"] / (
                coco.anns[row["annotation_id"]]["bbox"][2] * coco.anns[row["annotation_id"]]["bbox"][3]
            )),
        })
        if number % 50 == 0:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "PROGRESS.json").write_text(json.dumps({"done": number, "total": len(sample)}), encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "OBJECTS.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")
    summary = {"device": device, "source_group_sizes": {k: len(v) for k, v in groups.items()},
               "batch_gemm_fallbacks": sum(r["batch_gemm_fallback"] for r in results),
               "groups": {name: summarize([r for r in results if r["group"] == name]) for name in groups}}
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"objects": len(results)}), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("archive", "annotations", "joined", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=100)
    parser.add_argument("--all", action="store_true")
    main(parser.parse_args())
