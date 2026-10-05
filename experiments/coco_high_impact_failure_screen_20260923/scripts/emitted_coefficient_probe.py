"""Fit coefficients on the actual normal Box75-matched output candidate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO


def metric(mask, gt):
    tp = (mask & gt).sum().item()
    fp = (mask & ~gt).sum().item()
    fn = (~mask & gt).sum().item()
    return {"iou": tp / max(tp + fp + fn, 1), "coverage": tp / max(tp + fn, 1),
            "purity": tp / max(tp + fp, 1), "tp": tp, "fp": fp, "fn": fn}


def summarize(rows):
    result = {"n": len(rows), "baseline_mask75": sum(r["baseline"]["iou"] >= .75 for r in rows),
            "last_mask75": sum(r["last"]["iou"] >= .75 for r in rows),
            "best_mask75": sum(r["best_iou"] >= .75 for r in rows),
            "mean_baseline_iou": float(np.mean([r["baseline"]["iou"] for r in rows])),
            "mean_last_iou": float(np.mean([r["last"]["iou"] for r in rows])),
            "mean_coverage_gain": float(np.mean([r["last"]["coverage"] - r["baseline"]["coverage"] for r in rows])),
            "mean_purity_gain": float(np.mean([r["last"]["purity"] - r["baseline"]["purity"] for r in rows])),
            "mean_gt_class_score": float(np.mean([r["gt_class_score"] for r in rows]))}
    if "scalar_last" in rows[0]:
        result.update({"scalar_last_mask75": sum(r["scalar_last"]["iou"] >= .75 for r in rows),
                       "scalar_mean_last_iou": float(np.mean([r["scalar_last"]["iou"] for r in rows])),
                       "scalar_mean_coverage_gain": float(np.mean([r["scalar_last"]["coverage"] - r["baseline"]["coverage"] for r in rows])),
                       "scalar_mean_purity_gain": float(np.mean([r["scalar_last"]["purity"] - r["baseline"]["purity"] for r in rows]))})
    if "local4_last" in rows[0]:
        result.update({"local4_last_mask75": sum(r["local4_last"]["iou"] >= .75 for r in rows),
                       "local4_mean_last_iou": float(np.mean([r["local4_last"]["iou"] for r in rows])),
                       "local4_mean_coverage_gain": float(np.mean([r["local4_last"]["coverage"] - r["baseline"]["coverage"] for r in rows])),
                       "local4_mean_purity_gain": float(np.mean([r["local4_last"]["purity"] - r["baseline"]["purity"] for r in rows]))})
    if "roi32_last" in rows[0]:
        result.update({"roi32_last_mask75": sum(r["roi32_last"]["iou"] >= .75 for r in rows),
                       "roi32_mean_last_iou": float(np.mean([r["roi32_last"]["iou"] for r in rows])),
                       "roi32_mean_coverage_gain": float(np.mean([r["roi32_last"]["coverage"] - r["baseline"]["coverage"] for r in rows])),
                       "roi32_mean_purity_gain": float(np.mean([r["roi32_last"]["purity"] - r["baseline"]["purity"] for r in rows]))})
    return result


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    coco = COCO(str(args.annotations))
    category_to_index = {category: index for index, category in enumerate(sorted(coco.cats))}
    pools = defaultdict(list)
    with args.joined.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["area_group"] != "small" or not row["bbox_coco75"]:
                continue
            ann = coco.anns[row["annotation_id"]]
            fill = ann["area"] / (ann["bbox"][2] * ann["bbox"][3])
            if fill >= .5:
                continue
            if row["geometry_state"] == "box_good_mask_unavailable":
                pools["failure"].append(row)
            elif row["geometry_state"] == "joint_good":
                pools["success_control"].append(row)
    rng = random.Random(20260923)
    selected = []
    for name, n in (("failure", 100), ("success_control", 50)):
        source = sorted(pools[name], key=lambda row: row["annotation_id"])
        assert len(source) >= n
        selected.extend((name, row) for row in rng.sample(source, n))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for number, (group, row) in enumerate(selected, 1):
        raw_id = row["bbox_raw_id"]
        assert raw_id is not None
        with np.load(args.archive / "images" / f"{row['image_id']:012d}.npz") as archive:
            proto = torch.as_tensor(archive["proto"], device=device)
            coeff = torch.as_tensor(archive["coefficients"][:, raw_id], device=device)
            box = torch.as_tensor(archive["boxes_input"][raw_id], device=device)
            ih, iw = map(int, archive["input_shape"])
            oh, ow = map(int, archive["original_shape"])
            gt_index = np.flatnonzero(archive["gt_ids"] == row["annotation_id"])
            assert len(gt_index) == 1
            archive_iou = float(archive["mask_ious"][raw_id, gt_index[0]])
            gt_class_score = float(archive["scores"][raw_id, category_to_index[row["category_id"]]])
        gt = torch.as_tensor(coco.annToMask(coco.anns[row["annotation_id"]]), device=device).bool()
        gain = min(ih / oh, iw / ow)
        nh, nw = round(oh * gain), round(ow * gain)
        top, left = round((ih - nh) / 2 - .1), round((iw - nw) / 2 - .1)
        target_input = F.pad(F.interpolate(gt.float()[None, None], (nh, nw), mode="bilinear", align_corners=False)[0, 0],
                             (left, iw - nw - left, top, ih - nh - top))
        y0, y1 = max(0, int(torch.ceil(box[1]))), min(ih, int(torch.ceil(box[3])))
        x0, x1 = max(0, int(torch.ceil(box[0]))), min(iw, int(torch.ceil(box[2])))
        assert x1 > x0 and y1 > y0
        y, x = torch.meshgrid(torch.arange(y0, y1, device=device), torch.arange(x0, x1, device=device), indexing="ij")
        coordinates = torch.stack((2 * (x + .5) / iw - 1, 2 * (y + .5) / ih - 1), dim=-1)[None]
        local_coordinates = torch.stack((2 * (x - box[0]) / (box[2] - box[0]).clamp(min=1) - 1,
                                         2 * (y - box[1]) / (box[3] - box[1]).clamp(min=1) - 1), dim=-1)[None]
        target_roi = target_input[y0:y1, x0:x1]
        full_y, full_x = torch.meshgrid(torch.arange(ih, device=device), torch.arange(iw, device=device), indexing="ij")
        support = (full_x >= box[0]) & (full_x < box[2]) & (full_y >= box[1]) & (full_y < box[3])
        base = torch.einsum("c,chw->hw", coeff, proto).detach()

        def decode(low, bias=0):
            logits = F.interpolate(low[None, None], (ih, iw), mode="bilinear", align_corners=False)[0, 0] + bias
            binary = ((logits > 0) & support).float()
            pred = F.interpolate(binary[None, None, top:top + nh, left:left + nw], (oh, ow),
                                 mode="bilinear", align_corners=False)[0, 0].byte().bool()
            return metric(pred, gt)

        with torch.no_grad():
            baseline = decode(base)
            if abs(baseline["iou"] - archive_iou) >= 2e-6:
                with np.load(args.archive / "images" / f"{row['image_id']:012d}.npz") as archive:
                    all_coefficients = torch.as_tensor(archive["coefficients"], device=device)
                base = (all_coefficients.T @ proto.flatten(1)).reshape(-1, *proto.shape[-2:])[raw_id].detach()
                baseline = decode(base)
            assert abs(baseline["iou"] - archive_iou) < 2e-6, (row["annotation_id"], archive_iou, baseline)
        scale = proto.square().mean((1, 2)).sqrt().clamp(min=.1).detach()
        delta = torch.zeros(proto.shape[0], device=device, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=.08)
        best_iou = baseline["iou"]
        history = []
        for step in range(args.steps + 1):
            low = base + (delta[:, None, None] * proto / scale[:, None, None]).sum(0)
            pred = F.grid_sample(low[None, None], coordinates, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
            loss = F.binary_cross_entropy_with_logits(pred, target_roi)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss for {row['annotation_id']}")
            if step % 60 == 0 or step == args.steps:
                with torch.no_grad():
                    measured = decode(low)
                    best_iou = max(best_iou, measured["iou"])
                    history.append({"step": step, "iou": measured["iou"], "loss": float(loss)})
            if step < args.steps:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            last = decode(base + (delta[:, None, None] * proto / scale[:, None, None]).sum(0))
        scalar_last = None
        if args.scalar_control:
            bias = torch.zeros((), device=device, requires_grad=True)
            scalar_optimizer = torch.optim.Adam([bias], lr=.08)
            with torch.no_grad():
                base_roi = F.grid_sample(base[None, None], coordinates, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
            for step in range(args.steps):
                scalar_loss = F.binary_cross_entropy_with_logits(base_roi + bias, target_roi)
                scalar_optimizer.zero_grad()
                scalar_loss.backward()
                scalar_optimizer.step()
            with torch.no_grad():
                scalar_last = decode(base, bias)
        local4_last = None
        if args.local_control:
            local4 = torch.zeros((1, 1, 4, 4), device=device, requires_grad=True)
            local_optimizer = torch.optim.Adam([local4], lr=.08)
            with torch.no_grad():
                base_roi = F.grid_sample(base[None, None], coordinates, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
            for step in range(args.steps):
                correction = F.grid_sample(local4, local_coordinates, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]
                local_loss = F.binary_cross_entropy_with_logits(base_roi + correction, target_roi)
                local_optimizer.zero_grad()
                local_loss.backward()
                local_optimizer.step()
            with torch.no_grad():
                full_grid = torch.stack((2 * (full_x - box[0]) / (box[2] - box[0]).clamp(min=1) - 1,
                                         2 * (full_y - box[1]) / (box[3] - box[1]).clamp(min=1) - 1), dim=-1)[None]
                correction_full = F.grid_sample(local4, full_grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]
                local4_last = decode(base, correction_full)
        roi32_last = None
        if args.roi32_control:
            t = (torch.arange(32, device=device) + .5) / 32
            roi_y, roi_x = torch.meshgrid(t, t, indexing="ij")
            sample_x = box[0] + roi_x * (box[2] - box[0])
            sample_y = box[1] + roi_y * (box[3] - box[1])
            roi32_grid = torch.stack((2 * sample_x / iw - 1, 2 * sample_y / ih - 1), dim=-1)[None]
            with torch.no_grad():
                target32 = F.grid_sample(target_input[None, None], roi32_grid, align_corners=False)[0, 0]
            delta32 = torch.zeros(proto.shape[0], device=device, requires_grad=True)
            optimizer32 = torch.optim.Adam([delta32], lr=.08)
            for step in range(args.steps):
                low32 = base + (delta32[:, None, None] * proto / scale[:, None, None]).sum(0)
                pred32 = F.grid_sample(low32[None, None], roi32_grid, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
                loss32 = F.binary_cross_entropy_with_logits(pred32, target32)
                optimizer32.zero_grad()
                loss32.backward()
                optimizer32.step()
            with torch.no_grad():
                roi32_last = decode(base + (delta32[:, None, None] * proto / scale[:, None, None]).sum(0))
        results.append({"group": group, "image_id": row["image_id"], "annotation_id": row["annotation_id"],
                        "raw_id": raw_id, "gt_class_score": gt_class_score,
                        "baseline": baseline, "last": last, "best_iou": best_iou, "history": history,
                        **({"scalar_last": scalar_last} if scalar_last is not None else {}),
                        **({"local4_last": local4_last} if local4_last is not None else {}),
                        **({"roi32_last": roi32_last} if roi32_last is not None else {})})
        if number % 10 == 0:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "PROGRESS.json").write_text(json.dumps({"done": number, "total": len(selected)}), encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "OBJECTS.jsonl").write_text("\n".join(json.dumps(row) for row in results) + "\n", encoding="utf-8")
    result = {"source_counts": {key: len(value) for key, value in pools.items()}, "steps": args.steps,
              "groups": {key: summarize([row for row in results if row["group"] == key]) for key in pools}}
    (args.out / "SUMMARY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"objects": len(results)}), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("joined", "archive", "annotations", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--scalar-control", action="store_true")
    parser.add_argument("--local-control", action="store_true")
    parser.add_argument("--roi32-control", action="store_true")
    main(parser.parse_args())
