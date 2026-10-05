"""GT-assisted coefficient-versus-arbitrary-grid representation probe."""
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


def summary(rows, arms):
    result = {
        "n": len(rows),
        "mean_original_iou": float(np.mean([r["original_iou"] for r in rows])),
        "mean_grid_construct_iou": float(np.mean([r["grid_construct_iou"] for r in rows])),
    }
    for arm in arms:
        result[f"{arm}_last_mask75"] = sum(r[arm]["last_iou"] >= .75 for r in rows)
        result[f"{arm}_best_mask75"] = sum(r[arm]["best_iou"] >= .75 for r in rows)
        result[f"mean_{arm}_last_iou"] = float(np.mean([r[arm]["last_iou"] for r in rows]))
    return result


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with args.grid_objects.open(encoding="utf-8") as stream:
        grid_rows = [json.loads(line) for line in stream]
    pools = defaultdict(list)
    for row in grid_rows:
        if row["group"] == "small_low_fail":
            key = "fail_grid_below75" if row["grid_best_iou"] < .75 else "fail_grid_at75"
            pools[key].append(row)
        elif row["group"] == "small_low_good":
            pools["joint_good_control"].append(row)
    counts = {"fail_grid_below75": 30, "fail_grid_at75": 30, "joint_good_control": 20}
    rng = random.Random(20260923)
    chosen = []
    for key, n in counts.items():
        source = sorted(pools[key], key=lambda row: row["annotation_id"])
        assert len(source) >= n, (key, len(source))
        chosen.extend((key, row) for row in rng.sample(source, n))
    coco = COCO(str(args.annotations))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = []
    for number, (group, row) in enumerate(chosen, 1):
        with np.load(args.archive / "images" / f"{row['image_id']:012d}.npz") as archive:
            proto = torch.as_tensor(archive["proto"], device=device)
            coefficients = torch.as_tensor(archive["coefficients"][:, row["raw_id"]], device=device)
            box = torch.as_tensor(archive["boxes_input"][row["raw_id"]], device=device)
            ih, iw = map(int, archive["input_shape"])
            oh, ow = map(int, archive["original_shape"])
        gt = torch.as_tensor(coco.annToMask(coco.anns[row["annotation_id"]]), device=device).bool()
        gain = min(ih / oh, iw / ow)
        nh, nw = round(oh * gain), round(ow * gain)
        top, left = round((ih - nh) / 2 - .1), round((iw - nw) / 2 - .1)
        target_input = F.pad(
            F.interpolate(gt.float()[None, None], (nh, nw), mode="bilinear", align_corners=False)[0, 0],
            (left, iw - nw - left, top, ih - nh - top),
        )
        y0, y1 = max(0, int(torch.ceil(box[1]))), min(ih, int(torch.ceil(box[3])))
        x0, x1 = max(0, int(torch.ceil(box[0]))), min(iw, int(torch.ceil(box[2])))
        assert x1 > x0 and y1 > y0
        y, x = torch.meshgrid(torch.arange(y0, y1, device=device), torch.arange(x0, x1, device=device), indexing="ij")
        coordinates = torch.stack((2 * (x + .5) / iw - 1, 2 * (y + .5) / ih - 1), dim=-1)[None]
        target_roi = target_input[y0:y1, x0:x1]
        full_y, full_x = torch.meshgrid(torch.arange(ih, device=device), torch.arange(iw, device=device), indexing="ij")
        support = (full_x >= box[0]) & (full_x < box[2]) & (full_y >= box[1]) & (full_y < box[3])
        base = torch.einsum("c,chw->hw", coefficients, proto).detach()
        scale = proto.square().mean((1, 2)).sqrt().clamp(min=.1).detach()

        def measure(low):
            logits = F.interpolate(low[None, None], (ih, iw), mode="bilinear", align_corners=False)[0, 0]
            binary = ((logits > 0) & support).float()
            original_grid = F.interpolate(
                binary[None, None, top:top + nh, left:left + nw], (oh, ow),
                mode="bilinear", align_corners=False,
            )[0, 0].byte().bool()
            return iou(original_grid, gt)

        with torch.no_grad():
            original_iou = measure(base)
            if abs(original_iou - row["baseline_iou"]) >= 2e-6:
                with np.load(args.archive / "images" / f"{row['image_id']:012d}.npz") as archive:
                    all_coefficients = torch.as_tensor(archive["coefficients"], device=device)
                base = (all_coefficients.T @ proto.flatten(1)).reshape(-1, *proto.shape[-2:])[row["raw_id"]].detach()
                original_iou = measure(base)
            assert abs(original_iou - row["baseline_iou"]) < 2e-6
        arms = {}
        arm_names = ("bce", "bce_dice", "dice") if args.coefficient_objectives else ("coefficient", "grid")
        for arm in arm_names:
            if arm != "grid":
                parameter = torch.zeros(proto.shape[0], device=device, requires_grad=True)
                optimizer = torch.optim.Adam([parameter], lr=.08)
            else:
                initialization = F.interpolate(target_input[None, None], proto.shape[-2:], mode="area")[0, 0]
                parameter = (2 * initialization - 1).mul(4).detach().requires_grad_(True)
                optimizer = torch.optim.Adam([parameter], lr=.2)
            best_iou = 0.0
            best_step = 0
            history = []
            for step in range(args.steps + 1):
                low = base + (parameter[:, None, None] * proto / scale[:, None, None]).sum(0) if arm != "grid" else parameter
                pred = F.grid_sample(low[None, None], coordinates, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
                probability = pred.sigmoid()
                bce = F.binary_cross_entropy_with_logits(pred, target_roi)
                dice = 1 - (2 * (probability * target_roi).sum() + 1) / (probability.sum() + target_roi.sum() + 1)
                loss = dice if arm == "dice" else bce if arm == "bce" else bce + .5 * dice
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite {arm} loss: {row['annotation_id']}")
                if step % 20 == 0 or step == args.steps:
                    with torch.no_grad():
                        score = measure(low)
                        history.append({"step": step, "iou": score, "loss": float(loss)})
                        if score > best_iou:
                            best_iou, best_step = score, step
                if step < args.steps:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            arms[arm] = {"last_iou": history[-1]["iou"], "best_iou": best_iou,
                         "best_step": best_step, "history": history}
        result.append({"group": group, "image_id": row["image_id"], "annotation_id": row["annotation_id"],
                       "raw_id": row["raw_id"], "original_iou": original_iou,
                       "grid_construct_iou": row["grid_best_iou"], **arms})
        if number % 10 == 0:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "PROGRESS.json").write_text(json.dumps({"done": number, "total": len(chosen)}), encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "OBJECTS.jsonl").write_text("\n".join(json.dumps(row) for row in result) + "\n", encoding="utf-8")
    final = {"device": device, "steps": args.steps, "arms": list(arm_names),
             "source_counts": {key: len(value) for key, value in pools.items()},
             "groups": {key: summary([row for row in result if row["group"] == key], arm_names) for key in counts}}
    (args.out / "SUMMARY.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"objects": len(result)}), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("grid_objects", "archive", "annotations", "out"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--coefficient-objectives", action="store_true")
    main(parser.parse_args())
