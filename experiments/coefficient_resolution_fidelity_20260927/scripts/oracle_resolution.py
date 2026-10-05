"""Evaluate the *same* finite coefficient oracle on 640 and pooled 8 grids."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def clip_box(box, shape):
    height, width = shape
    x1 = max(0, min(width - 1, math.floor(float(box[0]) * width / 640)))
    y1 = max(0, min(height - 1, math.floor(float(box[1]) * height / 640)))
    x2 = max(x1 + 1, min(width, math.ceil(float(box[2]) * width / 640)))
    y2 = max(y1 + 1, min(height, math.ceil(float(box[3]) * height / 640)))
    return x1, y1, x2, y2


def measures(logit, label):
    prob = torch.sigmoid(logit).flatten().numpy()
    truth = (label.flatten().numpy() >= .5)
    pred = prob >= .5
    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    return {"iou": float(intersection / union) if union else float("nan"),
            "auc": float(roc_auc_score(truth, prob)) if truth.any() and
            not truth.all() else float("nan")}


def grouped_summary(rows, group_name, select, seed=20260927, reps=2000):
    subset = [r for r in rows if select(r)]
    ids = np.asarray([r["image_id"] for r in subset])
    unique, inverse = np.unique(ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(unique), size=(reps, len(unique)))
    output = {"instances": len(subset), "images": len(unique)}
    for metric in ("iou", "auc"):
        full = np.asarray([r["gain640"][metric] for r in subset])
        grid = np.asarray([r["gain8"][metric] for r in subset])
        valid = np.isfinite(full) & np.isfinite(grid)
        output[metric] = {}
        for name, values in (("gain640", full), ("gain8", grid),
                             ("gain8_minus_gain640", grid - full)):
            sums = np.bincount(inverse[valid], weights=values[valid], minlength=len(unique))
            counts = np.bincount(inverse[valid], minlength=len(unique))
            boot = sums[draw].sum(1) / counts[draw].sum(1)
            output[metric][name] = {"mean": float(np.mean(values[valid])),
                                    "ci95": np.quantile(boot, [.025, .975]).tolist()}
        output[metric]["spearman_gains"] = float(spearmanr(full[valid], grid[valid]).statistic)
        useful = valid & (full > .01)
        output[metric]["full_gain_gt_0p01"] = int(useful.sum())
        output[metric]["grid_gain_gt_0p01_given_full"] = float(
            np.mean(grid[useful] > .01)) if useful.any() else None
        output[metric]["sign_agreement"] = float(
            np.mean(np.sign(full[valid]) == np.sign(grid[valid])))
    return output


def main(args):
    torch.set_num_threads(6)
    val = torch.load(args.val_data, map_location="cpu", weights_only=True)
    targets = torch.load(args.targets, map_location="cpu", weights_only=True)
    keys = {(int(iid), int(aid)): (idx, int(raw))
            for idx, (iid, aid, raw) in enumerate(val["keys"])}
    assert len(keys) == len(val["keys"])
    rows = []
    assert len(targets) == 200
    for aid, versions in targets.items():
        # The val bank is indexed by (image, annotation). Find the sole val key.
        matches = [(iid, idx, raw) for (iid, ann), (idx, raw) in keys.items()
                   if ann == int(aid)]
        assert len(matches) == 1, (aid, len(matches))
        iid, idx, raw = matches[0]
        cache = torch.load(args.bank / "images" / f"{iid:012d}.pt",
                           map_location="cpu", weights_only=True)
        lookup = {(int(r["annotation_id"]), int(r["raw_id"])): j
                  for j, r in enumerate(cache["rows"])}
        j = lookup[(int(aid), raw)]
        assert cache["coeff"].shape[0] == cache["h"].shape[0] == 8400
        assert abs(float(cache["rows"][j]["box_iou"]) -
                   float(val["box_iou"][idx])) < 1e-5
        p = cache["proto"].float()
        c = cache["coeff"][raw].float()
        d = versions[args.penalty].float()
        z0160 = torch.einsum("c,chw->hw", c, p)
        z1160 = torch.einsum("c,chw->hw", c + d, p)
        owner = int(cache["owners"][j]) + 1
        y = (cache["masks"] == owner).float()
        x1, y1, x2, y2 = clip_box(cache["boxes"][raw], z0160.shape)
        z08 = F.adaptive_avg_pool2d(z0160[y1:y2, x1:x2][None, None], (8, 8)).squeeze()
        z18 = F.adaptive_avg_pool2d(z1160[y1:y2, x1:x2][None, None], (8, 8)).squeeze()
        bx1, by1, bx2, by2 = clip_box(cache["boxes"][raw], y.shape)
        y640 = y[by1:by2, bx1:bx2]
        y8 = F.adaptive_avg_pool2d(y640[None, None], (8, 8)).squeeze()
        assert torch.max(torch.abs(z08.clamp(-30, 30) - val["base"][idx, 0])).item() < 2e-3
        assert torch.max(torch.abs(y8 - val["label"][idx, 0])).item() < 1e-6
        z0640 = F.interpolate(z0160[None, None], size=y.shape,
                              mode="bilinear", align_corners=False).squeeze()
        z1640 = F.interpolate(z1160[None, None], size=y.shape,
                              mode="bilinear", align_corners=False).squeeze()
        original640 = measures(z0640[by1:by2, bx1:bx2], y640)
        oracle640 = measures(z1640[by1:by2, bx1:bx2], y640)
        original8, oracle8 = measures(z08, y8), measures(z18, y8)
        rows.append({"image_id": iid, "annotation_id": int(aid), "raw_id": raw,
                     "box_iou": float(val["box_iou"][idx]),
                     "original_full_iou": float(val["full_iou"][idx]),
                     "original640": original640, "oracle640": oracle640,
                     "original8": original8, "oracle8": oracle8,
                     "gain640": {m: oracle640[m] - original640[m] for m in original640},
                     "gain8": {m: oracle8[m] - original8[m] for m in original8}})
    assert len(rows) == 200
    groups = {"all": lambda r: True,
              "original_failure": lambda r: r["original_full_iou"] < .75,
              "good_box_failure": lambda r: r["box_iou"] >= .75 and
              r["original_full_iou"] < .75,
              "original_success": lambda r: r["original_full_iou"] >= .75}
    summary = {name: grouped_summary(rows, name, select) for name, select in groups.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ROWS.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "COMPLETE.json").write_text(json.dumps({"instances": len(rows),
        "images": len(set(r["image_id"] for r in rows)), "penalty": args.penalty},
        indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--penalty", default="0.003")
    main(parser.parse_args())
