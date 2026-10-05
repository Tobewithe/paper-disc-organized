"""Align frozen 7J-N ROI, GT ownership, h, and original mask logits."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def crop_pool(feature, box):
    channels, height, width = feature.shape
    x1 = max(0, min(width - 1, math.floor(float(box[0]) * width / 640)))
    y1 = max(0, min(height - 1, math.floor(float(box[1]) * height / 640)))
    x2 = max(x1 + 1, min(width, math.ceil(float(box[2]) * width / 640)))
    y2 = max(y1 + 1, min(height, math.ceil(float(box[3]) * height / 640)))
    return F.adaptive_avg_pool2d(feature[:, y1:y2, x1:x2], (8, 8))


def build(split, bank, metadata_path, neck_path, out):
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)["rows"]
    neck = torch.load(neck_path, map_location="cpu", weights_only=True)
    keys = [(int(r["image_id"]), int(r["annotation_id"]), int(r["raw_id"]))
            for r in metadata]
    assert keys == [tuple(map(int, key)) for key in neck["keys"]]
    assert len(set(keys)) == len(keys)
    grouped = defaultdict(list)
    for i, key in enumerate(keys):
        grouped[key[0]].append(i)
    h = torch.empty((len(keys), 64), dtype=torch.float32)
    base = torch.empty((len(keys), 1, 8, 8), dtype=torch.float32)
    label = torch.empty_like(base)
    box_iou = torch.empty(len(keys), dtype=torch.float32)
    full_iou = torch.full((len(keys),), float("nan"))
    for step, (iid, indices) in enumerate(sorted(grouped.items()), 1):
        cache = torch.load(bank / "images" / f"{iid:012d}.pt",
                           map_location="cpu", weights_only=True)
        cache_rows = {(int(r["annotation_id"]), int(r["raw_id"])): j
                      for j, r in enumerate(cache["rows"])}
        assert len(cache_rows) == len(cache["rows"])
        for i in indices:
            _, aid, raw = keys[i]
            j = cache_rows[(aid, raw)]
            box = cache["boxes"][raw]
            assert cache["coeff"].shape[0] == cache["h"].shape[0] == 8400
            z = cache["coeff"][raw].float() @ cache["proto"].float().reshape(32, -1)
            z = z.reshape(1, 160, 160)
            y = (cache["masks"] == int(cache["owners"][j]) + 1).float()[None]
            base[i] = crop_pool(z, box).clamp(-30, 30)
            label[i] = crop_pool(y, box).clamp(0, 1)
            h[i] = cache["h"][raw].float()
            box_iou[i] = float(cache["rows"][j]["box_iou"])
            if cache["rows"][j].get("initial_iou") is not None:
                full_iou[i] = float(cache["rows"][j]["initial_iou"])
        if step % 100 == 0:
            print(json.dumps({"split": split, "images": step,
                              "total_images": len(grouped)}), flush=True)
    assert torch.isfinite(base).all() and torch.isfinite(label).all()
    assert bool(((label >= 0) & (label <= 1)).all())
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(keys=keys, h=h, base=base, label=label,
                    box_iou=box_iou, full_iou=full_iou), out / f"{split.upper()}.pt")
    pred = base.sigmoid() >= .5
    truth = label >= .5
    overlap = (pred & truth).flatten(1).sum(1)
    union = (pred | truth).flatten(1).sum(1).clamp_min(1)
    grid_iou = overlap.float() / union.float()
    summary = {"split": split, "images": len(grouped), "instances": len(keys),
               "roi_shape": list(neck["roi"].shape), "owner_labels_soft": True,
               "mean_positive_area": float(label.mean()),
               "empty_label_instances": int((label.flatten(1).sum(1) == 0).sum()),
               "original_grid_iou_mean": float(grid_iou.mean()),
               "full_iou_available": int(torch.isfinite(full_iou).sum())}
    if split == "val":
        assert torch.isfinite(full_iou).all()
        summary["original_full_iou_mean"] = float(full_iou.mean())
        summary["grid_full_iou_pearson"] = float(torch.corrcoef(
            torch.stack((grid_iou, full_iou)))[0, 1])
    (out / f"{split.upper()}_AUDIT.json").write_text(json.dumps(summary, indent=2),
                                                       encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--fit-metadata", type=Path, required=True)
    parser.add_argument("--dev-metadata", type=Path, required=True)
    parser.add_argument("--val-metadata", type=Path, required=True)
    parser.add_argument("--fit-neck", type=Path, required=True)
    parser.add_argument("--dev-neck", type=Path, required=True)
    parser.add_argument("--val-neck", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    torch.set_num_threads(6)
    for split in ("fit", "dev", "val"):
        build(split, a.bank, getattr(a, f"{split}_metadata"),
              getattr(a, f"{split}_neck"), a.out)
