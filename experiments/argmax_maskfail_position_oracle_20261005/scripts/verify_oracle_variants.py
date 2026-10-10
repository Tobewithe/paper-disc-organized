"""Pin down the oracle definition of argmax_maskfail_position_oracle_20261005 by variant search.

Same candidate as verify_oracle_definition.py (image 139, annotation 26547, raw 1710).
Stored: oracle_mask_iou 0.49079754601226994, objective 0.3169753617083252,
delta_norm 3.940577983856201, baseline_mask_iou 0.3490153172866521.

Variants: ROI in {gt_box, pred_box} x GT rasterization in {two-step nearest, direct nearest}
with lambda=0.003 GT-ROI BCE + penalty, LBFGS 120. Reports each variant's baseline IoU,
oracle IoU, objective, delta norm, and the match against stored values.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

NPZ = Path(r"C:\Users\Administrator\AppData\Local\Temp\oracle_verify\images\000000000139.npz")
ANN = Path(r"C:\Dpan\document\model_datasets\datasets\coco\annotations\instances_val2017.json")
TARGET_ANN = 26547
PENALTY = 0.003
STORED = dict(oracle=0.49079754601226994, objective=0.3169753617083252,
              delta_norm=3.940577983856201, baseline=0.3490153172866521)


def load():
    data = np.load(NPZ, allow_pickle=True)
    row = int(np.nonzero(np.asarray(data["annotation_id"]) == TARGET_ANN)[0][0])
    with ANN.open() as f:
        ann_data = json.load(f)
    ann = next(a for a in ann_data["annotations"] if a["id"] == TARGET_ANN)
    import pycocotools.mask as mask_utils
    image = next(i for i in ann_data["images"] if i["id"] == ann["image_id"])
    rle = mask_utils.frPyObjects(ann["segmentation"], image["height"], image["width"])
    gt = mask_utils.decode(rle).astype(np.uint8).max(axis=2)
    return data, row, gt, ann, image


def solve(c0, p, y, area, penalty=PENALTY, iterations=120):
    p, y, c0 = p.double(), y.double(), c0.double()

    def objective(delta):
        logits = p @ (c0 + delta)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="sum") / area
        return bce + penalty * delta.square().sum() / 2

    delta = torch.zeros_like(c0, requires_grad=True)
    opt = torch.optim.LBFGS([delta], lr=1, max_iter=iterations,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        loss = objective(delta)
        loss.backward()
        return loss

    opt.step(closure)
    value = objective(delta)
    grad = torch.autograd.grad(value, delta)[0]
    return delta.detach().float(), float(value.detach()), float(grad.norm())


def iou_at(proto, c, box_grid, gt_grid):
    z = (c @ proto.flatten(1)).reshape(proto.shape[1], proto.shape[2])
    x1, y1, x2, y2 = box_grid
    rr = torch.arange(proto.shape[2])[None, :]
    cc = torch.arange(proto.shape[1])[:, None]
    support = (rr >= x1) & (rr < x2) & (cc >= y1) & (cc < y2)
    pred, truth = z > 0, gt_grid > 0.5
    inter = int((pred & truth & support).sum())
    union = int(((pred | truth) & support).sum())
    return inter / max(union, 1)


def main():
    data, row, gt, ann, image = load()
    proto = torch.from_numpy(np.asarray(data["proto"])).float()
    c0 = torch.from_numpy(np.asarray(data["coefficient"])[row]).float()
    box_input = np.asarray(data["box_input"])[row]  # input-pixel space (448x640)
    input_shape = [int(x) for x in np.asarray(data["input_shape"])]
    gt_h, gt_w = image["height"], image["width"]
    gain = min(input_shape[0] / gt_h, input_shape[1] / gt_w)
    nh, nw = round(gt_h * gain), round(gt_w * gain)
    top = round((input_shape[0] - nh) / 2 - 0.1)
    left = round((input_shape[1] - nw) / 2 - 0.1)
    scale = proto.shape[2] / input_shape[1]  # 160/640 = 0.25

    raster = {}
    gt_t = torch.from_numpy(gt).float()[None, None]
    two = F.pad(F.interpolate(gt_t, (nh, nw), mode="nearest"),
                (left, input_shape[1] - nw - left, top, input_shape[0] - nh - top))
    raster["two_step"] = F.interpolate(two, (proto.shape[1], proto.shape[2]), mode="nearest")[0, 0]
    raster["direct"] = F.interpolate(gt_t, (proto.shape[1], proto.shape[2]), mode="nearest")[0, 0]

    pred_grid = [box_input[0] * scale, box_input[1] * scale, box_input[2] * scale, box_input[3] * scale]
    bx, by, bw, bh = ann["bbox"]
    gt_grid_box = [(bx * gain + left) * scale, (by * gain + top) * scale,
                   ((bx + bw) * gain + left) * scale, ((by + bh) * gain + top) * scale]

    out = {"pred_grid": pred_grid, "gt_grid": gt_grid_box, "variants": {}}
    for rname, g in raster.items():
        for roi_name, box in (("gt_box", gt_grid_box), ("pred_box", pred_grid)):
            x1, y1, x2, y2 = box
            rr = torch.arange(proto.shape[2])[None, :]
            cc = torch.arange(proto.shape[1])[:, None]
            support = (rr >= x1) & (rr < x2) & (cc >= y1) & (cc < y2)
            p = proto[:, support].T.contiguous()
            y = g[support]
            area = float(len(y))
            delta, value, stat = solve(c0, p, y, area)
            iou = iou_at(proto, c0 + delta, pred_grid, g)
            base = iou_at(proto, c0, pred_grid, g)
            out["variants"][f"{rname}|{roi_name}"] = dict(
                baseline=float(base), oracle=float(iou), objective=float(value),
                delta_norm=float(delta.norm()), stationary=float(stat), roi_pixels=int(support.sum()))
    out["pred_grid"] = [float(v) for v in pred_grid]
    out["gt_grid"] = [float(v) for v in gt_grid_box]
    print(json.dumps(out, indent=2), flush=True)
    print("STORED " + json.dumps(STORED), flush=True)


if __name__ == "__main__":
    main()
