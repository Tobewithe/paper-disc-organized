"""Verify the oracle objective behind argmax_maskfail_position_oracle_20261005 (B1 numerics).

Environment note: executed on the LOCAL DESKTOP (CPU) because the designated laptop
environment (ssh 28358lan) was unreachable (TCP 22 timeout, 2026-10-10). This is a
records-verification computation over a retained local cache with local COCO annotations;
it is not a model forward and involves no training.

Reconstructs the documented project oracle (7D convention: GT-box BCE + lambda*||d||^2/2,
LBFGS) for one retained candidate (image 139, annotation 26547, raw 1710) and compares
with the stored record:
  oracle_mask_iou 0.49079754601226994, objective 0.3169753617083252, delta_norm 3.940577983856201.

Letterbox reconstruction from the retained NPZ: input_shape (448,640), original_shape
(426,640) -> gain 1.0, pad top 11, left 0 (confirmed by box_original vs box_input delta).
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
STORED = dict(oracle_mask_iou=0.49079754601226994, objective=0.3169753617083252,
              delta_norm=3.940577983856201)
LAMBDAS = (0.003, 0.03, 0.3, 0.001)


def gt_mask(ann_id):
    import pycocotools.mask as mask_utils
    with ANN.open() as f:
        data = json.load(f)
    ann = next(a for a in data["annotations"] if a["id"] == ann_id)
    image = next(i for i in data["images"] if i["id"] == ann["image_id"])
    gt_h, gt_w = image["height"], image["width"]
    rle = mask_utils.frPyObjects(ann["segmentation"], gt_h, gt_w)
    return mask_utils.decode(rle).astype(np.uint8).max(axis=2), ann


def letterbox(gt, original_shape, input_shape):
    oh, ow = original_shape
    ih, iw = input_shape
    gain = min(ih / oh, iw / ow)
    nh, nw = round(oh * gain), round(ow * gain)
    top = round((ih - nh) / 2 - 0.1)
    left = round((iw - nw) / 2 - 0.1)
    gt_t = torch.from_numpy(gt).float()[None, None]
    scaled = F.interpolate(gt_t, (nh, nw), mode="nearest")
    padded = F.pad(scaled, (left, iw - nw - left, top, ih - nh - top))
    return padded[0, 0], (gain, left, top, nh, nw)


def solve(c0, p, y, area, penalty, iterations=120):
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
    return delta.detach().float(), float(value), float(grad.norm())


def decode_iou(proto, c, box_input, gt_grid, scale=0.25):
    z = (c @ proto.flatten(1)).reshape(proto.shape[1], proto.shape[2])
    x1, y1, x2, y2 = (box_input * scale).tolist()
    rr = torch.arange(proto.shape[2])[None, :]
    cc = torch.arange(proto.shape[1])[:, None]
    support = (rr >= x1) & (rr < x2) & (cc >= y1) & (cc < y2)
    pred = z > 0
    truth = gt_grid > 0.5
    inter = int((pred & truth & support).sum())
    union = int(((pred | truth) & support).sum())
    return inter / max(union, 1)


def main():
    data = np.load(NPZ, allow_pickle=True)
    row = int(np.nonzero(np.asarray(data["annotation_id"]) == TARGET_ANN)[0][0])
    proto = torch.from_numpy(np.asarray(data["proto"])).float()
    c0 = torch.from_numpy(np.asarray(data["coefficient"])[row]).float()
    box_input = torch.from_numpy(np.asarray(data["box_input"])[row]).float()
    gt, ann = gt_mask(TARGET_ANN)
    input_shape = [int(x) for x in np.asarray(data["input_shape"])]
    original_shape = [int(x) for x in np.asarray(data["original_shape"])]
    gt_pad, geom = letterbox(gt, original_shape, input_shape)
    gain, left, top, nh, nw = geom
    gh, gw = proto.shape[1], proto.shape[2]
    gt_grid = F.interpolate(gt_pad[None, None], (gh, gw), mode="nearest")[0, 0]
    # GT box on the decode grid
    bx, by, bw, bh = ann["bbox"]
    gx1 = bx * gain + left
    gy1 = by * gain + top
    gx2 = (bx + bw) * gain + left
    gy2 = (by + bh) * gain + top
    scale = gw / input_shape[1]
    nx1, ny1, nx2, ny2 = gx1 * scale, gy1 * scale, gx2 * scale, gy2 * scale
    rr = torch.arange(gw)[None, :]
    cc = torch.arange(gh)[:, None]
    support = (rr >= nx1) & (rr < nx2) & (cc >= ny1) & (cc < ny2)
    p = proto[:, support].T.contiguous()
    y = gt_grid[support]
    area = float(len(y))
    report = {"geometry": {"gain": gain, "left": left, "top": top, "grid": [gh, gw],
                           "gt_box_grid": [nx1, ny1, nx2, ny2], "roi_pixels": int(support.sum())},
              "stored": STORED,
              "attempts": {}}
    for penalty in LAMBDAS:
        delta, value, stat = solve(c0, p, y, area, penalty)
        iou = decode_iou(proto, c0 + delta, box_input, gt_grid)
        report["attempts"][str(penalty)] = dict(oracle_mask_iou=iou, objective=value,
                                                delta_norm=float(delta.norm()),
                                                stationary_norm=stat,
                                                abs_diff_iou=abs(iou - STORED["oracle_mask_iou"]))
    report["baseline_iou_check"] = decode_iou(proto, c0, box_input, gt_grid)
    report["stored_baseline"] = float(np.asarray(data["baseline_mask_iou"])[row])
    report["stored_support_upper"] = float(np.asarray(data["support_upper"])[row])
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
