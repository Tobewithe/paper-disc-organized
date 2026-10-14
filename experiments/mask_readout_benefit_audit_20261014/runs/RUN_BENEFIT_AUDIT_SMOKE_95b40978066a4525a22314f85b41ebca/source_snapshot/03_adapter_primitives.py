"""GT-free spatial ablation and separately normalized training-loss primitives.

No model runner or training is started by importing this module. Torch is lazy.
"""
from __future__ import annotations

import hashlib


def spatial_permutation(height, width, image_id, output_rank, phase, seed=314159):
    """Independent of GT and tensor values; reproducible per output and train epoch.

    phase = 'train:0', 'train:1', 'train:2', or 'eval:0' under the frozen defaults.
    All P channels and z0 use ONE permutation; never shuffle channels separately.
    """
    if height <= 0 or width <= 0 or output_rank < 0 or phase not in {
        "train:0", "train:1", "train:2", "eval:0"
    }:
        raise ValueError("invalid frozen spatial-ablation key")
    prefix = f"spatial-shuffle-v1:{seed}:{phase}:{image_id}:{output_rank}:"
    return sorted(range(height * width),
                  key=lambda j: (hashlib.sha256(f"{prefix}{j}".encode()).digest(), j))


def shuffle_spatial(spatial, image_id, output_rank, phase, seed=314159):
    """[K+1,H,W] -> same shape/values, joint channel-vector permutation."""
    import torch
    if spatial.ndim != 3:
        raise ValueError("expected joint P/z0 channels [K+1,H,W]")
    c, h, w = spatial.shape
    indices = spatial_permutation(h, w, image_id, output_rank, phase, seed)
    return spatial.reshape(c, h * w).index_select(
        1, torch.tensor(indices, dtype=torch.long, device=spatial.device)
    ).reshape(c, h, w)


def sample_spatial_roi(proto, c0, boxes, input_hw, size=8):
    """GT-free predicted-box sampling at cell centers; bilinear, zero padding.

    proto: [K,Hp,Wp], c0: [N,K], boxes: [N,4] input-pixel xyxy.
    This feature sampling is NOT the native mask decoder or the loss rasterizer.
    """
    import torch
    import torch.nn.functional as F
    if proto.ndim != 3 or c0.ndim != 2 or boxes.shape != (c0.shape[0], 4):
        raise ValueError("invalid P/c0/box shapes")
    if c0.shape[1] != proto.shape[0] or min(input_hw) <= 0 or size <= 0:
        raise ValueError("invalid channel count/input geometry")
    if not all(torch.isfinite(t).all().item() for t in (proto, c0, boxes)):
        raise ValueError("non-finite feature input")
    n = len(c0)
    if n == 0:
        return proto.new_empty((0, proto.shape[0] + 1, size, size), dtype=torch.float32)
    p, c, b = proto.detach().float(), c0.detach().float(), boxes.detach().float()
    if not ((b[:, 2:] - b[:, :2]) > 0).all().item():
        raise ValueError("nonpositive predicted ROI")
    t = (torch.arange(size, device=p.device, dtype=p.dtype) + .5) / size
    xs = b[:, 0, None] + (b[:, 2] - b[:, 0])[:, None] * t
    ys = b[:, 1, None] + (b[:, 3] - b[:, 1])[:, None] * t
    gx = (2 * xs / input_hw[1] - 1)[:, None, :].expand(n, size, size)
    gy = (2 * ys / input_hw[0] - 1)[:, :, None].expand(n, size, size)
    grid = torch.stack((gx, gy), dim=-1)
    sampled_p = F.grid_sample(p[None].expand(n, -1, -1, -1), grid,
                              mode="bilinear", padding_mode="zeros", align_corners=False)
    z0 = (sampled_p * c[:, :, None, None]).sum(dim=1, keepdim=True)
    return torch.cat((sampled_p, z0), dim=1)


def grouped_mask_loss(logits, baseline_logits, roi_masks, matched_targets, preservation_weight=.1):
    """One image: mean(matched-GT instance BCE) + .1*mean(all valid instance BCE).

    Each instance BCE is itself averaged over its frozen predicted-box ROI.
    matched_targets maps output row index -> GT target [H,W], constructed OUTSIDE
    model forward. Unmatched rows never receive pseudo-GT labels.
    Empty ROIs are counted and excluded from BOTH group denominators.
    Returns a differentiable scalar plus detached accounting for logs.
    """
    import torch
    import torch.nn.functional as F
    if logits.ndim != 3 or baseline_logits.shape != logits.shape or roi_masks.shape != logits.shape:
        raise ValueError("expected equal [N,H,W] loss grids")
    if roi_masks.dtype != torch.bool or preservation_weight != .1:
        raise ValueError("frozen Boolean ROI and preservation weight=.1 required")
    if not torch.isfinite(logits).all().item() or not torch.isfinite(baseline_logits).all().item():
        raise ValueError("non-finite loss input")
    if any(type(i) is not int or i < 0 or i >= len(logits) for i in matched_targets):
        raise ValueError("invalid matched output index")
    gt_terms, preservation_terms, empty = [], [], []
    for i in range(len(logits)):
        roi = roi_masks[i]
        if not roi.any().item():
            empty.append(i)
            continue
        preservation_terms.append(F.binary_cross_entropy_with_logits(
            logits[i][roi], baseline_logits[i].detach()[roi].sigmoid(), reduction="mean"))
        if i in matched_targets:
            target = matched_targets[i].detach()
            if target.shape != logits[i].shape or not torch.isfinite(target).all().item():
                raise ValueError("invalid GT loss target")
            if not ((target >= 0) & (target <= 1)).all().item():
                raise ValueError("GT targets must be probabilities in [0,1]")
            gt_terms.append(F.binary_cross_entropy_with_logits(
                logits[i][roi], target[roi].to(logits), reduction="mean"))
    zero = logits.sum() * 0
    gt = torch.stack(gt_terms).mean() if gt_terms else zero
    preservation = torch.stack(preservation_terms).mean() if preservation_terms else zero
    return gt + preservation_weight * preservation, {
        "gt_loss": float(gt.detach()), "preservation_loss": float(preservation.detach()),
        "matched_supervised_instances": len(gt_terms),
        "preserved_instances": len(preservation_terms), "empty_roi_rows": empty,
        "no_output_image": len(logits) == 0,
    }
