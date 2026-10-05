"""Frozen readout/input pilot. No ground truth enters input construction.

This module can be checked locally without Ultralytics. Real COCO results require
the separate cache builder on the pinned remote runtime.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ARMS = ['linear', 'own', 'mean', 'ordered', 'shuffled']


def stable_seed(*parts):
    value = ':'.join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'little') % (2**63 - 1)


def sample_regions(feature_maps, levels, boxes, input_shape):
    """Nine points per predicted box, width/height x1.2, same feature scale.

    Coordinates are feature-cell-center aligned (align_corners=False). Border
    extension is explicit. Final masks still use the ORIGINAL predicted box.
    """
    if not len(boxes):
        return boxes.new_empty((0, 9, feature_maps[0].shape[1]))
    ih, iw = input_shape
    axis = boxes.new_tensor([1 / 6, 1 / 2, 5 / 6])
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    uv = torch.stack([xx, yy], -1).reshape(1, 9, 2)
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1) * 1.2
    points = center[:, None] + (uv - .5) * wh[:, None]
    grid = points / boxes.new_tensor([iw, ih]) * 2 - 1
    result = boxes.new_empty((len(boxes), 9, feature_maps[0].shape[1]))
    for level, fmap in enumerate(feature_maps):
        idx = (levels == level).nonzero().flatten()
        if len(idx):
            picked = F.grid_sample(fmap, grid[idx][None], mode='bilinear',
                                   padding_mode='border', align_corners=False)
            result[idx] = picked[0].permute(1, 2, 0)
    return result


def inputs(h, region, level, mode, identities, seed, draw, normalizer):
    """Only predicted h/region/level and nonsemantic RNG keys are accepted.

    Image/source identity seeds a permutation, but is never a network feature.
    All nonlinear arms have equal 10*C+3 inputs and equal active architecture.
    """
    mu, sd = normalizer
    own = (h - mu) / sd
    spatial = (region - mu) / sd
    if mode == 'own':
        spatial = own[:, None].expand(-1, 9, -1)
    elif mode == 'mean':
        spatial = spatial.mean(1, keepdim=True).expand(-1, 9, -1)
    elif mode == 'shuffled':
        orders = [np.random.default_rng(stable_seed('layout', seed, draw, *key)).permutation(9)
                  for key in identities]
        order = torch.as_tensor(np.asarray(orders,dtype=np.int64).reshape(-1,9), device=h.device)
        spatial = spatial.gather(1, order[:, :, None].expand(-1, -1, h.shape[1]))
    elif mode != 'ordered':
        raise ValueError(mode)
    return torch.cat([own, spatial.flatten(1), F.one_hot(level, 3).to(h.dtype)], 1)


class Readout(nn.Module):
    def __init__(self, channels, linear=False, dtype=torch.float32):
        super().__init__()
        self.linear = linear
        if linear:
            self.weight = nn.Parameter(torch.zeros(3, 32, channels, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(3, 32, dtype=dtype))
        else:
            self.net = nn.Sequential(nn.Linear(channels * 10 + 3, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 32)).to(dtype)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, level):
        if self.linear:
            return torch.einsum('nkh,nh->nk', self.weight[level], x) + self.bias[level]
        return self.net(x)


def objective(delta, c0, p, y, factor):
    z = torch.einsum('nsk,nk->ns', p, c0 + delta)
    # Fixed native640 GT support; factor restores mean full-grid BCE / GT area.
    return (F.binary_cross_entropy_with_logits(z, y, reduction='none').mean(1) * factor).mean()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
