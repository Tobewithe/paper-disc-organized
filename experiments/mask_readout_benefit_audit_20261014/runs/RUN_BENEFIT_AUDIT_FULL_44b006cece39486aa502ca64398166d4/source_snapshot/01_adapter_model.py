"""Frozen-cache, GT-free coefficient residual module for the preregistered B/C pilot."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from adapter_primitives import sample_spatial_roi, spatial_permutation


class CoefficientAdapter(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ('B', 'C'):
            raise ValueError('B or C required')
        self.arm = arm
        self.spatial = nn.Sequential(nn.Conv2d(33, 64, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.feature = nn.ModuleList([nn.Linear(channels, 64) for channels in (256, 512, 512)])
        self.coefficient = nn.Linear(32, 32)
        self.category = nn.Embedding(80, 16)
        self.readout = nn.Sequential(nn.Linear(181, 128), nn.GELU(), nn.Linear(128, 32))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, cache, phase):
        # No GT, matching, success/failure labels, or oracle coefficients are accepted.
        device = self.readout[0].weight.device
        p, c, boxes = (cache[k].detach().to(device).float() for k in ('proto', 'c0', 'boxes_input'))
        count = len(c)
        if c.shape != (count, 32) or [s[0] for s in cache['head_feature_shapes']] != [256, 512, 512]:
            raise ValueError('native feature architecture changed')
        spatial = sample_spatial_roi(p, c, boxes, cache['input_hw'])
        if self.arm == 'C':
            permutations = torch.tensor([spatial_permutation(8, 8, cache['image_id'], rank, phase)
                                         for rank in range(count)], device=device, dtype=torch.long).reshape(count, 64)
            spatial = spatial.flatten(2).gather(2, permutations[:, None].expand(-1, 33, -1)).reshape(count, 33, 8, 8)
        spatial = self.spatial(spatial).mean(dim=(-2, -1))
        features = c.new_zeros((count, 64))
        for level, projection in enumerate(self.feature):
            indices = [i for i, location in enumerate(cache['raw_locations']) if location['level'] == level]
            if indices:
                actual = torch.stack([cache['features'][i].detach() for i in indices]).to(device).float()
                features = features.index_copy(0, torch.tensor(indices, device=device), projection(actual))
        height, width = cache['input_hw']
        normalized_boxes = boxes / boxes.new_tensor([width, height, width, height])
        inputs = torch.cat((spatial, features, self.coefficient(c), normalized_boxes,
                            cache['scores'].detach().to(device)[:, None],
                            self.category(cache['classes'].detach().to(device).long())), dim=1)
        return self.readout(inputs)


def loss_grids(cache, coefficients):
    """Only coefficient gradients; every final output remains in its native row."""
    p = cache['proto'].detach().to(coefficients).flatten(1)
    c0 = cache['c0'].detach().to(coefficients)
    height, width = cache['proto'].shape[-2:]
    logits = (coefficients @ p).reshape(len(coefficients), height, width)
    baseline = (c0 @ p).reshape_as(logits).detach()
    ih, iw = cache['input_hw']
    boxes = cache['boxes_input'].detach().to(coefficients) * coefficients.new_tensor([width/iw, height/ih, width/iw, height/ih])
    x = torch.arange(width, device=coefficients.device)[None, None, :]
    y = torch.arange(height, device=coefficients.device)[None, :, None]
    roi = ((x >= boxes[:, 0, None, None]) & (x < boxes[:, 2, None, None]) &
           (y >= boxes[:, 1, None, None]) & (y < boxes[:, 3, None, None]))
    return logits, baseline, roi


def vector_grouped_loss(logits, baseline, roi, targets):
    """Same separate instance/image denominators as grouped_mask_loss; tested against it."""
    if logits.shape != baseline.shape or logits.shape != roi.shape or roi.dtype != torch.bool:
        raise ValueError('loss grid mismatch')
    if not torch.isfinite(logits).all().item() or not torch.isfinite(baseline).all().item():
        raise ValueError('non-finite loss')
    pixels = roi.sum(dim=(1, 2))
    valid = pixels > 0
    zero = logits.sum() * 0
    per_instance = (F.binary_cross_entropy_with_logits(logits, baseline.detach().sigmoid(), reduction='none') * roi).sum((1, 2)) / pixels.clamp_min(1)
    preservation = per_instance[valid].mean() if valid.any().item() else zero
    indices = [i for i in targets if bool(valid[i])]
    if indices:
        target = torch.stack([targets[i].detach().to(logits) for i in indices])
        if target.shape != logits[indices].shape or not ((target >= 0) & (target <= 1)).all().item():
            raise ValueError('invalid training target')
        terms = F.binary_cross_entropy_with_logits(logits[indices], target, reduction='none') * roi[indices]
        gt = (terms.sum((1, 2)) / pixels[indices]).mean()
    else:
        gt = zero
    return gt + .1 * preservation, {
        'gt_loss': float(gt.detach()), 'preservation_loss': float(preservation.detach()),
        'matched_supervised_instances': len(indices), 'preserved_instances': int(valid.sum()),
        'empty_roi_rows': (~valid).nonzero().flatten().tolist(), 'no_output_image': len(logits) == 0}
