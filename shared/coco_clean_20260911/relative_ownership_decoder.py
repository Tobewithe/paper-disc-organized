"""Prediction-only area-preserving relative mask ranking, frozen decoder."""
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils import ops


def keep_topk(scores, count):
    """Exact cardinality, index-stable ties, including empty/full supports."""
    n = scores.numel()
    if count == 0:
        return torch.zeros(n, dtype=torch.bool, device=scores.device)
    if count == n:
        return torch.ones(n, dtype=torch.bool, device=scores.device)
    boundary = torch.kthvalue(scores, n - count + 1).values
    keep = scores > boundary
    remaining = count - int(keep.sum())
    ties = (scores == boundary).nonzero().flatten()
    keep[ties[:remaining]] = True
    if int(keep.sum()) != count:
        raise RuntimeError('Top-k cardinality mismatch')
    return keep


@torch.inference_mode()
def decode(item, arms, iid, config):
    """Accepts only model outputs, no annotations or GT ownership mappings.

    arms maps name to dict(kind=initial|relative|geometry|random|threshold,
    value=alpha or threshold, seed=shuffle seed).
    """
    p = torch.as_tensor(item['proto'], device='cuda').float()
    c = torch.as_tensor(item['coeff'], device='cuda').float()
    boxes = torch.as_tensor(item['boxes'], device='cuda').float()
    det = torch.as_tensor(item['detections'], device='cuda').float()
    shape = tuple(map(int, item['input_shape']))
    n = len(c)
    if not n:
        return {k: torch.zeros((0, *shape), dtype=torch.uint8, device='cuda') for k in arms}, [], 0
    z = F.interpolate((c @ p.flatten(1)).reshape(1, n, *p.shape[-2:]),
                      shape, mode='bilinear', align_corners=False)[0]
    if not torch.isfinite(z).all():
        raise RuntimeError('Nonfinite decoded logits')
    original = ops.process_mask(p, c, boxes, shape, upsample=True)
    binary = ops.crop_mask((z > 0).to(torch.uint8), boxes)
    xor = int((original != binary).sum())
    if xor:
        raise RuntimeError(f'Official decoder mismatch: {xor}')
    result = {}
    for name, arm in arms.items():
        result[name] = (ops.crop_mask((z > arm['value']).to(torch.uint8), boxes)
                        if arm['kind'] == 'threshold' else binary.clone())
    support = ops.crop_mask(torch.ones_like(binary), boxes).bool()
    rms = ((z.square() * support).sum((1, 2)) / support.sum((1, 2)).clamp_min(1)).sqrt().clamp_min(.1)
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1)
    centers = (boxes[:, 2:] + boxes[:, :2]) / 2
    area = wh.prod(1)
    lt = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    rb = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(2)
    iou = inter / (area[:, None] + area[None, :] - inter).clamp_min(1e-6)
    allowed = ((det[:, None, 5] == det[None, :, 5]) &
               (det[:, None, 4] >= config['target_and_neighbor_conf_min']) &
               (det[None, :, 4] >= config['target_and_neighbor_conf_min']) &
               (iou > config['box_iou_min']))
    allowed.fill_diagonal_(False)
    records = []
    for j in range(n):
        candidates = allowed[j].nonzero().flatten()
        if not len(candidates):
            continue
        neighbors = candidates[torch.argsort(iou[j, candidates], descending=True, stable=True)[:config['max_neighbors']]]
        positions = support[j].flatten().nonzero().flatten()
        if not len(positions):
            continue
        base = binary[j].flatten()[positions].bool()
        count = int(base.sum())
        own = z[j].flatten()[positions] / rms[j]
        neighbor_support = support[neighbors].flatten(1)[:, positions]
        overlap = neighbor_support.any(0)
        relative = ((z[neighbors].flatten(1)[:, positions] / rms[neighbors, None]).clamp(0, 2)
                    * neighbor_support).amax(0)
        # Grid indices agree with the actual crop support; no extra half-pixel shift.
        y = torch.div(positions, shape[1], rounding_mode='floor')
        x = positions % shape[1]
        xy = torch.stack([x, y], 1).float()
        own_distance = ((xy - centers[j]) / (wh[j] / 2)).square().sum(1)
        neighbor_distance = ((xy[None] - centers[neighbors, None]) / (wh[neighbors, None] / 2)).square().sum(2)
        geometry = ((own_distance[None] - neighbor_distance).clamp(0, 2) * neighbor_support).amax(0)
        shuffled = {}
        for name, arm in arms.items():
            kind = arm['kind']
            if kind in ('initial', 'threshold') or arm['value'] == 0:
                continue
            if kind == 'relative':
                penalty = relative
            elif kind == 'geometry':
                penalty = geometry
            elif kind == 'random':
                seed = arm['seed']
                if seed not in shuffled:
                    generator = torch.Generator(device='cuda')
                    generator.manual_seed(20260911 + iid * 1009 + j * 131 + seed * 7907)
                    values = relative[overlap]
                    permutation = torch.randperm(len(values), device='cuda', generator=generator)
                    q = relative.clone()
                    q[overlap] = values[permutation]
                    shuffled[seed] = q
                penalty = shuffled[seed]
            else:
                raise ValueError(kind)
            kept = keep_topk(own - arm['value'] * penalty, count)
            result[name][j].flatten()[positions] = kept.to(torch.uint8)
            records.append(dict(image_id=iid, prediction_index=j, arm=name,
                                neighbor_indices=','.join(map(str, neighbors.cpu().tolist())),
                                original_area=count, resulting_area=int(kept.sum()),
                                changed_pixels=int((kept != base).sum()),
                                correction_rms=float((arm['value'] * penalty).square().mean().sqrt())))
    base_counts = binary.flatten(1).sum(1)
    for name, arm in arms.items():
        if arm['kind'] != 'threshold' and not torch.equal(result[name].flatten(1).sum(1), base_counts):
            raise RuntimeError('Area preservation failed')
    return result, records, xor
