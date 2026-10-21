"""Frozen functional target/loss; GT appears only in the target provider."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from adapter_model import loss_grids
from s1_geometry import predicted_roi


def target(cache, row):
    rank = row['rank']
    if cache['prediction_indices'][rank] != row['global_index']:
        raise ValueError('teacher/output identity mismatch')
    p = cache['proto'].detach().cpu().double().flatten(1)
    delta = torch.tensor(row['coefficients_fp32'], dtype=torch.float32).double() - cache['c0'][rank].detach().cpu().double()
    response = (delta @ p).reshape(cache['proto'].shape[-2:])
    roi = predicted_roi(cache, rank)
    if not roi.any():
        raise ValueError('accepted teacher has empty ROI')
    scale = response[roi].square().mean().sqrt().item()
    if not torch.isfinite(response).all() or not scale > 1e-6:
        raise ValueError('accepted teacher has invalid functional amplitude')
    return (response / scale).float().detach(), scale


def context(cache, device):
    c0 = cache['c0'].detach().to(device).float()
    # The inherited full-matrix expression fixes the original baseline and ROI.
    _, baseline, roi = loss_grids(cache, c0)
    p = cache['proto'].detach().to(device).float().flatten(1)
    return dict(c0=c0, p=p, baseline=baseline, roi=roi, shape=cache['proto'].shape[-2:])


def remap(direction, mapping, arm):
    if arm not in ('REAL', 'PERM', 'J'):
        raise ValueError('invalid arm')
    if arm == 'REAL':
        return direction
    return direction.index_select(0, torch.as_tensor(mapping, dtype=torch.long, device=direction.device))


def functional_direction(direction, ctx):
    response = (direction @ ctx['p']).reshape(len(direction), *ctx['shape'])
    pixels = ctx['roi'].sum((1, 2))
    square_mean = (response.square() * ctx['roi']).sum((1, 2)) / pixels.clamp_min(1)
    # sqrt(clamp(x,1)) == max(1,sqrt(x)); avoids 0*infinity at zero initialization.
    denominator = square_mean.clamp_min(1).sqrt()
    bounded = direction / denominator[:, None]
    bounded = bounded * (pixels > 0)[:, None]
    return response, bounded, square_mean, pixels


def loss(direction, ctx, targets, amplitude):
    response, bounded, square_mean, pixels = functional_direction(direction, ctx)
    zero = direction.sum() * 0
    terms = []
    for rank, item in targets.items():
        roi = ctx['roi'][rank]
        if not roi.any():
            raise ValueError('accepted teacher has empty target ROI')
        terms.append((response[rank][roi] - item.to(direction)[roi]).square().mean())
    direction_loss = torch.stack(terms).mean() if terms else zero
    coefficients = ctx['c0'] + float(amplitude) * bounded
    logits = (coefficients @ ctx['p']).reshape_as(ctx['baseline'])
    per_row = (F.binary_cross_entropy_with_logits(logits, ctx['baseline'].sigmoid().detach(), reduction='none') * ctx['roi']).sum((1, 2)) / pixels.clamp_min(1)
    preservation = per_row[pixels > 0].mean() if (pixels > 0).any() else zero
    total = direction_loss + .1 * preservation
    return total, dict(direction_loss=float(direction_loss.detach()), preservation_loss=float(preservation.detach()),
                      teachers=len(terms), outputs=len(direction), empty_roi_rows=int((pixels == 0).sum()),
                      clipped_rows=int((square_mean.detach() > 1).sum()))


def deploy_coefficients(direction, ctx, amplitude):
    return ctx['c0'] + float(amplitude) * functional_direction(direction, ctx)[1]


def probe_coefficients(direction, ctx, teacher_scales):
    """GT-assisted equal amplitude, only for already frozen evaluation objects."""
    response, _, square_mean, _ = functional_direction(direction, ctx)
    result = ctx['c0'].clone()
    for rank, scale in teacher_scales.items():
        rms = square_mean[rank].sqrt()
        if float(rms) > 1e-6:
            result[rank] += float(scale) * direction[rank] / rms
    return result
