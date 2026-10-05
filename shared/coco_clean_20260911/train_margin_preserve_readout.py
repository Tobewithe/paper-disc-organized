"""S062 loss: foreground-logit trust region plus background-risk suppression."""
import torch
import torch.nn.functional as F

def margin_preserving_loss(z, y, z0, factor):
    y = y.float(); z0 = z0.detach(); p0 = z0.sigmoid()
    pos_n = y.sum(1).clamp_min(1.0)
    # Fit the foreground while penalizing only background responses that the
    # frozen readout already considers plausible. This keeps the intervention
    # targeted at false-positive risk rather than all negative pixels.
    pos = (F.softplus(-z) * y).sum(1) / pos_n
    risk = (1.0 + 1.5 * p0) * (1.0-y)
    bg = (F.softplus(z) * risk).sum(1) / risk.sum(1).clamp_min(1.0)
    prob = z.sigmoid()
    dice = 1.0 - (2*(prob*y).sum(1)+1)/(prob.sum(1)+y.sum(1)+1)
    # Per-positive-pixel trust region. A mean support constraint allowed a
    # few pixels to collapse; this margin prevents that failure mode.
    trust = (F.relu((z0 - 0.25) - z) * y).sum(1) / pos_n
    return ((pos + 0.75*bg + dice + 6.0*trust) * factor).mean()
