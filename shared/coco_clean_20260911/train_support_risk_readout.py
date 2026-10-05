"""S061: support-preserving background-risk coefficient readout.

This is a mechanism-directed pilot built on the existing frozen YOLO outputs.
The head remains the S032 73D->128->128->128 global residual head.  During
training, likely background false positives (high frozen response on y=0)
receive more weight, while a hinge constraint prevents the learned head from
reducing its own positive support below the frozen readout.  GT is used only
for training labels; inference uses the learned head and frozen model inputs.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4'); os.environ.setdefault('MKL_NUM_THREADS','4'); os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import torch
import torch.nn.functional as F

def support_risk_loss(z, y, z0, factor):
    """Asymmetric background risk with a frozen-support floor.

    z0 is the unadapted frozen logit.  The risk weight is detached, so it
    identifies where the frozen model is confident on background without
    creating a moving target.  The support hinge is a soft trust-region
    constraint, not a GT-dependent inference rule.
    """
    y = y.float(); prob = z.sigmoid(); p0 = z0.sigmoid().detach()
    pos_n = y.sum(1).clamp_min(1.0); neg_n = (1-y).sum(1).clamp_min(1.0)
    pos = -(y * F.logsigmoid(z)).sum(1) / pos_n
    risk = (1.0 + 1.5 * p0) * (1-y)
    bg = -(risk * F.logsigmoid(-z)).sum(1) / risk.sum(1).clamp_min(1.0)
    dice = 1.0 - (2*(prob*y).sum(1)+1) / (prob.sum(1)+y.sum(1)+1)
    ref = (p0*y).sum(1) / pos_n
    cur = (prob*y).sum(1) / pos_n
    support_hinge = F.relu(ref-cur).square()
    # The three terms have distinct roles: foreground fit, risky-background
    # suppression, and preservation of the original positive support.
    return ((pos + 1.5*bg + dice + 4.0*support_hinge) * factor).mean()

