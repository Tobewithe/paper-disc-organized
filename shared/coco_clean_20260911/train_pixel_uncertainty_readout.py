"""S066 pixelwise uncertainty weighted coefficient readout loss."""
import torch
import torch.nn.functional as F

def pixel_uncertainty_loss(z, y, z0, factor):
    y=y.float(); z0=z0.detach(); p0=z0.sigmoid().detach()
    # The frozen response is uncertain near the decision boundary. Weight
    # those pixels directly rather than multiplying the entire instance by a
    # single mean uncertainty.
    w=1.0+4.0*(4.0*p0*(1.0-p0))
    bce=(F.binary_cross_entropy_with_logits(z,y,reduction='none')*w).mean(1)
    prob=z.sigmoid(); dice=1-(2*(prob*y).sum(1)+1)/(prob.sum(1)+y.sum(1)+1)
    return ((bce+dice)*factor).mean()
