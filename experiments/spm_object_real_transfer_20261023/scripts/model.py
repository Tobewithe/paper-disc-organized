"""A zero-initialized ROI logit residual; backbone and object bank stay frozen."""
import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bound = cfg['delta_logit_bound']
        self.net = nn.Sequential(nn.Conv2d(26, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.SiLU(),
                                 nn.Conv2d(48, 48, 3, padding=2, dilation=2), nn.GroupNorm(8, 48), nn.SiLU(),
                                 nn.Conv2d(48, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
                                 nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, common, prior):
        return self.bound*torch.tanh(self.net(torch.cat([common, prior], 1)))[:, 0]


def prior_maps(bank, indices, mode='structured'):
    value = bank['maps'][indices].float()
    if mode in ('scrambled_control', 'scrambled'):
        value = value.flatten(2)[:, :, bank['pixel_permutation']].reshape_as(value)
    elif mode == 'off':
        value = torch.zeros_like(value)
    else:
        assert mode == 'structured', mode
    return value


def loss_for(model, common, prior, native, target, cfg):
    delta = model(common, prior)
    logits = native+delta
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid()
    dice = (1-(2*(p*target).sum((-2, -1))+1)/(p.sum((-2, -1))+target.sum((-2, -1))+1)).mean()
    reg = delta.square().mean()
    w = cfg['loss_weights']
    return w['bce']*bce+w['dice']*dice+w['delta_l2']*reg, {'bce': bce, 'dice': dice, 'delta_l2': reg}


def roi_grid(boxes, size, canvas=640):
    t = (torch.arange(size, device=boxes.device, dtype=boxes.dtype)+.5)/size
    y, x = torch.meshgrid(t, t, indexing='ij')
    xx = boxes[:, 0, None, None]+x*(boxes[:, 2]-boxes[:, 0])[:, None, None]
    yy = boxes[:, 1, None, None]+y*(boxes[:, 3]-boxes[:, 1])[:, None, None]
    return torch.stack([xx/canvas*2-1, yy/canvas*2-1], -1)


def crop_tensor(image, boxes, size, canvas=640):
    return F.grid_sample(image.expand(len(boxes), -1, -1, -1), roi_grid(boxes, size, canvas),
                         mode='bilinear', padding_mode='zeros', align_corners=False)


def scatter_delta(delta, boxes, size=160, canvas=640):
    t = (torch.arange(size, device=boxes.device, dtype=boxes.dtype)+.5)*canvas/size
    y, x = torch.meshgrid(t, t, indexing='ij')
    xx = (x-boxes[:, 0, None, None])/(boxes[:, 2]-boxes[:, 0]).clamp_min(1e-6)[:, None, None]
    yy = (y-boxes[:, 1, None, None])/(boxes[:, 3]-boxes[:, 1]).clamp_min(1e-6)[:, None, None]
    grid = torch.stack([xx*2-1, yy*2-1], -1)
    values = F.grid_sample(delta[:, None], grid, align_corners=False)[:, 0]
    return values*((xx >= 0) & (xx < 1) & (yy >= 0) & (yy < 1))
