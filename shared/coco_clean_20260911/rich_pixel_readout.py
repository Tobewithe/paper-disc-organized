"""Prediction-only pixel readout; GT never enters this module."""
import torch
import torch.nn.functional as F
from torch import nn

MODES=['global','scalar','rich','rich_neighbor','rich_shuffled']


def instance_features(h,level,boxes,shape):
    height,width=shape
    scaled=boxes/boxes.new_tensor([width,height,width,height])
    wh=(scaled[:,2:]-scaled[:,:2]).clamp_min(1e-5)
    return torch.cat([h,F.one_hot(level,3).float(),scaled,wh],1)


class GlobalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(73,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x).reshape(-1,4,32).mean(1)


class PixelHead(nn.Module):
    """110->64->64->1, split first projection to avoid expanding h at every pixel.

    All pixel arms have exactly the same parameterization; zero slots in scalar
    or rich mean some weights are inactive. No exact effective-capacity claim.
    """
    def __init__(self):
        super().__init__()
        self.local=nn.Linear(46,64)
        self.context=nn.Linear(64,64,bias=False)
        self.hidden=nn.Linear(64,64);self.output=nn.Linear(64,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,scalar,p,h,mode):
        if mode=='scalar':p=torch.zeros_like(p);h=torch.zeros_like(h)
        if mode=='rich':scalar=scalar.clone();scalar[...,8:]=0
        v=self.local(torch.cat([scalar,p],-1))+self.context(h)[:,None]
        return self.output(F.silu(self.hidden(F.silu(v)))).squeeze(-1)


def loss_value(z,y,factor):
    bce=F.binary_cross_entropy_with_logits(z,y,reduction='none').mean(1)*factor
    prob=z.sigmoid()
    # Same sampled native support for every group. Empty-positive targets remain.
    dice=1-(2*(prob*y).sum(1)+1)/(prob.sum(1)+y.sum(1)+1)
    return (bce+dice).mean()
