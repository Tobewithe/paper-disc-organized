"""Small zero-initialized residual coefficient head with optional box-pooled P.

Both arms have identical dimensions and parameters. Own arm hides only pooled P;
spatial arm receives a prediction-box 4x4 sample of the prototype feature maps.
"""
import torch
from torch import nn
import torch.nn.functional as F


def features(base,levels,boxes,input_shape,protos=None):
    h,w=input_shape;n=len(base);scaled=boxes/boxes.new_tensor([w,h,w,h]);wh=(scaled[:,2:]-scaled[:,:2]).clamp_min(1e-5)
    geometry=torch.cat([scaled,wh],1)
    common=torch.cat([base,F.one_hot(levels,3).float(),geometry],1)
    spatial=base.new_zeros((n,512))
    if protos is not None and n:
        v=(torch.arange(4,device=base.device)+.5)/4
        gy,gx=torch.meshgrid(v,v,indexing='ij');unit=torch.stack([gx.flatten(),gy.flatten()],1)
        xy=scaled[:,:2,None].transpose(1,2)+unit[None]*wh[:,None]
        grid=(2*xy-1).reshape(1,n*4,4,2)
        sample=F.grid_sample(protos[None],grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0]
        spatial=sample.reshape(32,n,16).permute(1,0,2).flatten(1)
    result=torch.cat([common,spatial],1);assert result.shape==(n,585) and torch.isfinite(result).all()
    return result


class SpatialCoefficientHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(585,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,32))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x)
