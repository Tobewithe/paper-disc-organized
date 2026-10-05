"""Prediction-only local ownership features and a zero-initialized pixel residual.

No GT enters build_features. All pixel coordinates and boxes must share input-space
units; training converts original points to input coordinates before calling it.
"""
import torch
from torch import nn

FEATURE_DIM=14

def build_features(logits,boxes,detections,target,points,mode):
    # logits: [predictions, points]; points:[points,2] xy in letterboxed input.
    own=logits[target];box=boxes[target];wh=(box[2:]-box[:2]).clamp_min(1)
    uv=(points-box[:2])/wh;center=2*uv-1
    own_feature=torch.stack([own.clamp(-10,10)/5,own.sigmoid(),center[:,0],center[:,1],center[:,0].square(),center[:,1].square(),
                             torch.full_like(own,detections[target,4]),torch.full_like(own,(wh[0]/wh[1]).log().clamp(-3,3)/3)],dim=1)
    adjacent=torch.zeros((len(points),6),device=own.device)
    if mode=='neighbor':
        inter=(torch.minimum(box[2:],boxes[:,2:])-torch.maximum(box[:2],boxes[:,:2])).clamp_min(0).prod(1)
        area=(boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(1);iou=inter/(wh.prod()+area-inter).clamp_min(1e-6)
        eligible=(detections[:,5]==detections[target,5])&(detections[:,4]>=.1)&(iou>.05)
        eligible[target]=False;ids=eligible.nonzero().flatten()
        if len(ids):
            b=boxes[ids];inside=((points[None]>=b[:,None,:2])&(points[None]<b[:,None,2:])).all(2)
            values=logits[ids].masked_fill(~inside,-float('inf'));best,k=values.max(0);exists=inside.any(0)
            picked=ids[k];otherbox=boxes[picked];v=(points-otherbox[:,:2])/(otherbox[:,2:]-otherbox[:,:2]).clamp_min(1)
            adjacent=torch.stack([best.clamp(-10,10)/5,(own-best).clamp(-10,10)/5,2*v[:,0]-1,2*v[:,1]-1,detections[picked,4],exists.float()],dim=1)
            adjacent=torch.where(exists[:,None],adjacent,torch.zeros_like(adjacent))
    elif mode!='own':raise ValueError(mode)
    result=torch.cat([own_feature,adjacent],1);assert torch.isfinite(result).all()
    return result

class PixelRefiner(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(FEATURE_DIM,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x).squeeze(-1)
