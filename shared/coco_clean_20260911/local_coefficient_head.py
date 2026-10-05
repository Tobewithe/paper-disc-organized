"""Bilinear spatial coefficient residual vs equal-capacity global residual."""
import torch
from torch import nn

class LocalCoefficientHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(73,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x).reshape(-1,4,32)

def gates(points):
    u,v=points.clamp(0,1).unbind(-1)
    return torch.stack([(1-u)*(1-v),u*(1-v),(1-u)*v,u*v],-1)
