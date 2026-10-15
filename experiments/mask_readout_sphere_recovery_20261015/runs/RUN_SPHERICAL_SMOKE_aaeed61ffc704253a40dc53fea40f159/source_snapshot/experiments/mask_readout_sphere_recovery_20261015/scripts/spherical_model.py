"""GT-free spherical residual model; c0/P/native detector are never trainable."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/mask_readout_mechanism_v2_20261012/scripts'))
sys.path.insert(0,str(ROOT/'experiments/mask_readout_subspace_repulsion_20261015/scripts'))
from adapter_model import CoefficientAdapter
from repulsion import choose_neighbors


def tangent_residual(c,raw,gate):
    r2=c.square().sum(1,keepdim=True);den=r2.clamp_min(1e-8)
    v=raw-(raw*c).sum(1,keepdim=True)*c/den
    t=gate*v;scale=torch.rsqrt(1+t.square().sum(1,keepdim=True)/den)
    delta=c*(scale-1)+t*scale
    return torch.where(r2>1e-8,delta,torch.zeros_like(delta))


def neighbor_inputs(cache,device):
    c=cache['c0'].detach().to(device).float();p=cache['proto'].detach().to(device).float();boxes=cache['boxes_input'].detach().to(device).float()
    n=len(c)
    if n==0:return c.new_zeros((0,41)),c.new_zeros((0,1)),[]
    neighbors,ious=choose_neighbors(cache['boxes_input'].tolist(),cache['scores'].tolist(),cache['classes'].tolist(),cache['raw_ids'])
    index=torch.tensor([max(0,j) for j in neighbors],device=device);valid=torch.tensor([j>=0 for j in neighbors],device=device)
    h,w=p.shape[-2:];ih,iw=cache['input_hw'];x=torch.arange(w,device=device)[None,None,:]*(iw/w);y=torch.arange(h,device=device)[None,:,None]*(ih/h)
    other=boxes[index];lo=torch.maximum(boxes[:,:2],other[:,:2]);hi=torch.minimum(boxes[:,2:],other[:,2:])
    overlap=(x>=lo[:,0,None,None])&(x<hi[:,0,None,None])&(y>=lo[:,1,None,None])&(y<hi[:,1,None,None])&valid[:,None,None]
    z=(c@p.flatten(1)).reshape(n,h,w);zj=z[index];count=overlap.sum((1,2));div=count.clamp_min(1)
    collision=(z.sigmoid()*zj.sigmoid()*overlap).sum((1,2))/div
    entropy=((F.softplus(z)-z*z.sigmoid())*overlap).sum((1,2))/div
    positive=((z>0)&(zj>0)&overlap).sum((1,2));gate=valid&(positive>=4)&(collision>.25)
    wh=(boxes[:,2:]-boxes[:,:2]).clamp_min(1e-6);other_wh=wh[index]
    relcenter=((other[:,:2]+other[:,2:])-(boxes[:,:2]+boxes[:,2:]))/2/wh
    relsize=(other_wh/wh).log();score=cache['scores'].detach().to(device).float()
    inputs=torch.cat((c-c[index],relcenter,relsize,c.new_tensor(ious)[:,None],score[:,None],score[index,None],collision[:,None],entropy[:,None]),1).detach()
    records=[dict(output_rank=i,neighbor_rank=j,neighbor_box_iou=ious[i],omega_pixels=int(positive[i]),collision=float(collision[i]),applied=bool(gate[i])) for i,j in enumerate(neighbors)]
    return inputs,gate[:,None].float(),records


class SphericalAdapter(CoefficientAdapter):
    def __init__(self):
        super().__init__('B')
        self.neighbor=nn.Sequential(nn.Linear(41,64),nn.GELU(),nn.Linear(64,33))
        nn.init.zeros_(self.neighbor[-1].weight);nn.init.zeros_(self.neighbor[-1].bias)
        self.last_records=[]
        self.last_math={}

    def forward(self,cache,phase):
        raw=super().forward(cache,phase);features,hard,records=neighbor_inputs(cache,raw.device)
        extra=self.neighbor(features);gate=hard*extra[:,32:].sigmoid();raw=raw+extra[:,:32]
        c=cache['c0'].detach().to(raw);delta=tangent_residual(c,raw,gate)
        relative=((c+delta).norm(dim=1)-c.norm(dim=1)).abs()/(1+c.norm(dim=1))
        if len(c) and float(relative.detach().max())>1e-5:raise ValueError('spherical norm constraint failed')
        if not torch.equal(delta[hard[:,0]==0],torch.zeros_like(delta[hard[:,0]==0])):raise ValueError('hard gate failed identity')
        self.last_records=records
        self.last_math=dict(raw_delta=raw.detach(),effective_gate=gate.detach())
        return delta


def initialized_model(original_initial,device):
    torch.manual_seed(0);model=SphericalAdapter().to(device)
    saved=torch.load(original_initial,map_location='cpu',weights_only=True)
    result=model.load_state_dict(saved['state_dict'],strict=False)
    assert set(result.missing_keys)=={'neighbor.0.weight','neighbor.0.bias','neighbor.2.weight','neighbor.2.bias'} and not result.unexpected_keys
    assert all(torch.equal(v.cpu(),model.state_dict()[k].cpu()) for k,v in saved['state_dict'].items())
    assert sum(p.numel() for p in model.parameters())==172705
    return model
