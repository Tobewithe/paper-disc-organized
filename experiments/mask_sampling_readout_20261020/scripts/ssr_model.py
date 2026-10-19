"""Matched sampling/readout intervention; only trainable coefficient residuals."""
from __future__ import annotations
import math
from pathlib import Path
import sys
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/mask_readout_mechanism_v2_20261012/scripts'))
from adapter_model import CoefficientAdapter

ARMS=('A','B','C','D','M','G20k')

def coarsen(field):
    """Deterministic information reduction on an unchanged spatial canvas."""
    h,w=field.shape[-2:]
    if h<=8 and w<=8:return field
    small=F.interpolate(field,size=(min(h,8),min(w,8)),mode='bilinear',align_corners=False)
    return F.interpolate(small,size=(h,w),mode='bilinear',align_corners=False)

def permutation(count,image_id,rank,height,width,device):
    if count<=1:return torch.arange(count,device=device)
    value=f'ssr-correspondence-v1:20261020:{image_id}:{rank}:{height}:{width}'
    seed=int.from_bytes(hashlib.sha256(value.encode()).digest()[:8],'little')%(2**63-1)
    gen=torch.Generator(device='cpu').manual_seed(seed)
    result=torch.randperm(count,generator=gen)
    if torch.equal(result,torch.arange(count)):result=result.roll(1)
    return result.to(device)

def positions(h,w,device,dtype):
    y=(torch.arange(h,device=device,dtype=dtype)+.5)/h
    x=(torch.arange(w,device=device,dtype=dtype)+.5)/w
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    f=2**torch.arange(8,device=device,dtype=dtype)
    ax=2*math.pi*xx[None]*f[:,None,None]
    ay=2*math.pi*yy[None]*f[:,None,None]
    return torch.cat((ax.sin(),ax.cos(),ay.sin(),ay.cos()))

class SamplingReadout(CoefficientAdapter):
    def __init__(self,arm,initial):
        if arm not in ARMS[:-1]:raise ValueError('unknown matched arm')
        super().__init__('B')
        self.load_state_dict(initial['state_dict'],strict=True)
        self.ssr_arm=arm
        self.query=nn.Linear(117,32)
        self.key=nn.Linear(64,32)
        self.channel=nn.Linear(32,64)

    def parameter_manifest(self):
        return {'arm':self.ssr_arm,'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'tensors':{n:list(p.shape) for n,p in self.named_parameters()},'pool':'uniform' if self.ssr_arm in ('A','B') else 'instance_query',
                'sampling':'coarse_reconstructed' if self.ssr_arm in ('A','C') else 'native_canvas'}

    def _chunk(self,p,regional,c0,context,box_rows,image_id,ranks,mode):
        """Variable native ROIs padded only within a fixed four-row compute block."""
        ph,pw=p.shape[-2:]
        fields=[];offsets=[]
        for j,box in enumerate(box_rows):
            x0,y0,x1,y1=box
            pc=p[:,y0:y1,x0:x1]
            ff=regional[j][:,y0:y1,x0:x1]
            if mode in ('A','C'):
                pc=coarsen(pc[None])[0];ff=coarsen(ff[None])[0]
            if mode in ('M','J'):
                idx=permutation(ff.shape[-2]*ff.shape[-1],image_id,ranks[j],ff.shape[-2],ff.shape[-1],ff.device)
                ff=ff.flatten(1)[:,idx].reshape_as(ff)
            z=torch.einsum('c,chw->hw',c0[j],pc)[None]
            fields.append((torch.cat((pc,z)),ff,positions(pc.shape[-2],pc.shape[-1],pc.device,pc.dtype)))
            offsets.append((pc.shape[-2],pc.shape[-1]))
        mh=max(s[0] for s in offsets);mw=max(s[1] for s in offsets)
        def pad(t):return F.pad(t,(0,mw-t.shape[-1],0,mh-t.shape[-2]))
        spatial=torch.stack([pad(v[0]) for v in fields])
        region=torch.stack([pad(v[1]) for v in fields])
        pos=torch.stack([pad(v[2]) for v in fields])
        mask=torch.stack([pad(p.new_ones((1,h,w))) for h,w in offsets])
        h1=self.spatial[1](self.spatial[0](spatial))*mask
        h2=self.spatial[3](self.spatial[2]((h1+region)*mask))*mask
        pixels=mask.sum((-2,-1)).clamp_min(1)
        keys=self.key(h2.permute(0,2,3,1)).permute(0,3,1,2)+pos
        keys=F.layer_norm(keys.permute(0,2,3,1),(32,)).permute(0,3,1,2)
        query=F.layer_norm(self.query(context),(32,))
        kmean=(keys*mask).sum((-2,-1))/pixels
        gate=1+.1*torch.tanh(self.channel(torch.tanh(query+kmean)))
        if mode in ('A','B'):
            pooled=h2.sum((-2,-1))/pixels
        else:
            logits=(keys*query[:,:,None,None]).sum(1)/math.sqrt(32)
            logits=logits.masked_fill(~mask[:,0].bool(),float('-inf'))
            weights=logits.flatten(1).softmax(1).reshape_as(logits)
            pooled=(h2*weights[:,None]).sum((-2,-1))
        return self.readout(torch.cat((pooled*gate,context),dim=1))

    def forward(self,cache,pyramid_features,phase='train:0',intervention=None):
        if intervention not in (None,'J'):raise ValueError('only predeclared J intervention')
        if intervention=='J' and self.ssr_arm!='D':raise ValueError('J belongs to D only')
        mode=intervention or self.ssr_arm
        device=self.readout[0].weight.device
        p=cache['proto'].detach().to(device).float()
        c0=cache['c0'].detach().to(device).float()
        boxes=cache['boxes_input'].detach().to(device).float()
        if len(c0)==0:return self.readout(p.new_empty((0,181)))
        ph,pw=p.shape[-2:];ih,iw=cache['input_hw']
        point=p.new_zeros((len(c0),64));maps={}
        levels=[loc['level'] for loc in cache['raw_locations']]
        for level,proj in enumerate(self.feature):
            indices=[i for i,l in enumerate(levels) if l==level]
            if not indices:continue
            ff=pyramid_features[level]
            if ff.ndim==4:ff=ff[cache['batch_position']]
            ff=ff.detach().to(device).float()
            mapped=F.conv2d(ff[None],proj.weight[:,:,None,None],proj.bias)
            maps[level]=F.interpolate(mapped,size=(ph,pw),mode='bilinear',align_corners=False)[0]
            old=torch.stack([cache['features'][i].detach().to(device).float() for i in indices])
            point=point.index_copy(0,torch.tensor(indices,device=device),proj(old))
        context=torch.cat((point,self.coefficient(c0),boxes/boxes.new_tensor([iw,ih,iw,ih]),
                           cache['scores'].detach().to(device).float()[:,None],
                           self.category(cache['classes'].detach().to(device).long())),dim=1)
        scaled=(boxes*boxes.new_tensor([pw/iw,ph/ih,pw/iw,ph/ih])).detach().cpu().tolist()
        rect=[];valid=[]
        for i,(x0,y0,x1,y1) in enumerate(scaled):
            r=(max(0,min(pw,math.ceil(x0))),max(0,min(ph,math.ceil(y0))),
               max(0,min(pw,math.ceil(x1))),max(0,min(ph,math.ceil(y1))))
            rect.append(r)
            if r[2]>r[0] and r[3]>r[1]:valid.append(i)
        result=c0.new_zeros(c0.shape)
        for start in range(0,len(valid),4):
            ranks=valid[start:start+4]
            # Projection bias cancels in the regional-minus-point injection, as RFC.
            regional=torch.stack([maps[levels[i]]-point[i,:,None,None] for i in ranks])
            idx=torch.tensor(ranks,device=device)
            cr=c0[idx];ctx=context[idx];br=tuple(rect[i] for i in ranks);rr=tuple(ranks)
            def compute(pp,ff,cc,xx,br=br,rr=rr,mode=mode,iid=cache['image_id']):
                return self._chunk(pp,ff,cc,xx,br,iid,rr,mode)
            if self.training and torch.is_grad_enabled():
                value=checkpoint(compute,p,regional,cr,ctx,use_reentrant=False,preserve_rng_state=False)
            else:value=compute(p,regional,cr,ctx)
            result=result.index_copy(0,idx,value)
        # Retain differentiability for an image whose every output has empty ROI.
        return result+self.readout[-1].weight.sum()*0

def make_model(arm,initial,seed=0):
    torch.manual_seed(seed)
    if arm=='G20k':
        model=CoefficientAdapter('B');model.load_state_dict(initial['state_dict'],strict=True)
        model.ssr_arm='G20k';return model
    return SamplingReadout(arm,initial)

def predict(model,cache,features,phase='train:0',intervention=None):
    if model.ssr_arm=='G20k':
        if intervention:raise ValueError('G has no intervention')
        return model(cache,phase)
    return model(cache,features,phase,intervention)
