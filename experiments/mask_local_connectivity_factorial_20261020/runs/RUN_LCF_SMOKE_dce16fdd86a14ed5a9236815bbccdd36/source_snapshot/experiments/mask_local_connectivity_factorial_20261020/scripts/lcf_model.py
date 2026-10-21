"""One effective network; only prototype-span readout versus local field differs."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ARMS=('A','B','C','D')

def boxes_proto(cache):
    h,w=cache['proto'].shape[-2:];ih,iw=cache['input_hw']
    rows=(cache['boxes_input'].detach().cpu()*torch.tensor([w/iw,h/ih,w/iw,h/ih])).tolist()
    return [(max(0,min(w,math.ceil(a))),max(0,min(h,math.ceil(b))),
             max(0,min(w,math.ceil(c))),max(0,min(h,math.ceil(d)))) for a,b,c,d in rows]

def projection_matrices(cache,spec):
    if '_lcf_projection' in cache:return cache['_lcf_projection']
    p=cache['proto'].detach().double();result=[]
    for x0,y0,x1,y1 in boxes_proto(cache):
        if x1<=x0 or y1<=y0:result.append(None);continue
        phi=p[:,y0:y1,x0:x1].reshape(32,-1)
        gram=phi@phi.T
        ridge=max(float(gram.trace())/32*spec['ridge_relative'],spec['ridge_absolute_floor'])
        result.append(torch.linalg.solve(gram+torch.eye(32,device=p.device,dtype=p.dtype)*ridge,phi).detach())
    cache['_lcf_projection']=result
    return result

class LocalConnectivity(nn.Module):
    def __init__(self,arm,spec):
        super().__init__();self.ssr_arm=arm;self.spec=spec
        self.feature=nn.ModuleList([nn.Conv2d(c,16,1) for c in [256,512,512]])
        self.coefficient=nn.Linear(32,16);self.category=nn.Embedding(80,8)
        self.conv1=nn.Conv2d(108,32,3,padding=1)
        self.conv2=nn.Conv2d(32,32,3,padding=1)
        self.conv3=nn.Conv2d(32,32,3,padding=1)
        self.output=nn.Conv2d(32,1,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,cache,features,force_global=False):
        p=cache['proto'].detach().float();c0=cache['c0'].detach().float()
        ph,pw=p.shape[-2:];n=len(c0);baseline=(c0@p.flatten(1)).reshape(n,ph,pw)
        if not n:return baseline+self.output.weight.sum()*0,baseline
        maps=[]
        for level,proj in enumerate(self.feature):
            source=features[level][cache['batch_position']].detach()[None]
            maps.append(F.interpolate(proj(source),size=(ph,pw),mode='bilinear',align_corners=False)[0])
        regional=torch.cat(maps);condition=torch.cat((self.coefficient(c0),self.category(cache['classes'].long()),cache['scores'][:,None]),1)
        boxes=boxes_proto(cache);valid=[i for i,b in enumerate(boxes) if b[2]>b[0] and b[3]>b[1]]
        fields=[None]*n;pad=self.spec['context_padding_proto_pixels']
        for start in range(0,len(valid),4):
            ranks=valid[start:start+4];inputs=[];sizes=[];locations=[]
            for i in ranks:
                x0,y0,x1,y1=boxes[i];a=max(0,x0-pad);b=max(0,y0-pad);c=min(pw,x1+pad);d=min(ph,y1+pad)
                yy,xx=torch.meshgrid(torch.arange(b,d,device=p.device),torch.arange(a,c,device=p.device),indexing='ij')
                coords=torch.stack(((xx-x0)/max(1,x1-x0)*2-1,(yy-y0)/max(1,y1-y0)*2-1)).float()
                inputs.append(torch.cat((regional[:,b:d,a:c],p[:,b:d,a:c],baseline[i:i+1,b:d,a:c],coords,
                                         condition[i,:,None,None].expand(-1,d-b,c-a))))
                sizes.append((d-b,c-a));locations.append((y0-b,x0-a,y1-y0,x1-x0))
            mh=max(v[0] for v in sizes);mw=max(v[1] for v in sizes)
            xx=torch.stack([F.pad(t,(0,mw-t.shape[-1],0,mh-t.shape[-2])) for t in inputs])
            mask=torch.stack([F.pad(p.new_ones((1,h,w)),(0,mw-w,0,mh-h)) for h,w in sizes])
            def block(x,m):
                h=F.gelu(self.conv1(x))*m;h=F.gelu(self.conv2(h))*m;h=F.gelu(self.conv3(h))*m
                return self.output(h)*m
            if self.training and torch.is_grad_enabled():values=checkpoint(block,xx,mask,use_reentrant=False,preserve_rng_state=False)
            else:values=block(xx,mask)
            for j,i in enumerate(ranks):
                y,x,h,w=locations[j];fields[i]=values[j,0,y:y+h,x:x+w]
        projected=self.ssr_arm in ('A','B') or force_global
        if projected:
            matrices=projection_matrices(cache,self.spec);deltas=[]
            for i in range(n):
                deltas.append((matrices[i]@fields[i].flatten().double()).float() if fields[i] is not None else c0[i]*0)
            delta_c=torch.stack(deltas);self.last_delta_c=delta_c.detach()
            coefficients=c0+delta_c
            changed=(coefficients@p.flatten(1)).reshape_as(baseline)
        else:
            maps=[]
            for i,(x0,y0,x1,y1) in enumerate(boxes):
                maps.append(F.pad(fields[i],(x0,pw-x1,y0,ph-y1)) if fields[i] is not None else baseline[i]*0)
            changed=baseline+torch.stack(maps)
        return changed+self.output.weight.sum()*0,baseline

def make_model(arm,spec):
    torch.manual_seed(spec['seed']);return LocalConnectivity(arm,spec)

def upsample(low,shape):
    from ultralytics.utils import ops
    if not len(low):return low.new_empty((0,*shape))
    step=max(1,32_000_000//math.prod(shape))
    return torch.cat([ops.scale_masks(low[a:a+step][None],shape)[0] for a in range(0,len(low),step)])

def decode_low(cache,low):
    from ultralytics.utils import ops
    grid=upsample(low,cache['native_decode_hw'])
    first=ops.crop_mask((grid>0).byte(),cache['boxes_input'])
    return ops.scale_masks(first[None],cache['original_hw'],ratio_pad=cache['ratio_pad'])[0].byte()

def input_roi(cache):
    h,w=cache['native_decode_hw'];b=cache['boxes_input']
    xx=torch.arange(w,device=b.device)[None,None,:];yy=torch.arange(h,device=b.device)[None,:,None]
    return (xx>=b[:,0,None,None])&(xx<b[:,2,None,None])&(yy>=b[:,1,None,None])&(yy<b[:,3,None,None])
