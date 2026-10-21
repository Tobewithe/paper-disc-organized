"""GT-free local residual construction and separately supplied training labels."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pycocotools import mask as masks
from native_geometry import native_stages
from lea_ops import bbox_support,encode

class PixelResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(68,32),nn.GELU(),nn.Linear(32,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        return 2*torch.tanh(self.net(x).squeeze(-1)/2)

def sample_indices(count,image_id,rank):
    seed=int.from_bytes(hashlib.sha256(f'lrp-v1|image_id={image_id}|rank={rank}'.encode('ascii')).digest()[:8],'little')
    return np.sort(np.random.Generator(np.random.PCG64(seed)).choice(count,min(512,count),replace=False)) if count>512 else np.arange(count)

def image_context(cache,rows):
    from ultralytics.utils import ops
    ranks=[j for j,r in enumerate(rows) if int(masks.area(r['segmentation']))>=9216]
    first,final,selected=native_stages(cache,cache['c0'],ranks)
    if not all(encode(final[j].numpy())==r['segmentation'] for j,r in enumerate(rows)):
        raise ValueError('native zero-delta RLE drift')
    del first,final
    with torch.inference_mode():
        lifted=ops.scale_masks(cache['proto'].cuda()[None],cache['native_decode_hw'])
        basis=ops.scale_masks(lifted,cache['original_hw'],ratio_pad=cache['ratio_pad'])[0]
    return ranks,selected,basis

def points(cache,row,rank,z_input,basis):
    """No GT accepted. All returned spatial records are defined by frozen native output."""
    from ultralytics.utils import ops
    with torch.inference_mode():
        z=ops.scale_masks(z_input.cuda()[None,None],cache['original_hw'],ratio_pad=cache['ratio_pad'])[0,0]
    n=masks.decode(row['segmentation']).astype(bool)
    roi=bbox_support(n.shape,row['bbox'])
    band=(np.abs(z.cpu().numpy())<=1.5)&roi
    yy,xx=np.nonzero(band)
    yi=torch.as_tensor(yy,device='cuda'); xi=torch.as_tensor(xx,device='cuda')
    p=basis[:,yi,xi].T
    cz=p*cache['c0'][rank].cuda()[None]
    zs=z[yi,xi]
    ns=torch.as_tensor(n[yy,xx],device='cuda',dtype=torch.float32)
    left,top,w,h=map(float,row['bbox'])
    xy=torch.stack(((xi.float()-left)/max(w,1e-6)*2-1,(yi.float()-top)/max(h,1e-6)*2-1),1)
    x=torch.cat((p,cz,zs[:,None],ns[:,None],xy),1).detach()
    a=((2*ns-1)*zs.abs().clamp_min(1e-6)).detach()
    if not torch.isfinite(x).all() or not torch.isfinite(a).all(): raise ValueError('nonfinite native feature')
    if not np.array_equal((a.cpu().numpy()>0),n[yy,xx]): raise ValueError('anchor zero replay differs')
    return dict(x=x,a=a,yy=yy,xx=xx,band=band,native=n,sample=sample_indices(len(yy),cache['image_id'],rank))

def objective(model,instances,arm):
    zero=sum(p.sum()*0 for p in model.parameters())
    gt_terms=[]; preserve=[]
    for rec in instances:
        x,a=rec['x'].cuda(),rec['a'].cuda()
        if not len(x): continue
        d=model(x)
        if arm=='UNIFORM': d=d.mean().expand_as(d)
        z=a+d
        preserve.append(F.binary_cross_entropy_with_logits(z,a.sigmoid()))
        if rec['annotation_id'] is not None:
            gt_terms.append(F.binary_cross_entropy_with_logits(z,rec['target'].cuda()))
    g=torch.stack(gt_terms).mean() if gt_terms else zero
    p=torch.stack(preserve).mean() if preserve else zero
    return g+.1*p,dict(gt_loss=float(g.detach()),preserve_loss=float(p.detach()),gt_instances=len(gt_terms),editable_instances=len(preserve))

def deploy(model,record,arm):
    out=record['native'].copy()
    x,a=record['x'],record['a']
    if not len(x): return out
    with torch.inference_mode():
        if arm=='UNIFORM':
            d=model(x[record['sample']]).mean()
            value=(a+d)>0
        else:
            value=torch.cat([(a[start:start+65536]+model(x[start:start+65536]))>0 for start in range(0,len(x),65536)])
    out[record['yy'],record['xx']]=value.cpu().numpy()
    if np.any((out!=record['native'])&~record['band']): raise ValueError('outside-region edit')
    return out
