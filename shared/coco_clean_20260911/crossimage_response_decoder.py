"""Prediction-only frozen response features and linear readout decoder."""
import torch
import torch.nn.functional as F
from ultralytics.utils import ops


@torch.inference_mode()
def prepare(item,expansion=0.):
    p=torch.as_tensor(item['proto'],device='cuda').float();c=torch.as_tensor(item['coeff'],device='cuda').float()
    boxes=torch.as_tensor(item['boxes'],device='cuda').float();det=torch.as_tensor(item['detections'],device='cuda').float()
    shape=tuple(map(int,item['input_shape']));n=len(c)
    if n:
        z=F.interpolate((c@p.flatten(1)).reshape(1,n,*p.shape[-2:]),shape,mode='bilinear',align_corners=False)[0]
    else:z=torch.zeros((0,*shape),device='cuda')
    assert torch.isfinite(z).all()
    wh=(boxes[:,2:]-boxes[:,:2]).clamp_min(0);area=wh.prod(1)
    inter=(torch.minimum(boxes[:,None,2:],boxes[None,:,2:])-torch.maximum(boxes[:,None,:2],boxes[None,:,:2])).clamp_min(0).prod(2)
    iou=inter/(area[:,None]+area[None,:]-inter).clamp_min(1e-9)
    allowed=(det[:,None,5]==det[None,:,5])&(det[:,None,4]>=.1)&(det[None,:,4]>=.1)&(iou>.05)
    allowed.fill_diagonal_(False)
    neighbors=[]
    for j in range(n):
        ids=allowed[j].nonzero().flatten();neighbors.append(ids[torch.argsort(iou[j,ids],descending=True,stable=True)[:3]])
    crop=boxes+torch.cat([-wh,wh],1)*expansion
    support=ops.crop_mask(torch.ones_like(z,dtype=torch.uint8),crop).bool() if n else z.bool()
    initial=ops.crop_mask((z>0).to(torch.uint8),crop) if n else z.to(torch.uint8)
    if expansion==0 and n:
        official=ops.process_mask(p,c,boxes,shape,upsample=True)
        assert torch.equal(initial,official),'Original official decoder mismatch'
    return dict(z=z,boxes=boxes,crop=crop,support=support,neighbors=neighbors,initial=initial)


def features(prepared,j,iid,seed=0,shuffle=False):
    own=prepared['z'][j];ids=prepared['neighbors'][j];support=prepared['support'][j]
    other=torch.zeros_like(own);exists=torch.zeros_like(support)
    if len(ids):
        valid=prepared['support'][ids]&support[None]
        exists=valid.any(0)
        other=prepared['z'][ids].masked_fill(~valid,-float('inf')).amax(0)
        other=torch.where(exists,other,torch.zeros_like(other))
        if shuffle:
            positions=exists.flatten().nonzero().flatten()
            generator=torch.Generator(device=own.device);generator.manual_seed(20260911+iid*1009+j*131+seed*7907)
            permutation=torch.randperm(len(positions),generator=generator,device=own.device)
            flat=other.flatten();flat[positions]=flat[positions[permutation]].clone()
    return torch.stack([own/5,other/5,exists.float()],-1)


def score(x,model):
    weights=torch.tensor(model['raw_weights'],device=x.device,dtype=x.dtype)
    result=x@weights+model['raw_intercept']
    return result/model['own_logit_slope']


@torch.inference_mode()
def decode(prepared,iid,models,thresholds):
    result={'initial':prepared['initial']}
    records=[]
    for arm,model in models.items():
        binary=prepared['initial'].clone();changed=0;eligible=0
        for j,ids in enumerate(prepared['neighbors']):
            if not len(ids):continue
            eligible+=1;x=features(prepared,j,iid,model['seed'],model['kind']=='sham')
            predicted=(score(x,model)>thresholds[arm])&prepared['support'][j]
            changed+=int((predicted!=binary[j].bool()).sum());binary[j]=predicted.to(torch.uint8)
        result[arm]=binary
        records.append(dict(arm=arm,image_id=iid,eligible_predictions=eligible,changed_input_pixels=changed,
            original_nonempty=int(prepared['initial'].flatten(1).any(1).sum()),nonempty=int(binary.flatten(1).any(1).sum())))
    return result,records
