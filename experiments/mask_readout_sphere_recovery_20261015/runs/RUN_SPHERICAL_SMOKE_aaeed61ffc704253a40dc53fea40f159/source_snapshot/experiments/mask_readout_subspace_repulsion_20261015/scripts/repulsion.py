"""Frozen E1-R1-v1: GT-free native-coefficient local Gram repulsion."""
import numpy as np
import torch

EPS=1e-8
IOU_MIN=.1


def choose_neighbors(boxes,scores,classes,raw_ids):
    b=np.asarray(boxes,dtype=np.float64);scores=np.asarray(scores,dtype=np.float64)
    area=np.maximum(b[:,2]-b[:,0],0)*np.maximum(b[:,3]-b[:,1],0)
    lo=np.maximum(b[:,None,:2],b[None,:,:2]);hi=np.minimum(b[:,None,2:],b[None,:,2:])
    wh=np.maximum(hi-lo,0);inter=wh[:,:,0]*wh[:,:,1];union=area[:,None]+area[None,:]-inter
    ious=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
    result=[];values=[]
    for i in range(len(b)):
        candidates=[j for j in range(len(b)) if classes[j]==classes[i] and raw_ids[j]!=raw_ids[i]
                    and (-scores[j],j)<(-scores[i],i) and ious[i,j]>IOU_MIN]
        j=min(candidates,key=lambda j:(-ious[i,j],-scores[j],j)) if candidates else -1
        result.append(j);values.append(float(ious[i,j]) if j>=0 else 0.)
    return result,values


def update_coefficients(cache,device='cuda'):
    c=cache['c0'].detach().to(device).float();p=cache['proto'].detach().to(device).float()
    boxes=cache['boxes_input'].detach().to(device).float();n=len(c)
    if n==0:return c,[]
    neighbors,ious=choose_neighbors(cache['boxes_input'].tolist(),cache['scores'].tolist(),cache['classes'].tolist(),cache['raw_ids'])
    index=torch.tensor([max(0,j) for j in neighbors],device=device,dtype=torch.long)
    valid=torch.tensor([j>=0 for j in neighbors],device=device)
    other=boxes[index];centers=(boxes[:,:2]+boxes[:,2:])/2;other_centers=centers[index]
    h,w=p.shape[-2:];ih,iw=cache['input_hw']
    x=torch.arange(w,device=device,dtype=torch.float32)[None,None,:]*(iw/w)
    y=torch.arange(h,device=device,dtype=torch.float32)[None,:,None]*(ih/h)
    lo=torch.maximum(boxes[:,:2],other[:,:2]);hi=torch.minimum(boxes[:,2:],other[:,2:])
    omega=(x>=lo[:,0,None,None])&(x<hi[:,0,None,None])&(y>=lo[:,1,None,None])&(y<hi[:,1,None,None])
    core_lo=centers-(boxes[:,2:]-boxes[:,:2])/4;core_hi=centers+(boxes[:,2:]-boxes[:,:2])/4
    core=(x>=core_lo[:,0,None,None])&(x<core_hi[:,0,None,None])&(y>=core_lo[:,1,None,None])&(y<core_hi[:,1,None,None])
    own_distance=(x-centers[:,0,None,None])**2+(y-centers[:,1,None,None])**2
    other_distance=(x-other_centers[:,0,None,None])**2+(y-other_centers[:,1,None,None])**2
    omega &= ~core & (other_distance<own_distance) & valid[:,None,None]
    z=(c@p.flatten(1)).reshape(n,h,w);zj=z[index];count=omega.sum((1,2))
    numerator=(z*zj*omega).sum((1,2))/count.clamp_min(1)
    denominator=(zj.square()*omega).sum((1,2))/count.clamp_min(1)
    applied=valid&(count>0)&(denominator>EPS)&(numerator>0)
    alpha=torch.where(applied,numerator/(denominator+EPS),torch.zeros_like(numerator))
    result=c-alpha[:,None]*c[index]
    if not torch.isfinite(result).all().item():raise ValueError('nonfinite coefficients')
    assert torch.equal(result[~applied],c[~applied])
    newz=(result@p.flatten(1)).reshape_as(z)
    correlation_after=(newz*zj*omega).sum((1,2))/count.clamp_min(1)
    data=[v.detach().cpu().tolist() for v in (count,numerator,denominator,alpha,applied,correlation_after,c,result)]
    records=[dict(output_rank=i,neighbor_rank=neighbors[i],neighbor_box_iou=ious[i],omega_pixels=data[0][i],
        numerator=data[1][i],denominator=data[2][i],alpha=data[3][i],applied=data[4][i],
        correlation_after=data[5][i],c0=data[6][i],c_new=data[7][i]) for i in range(n)]
    return result,records
