"""Frozen E1-R2: foreground-only rank-one and box-regularized local response fit."""
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/mask_readout_subspace_repulsion_20261015/scripts'))
from repulsion import choose_neighbors

EPS=1e-8
MIN_PIXELS=4
RIDGE_REL=.01
PARAMETERS=dict(box_iou_strict_min=.1,core_side_fraction=.5,minimum_conflict_pixels=4,
    conflict='geometric omega AND zi>0 AND zj>zi',epsilon=EPS,ridge_relative=RIDGE_REL,
    strength=1.,coefficient_origin='native c0',simultaneous_native_neighbors=True)


def update(cache,device='cuda'):
    c=cache['c0'].detach().to(device).float();p=cache['proto'].detach().to(device).float()
    boxes=cache['boxes_input'].detach().to(device).float();n,k=c.shape
    if n==0:return {'A':c.clone(),'B':c.clone()},[]
    neighbors,ious=choose_neighbors(cache['boxes_input'].tolist(),cache['scores'].tolist(),cache['classes'].tolist(),cache['raw_ids'])
    ix=torch.tensor([max(0,j) for j in neighbors],device=device,dtype=torch.long)
    valid=torch.tensor([j>=0 for j in neighbors],device=device)
    center=(boxes[:,:2]+boxes[:,2:])/2;other=boxes[ix];oc=center[ix]
    h,w=p.shape[-2:];ih,iw=cache['input_hw']
    x=torch.arange(w,device=device,dtype=torch.float32)[None,None,:]*(iw/w)
    y=torch.arange(h,device=device,dtype=torch.float32)[None,:,None]*(ih/h)
    lo=torch.maximum(boxes[:,:2],other[:,:2]);hi=torch.minimum(boxes[:,2:],other[:,2:])
    omega=(x>=lo[:,0,None,None])&(x<hi[:,0,None,None])&(y>=lo[:,1,None,None])&(y<hi[:,1,None,None])
    clo=center-(boxes[:,2:]-boxes[:,:2])/4;chi=center+(boxes[:,2:]-boxes[:,:2])/4
    core=(x>=clo[:,0,None,None])&(x<chi[:,0,None,None])&(y>=clo[:,1,None,None])&(y<chi[:,1,None,None])
    omega &= ~core & ((x-oc[:,0,None,None])**2+(y-oc[:,1,None,None])**2 < (x-center[:,0,None,None])**2+(y-center[:,1,None,None])**2) & valid[:,None,None]
    box=(x>=boxes[:,0,None,None])&(x<boxes[:,2,None,None])&(y>=boxes[:,1,None,None])&(y<boxes[:,3,None,None])
    z=(c@p.flatten(1)).reshape(n,h,w);zj=z[ix];old_count=omega.sum((1,2));old_div=old_count.clamp_min(1)
    cross=z*zj;old_num=(cross*omega).sum((1,2))/old_div;old_den=(zj.square()*omega).sum((1,2))/old_div
    old_pp=(cross*(omega&(z>0)&(zj>0))).sum((1,2))/old_div
    old_nn=(cross*(omega&(z<0)&(zj<0))).sum((1,2))/old_div
    old_mixed=(cross*(omega&(cross<0))).sum((1,2))/old_div
    old_applied=valid&(old_count>0)&(old_den>EPS)&(old_num>0)
    old_alpha=torch.where(old_applied,old_num/(old_den+EPS),torch.zeros_like(old_num))
    fg=omega&(z>0)&(zj>z);count=fg.sum((1,2));div=count.clamp_min(1)
    num=(cross*fg).sum((1,2))/div;den=(zj.square()*fg).sum((1,2))/div
    applied=valid&(count>=MIN_PIXELS)&(den>EPS)&(num>0)
    alpha=torch.where(applied,num/(den+EPS),torch.zeros_like(num))
    a=c-alpha[:,None]*c[ix];b=c.clone();extras={}
    for i in applied.nonzero(as_tuple=False).flatten().tolist():
        pp=p[:,box[i]];m=pp.shape[1];g=(pp@pp.T)/m
        ridge=max(EPS,RIDGE_REL*float(g.trace())/k)
        target=torch.where(fg[i][box[i]],-alpha[i]*zj[i][box[i]],0.)
        rhs=(pp@target)/m;matrix=g+ridge*torch.eye(k,device=device,dtype=torch.float32)
        delta=torch.linalg.solve(matrix,rhs);b[i]=c[i]+delta
        residual=float((matrix@delta-rhs).norm()/(1+rhs.norm()))
        # Audit the effective FP32 coefficient change, not just the mathematical solve.
        actual_delta=b[i]-c[i];da=a[i]-c[i]
        objective=lambda d:float(((d@pp-target).square().mean()+ridge*d.square().sum()))
        jb=objective(actual_delta);ja=objective(da);j0=float(target.square().mean())
        if residual>5e-5 or jb>ja+5e-5*(1+ja) or jb>j0+5e-5*(1+j0):
            raise ValueError('regularized solution failed residual/objective audit')
        extras[i]=dict(ridge=ridge,normal_residual=residual,objective_B=jb,objective_A=ja,objective_identity=j0)
    if not all(torch.isfinite(v).all().item() for v in (a,b)):raise ValueError('nonfinite coefficients')
    if not torch.all((alpha>=0)&(alpha<=1+1e-6)).item():raise ValueError('positive-region alpha bound failed')
    assert torch.equal(a[~applied],c[~applied]) and torch.equal(b[~applied],c[~applied])
    newza=(a@p.flatten(1)).reshape_as(z);newzb=(b@p.flatten(1)).reshape_as(z)
    outside=box&~fg;outdiv=outside.sum((1,2)).clamp_min(1)
    arms={}
    for name,cn,zn in [('A',a,newza),('B',b,newzb)]:
        diff=zn-z
        arms[name]=dict(c_new=cn,delta_norm=(cn-c).norm(dim=1),
            nonconflict_logit_rms=((diff.square()*outside).sum((1,2))/outdiv).sqrt(),
            conflict_logit_delta_mean=(diff*fg).sum((1,2))/div,
            proto_background_to_foreground=((z<=0)&(zn>0)&box).sum((1,2)),
            proto_foreground_to_background=((z>0)&(zn<=0)&box).sum((1,2)))
    raw=dict(c0=c,omega_pixels=old_count,conflict_pixels=count,box_pixels=box.sum((1,2)),
        old_numerator=old_num,old_denominator=old_den,old_positive_positive=old_pp,
        old_negative_negative=old_nn,old_opposite_sign=old_mixed,old_applied=old_applied,old_alpha=old_alpha,
        numerator=num,denominator=den,alpha=alpha,applied=applied)
    raw={key:v.cpu().tolist() for key,v in raw.items()}
    arm_lists={name:{key:v.cpu().tolist() for key,v in values.items()} for name,values in arms.items()}
    records=[]
    for i in range(n):
        record={key:values[i] for key,values in raw.items()}
        record.update(output_rank=i,neighbor_rank=neighbors[i],neighbor_box_iou=ious[i],
            **extras.get(i,dict(ridge=0.,normal_residual=0.,objective_B=0.,objective_A=0.,objective_identity=0.)))
        record['arms']={name:{key:v[i] for key,v in values.items()} for name,values in arm_lists.items()}
        records.append(record)
    return {'A':a,'B':b},records
