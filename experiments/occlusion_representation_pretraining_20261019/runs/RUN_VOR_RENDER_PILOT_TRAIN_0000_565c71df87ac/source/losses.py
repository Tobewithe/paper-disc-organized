import torch
from torch.nn import functional as F


def masked_mean(values, mask):
    weights = mask.to(values.dtype)
    return (values*weights).sum()/weights.sum().clamp_min(1)


def soft_labels(xy, visible, height=20, width=32):
    # Continuous edge-origin coordinates -> cell-centred bilinear labels.
    u = xy*torch.tensor([width,height],device=xy.device)-.5
    lower = u.floor()
    frac = u-lower
    labels = torch.zeros(*xy.shape[:-1],height*width+1,device=xy.device)
    for ox,oy in ((0,0),(1,0),(0,1),(1,1)):
        grid = lower+torch.tensor([ox,oy],device=xy.device)
        idx = (grid[...,1].clamp(0,height-1)*width+grid[...,0].clamp(0,width-1)).long()
        weight = (frac[...,0] if ox else 1-frac[...,0])*(frac[...,1] if oy else 1-frac[...,1])
        labels.scatter_add_(-1,idx[...,None],(weight*visible)[...,None])
    labels[...,-1] = (~visible).to(labels.dtype)
    return labels


def base_losses(pred, targets, arm):
    # Query existence is decided at initial observation, not on future return.
    points = targets["xy"].shape[2]
    active = targets["active"][:,None,:].expand_as(targets["state"])
    valid = targets["valid"] & active
    visible = targets["state"] == 0
    logp = pred["logits"][:,:,:points].log_softmax(-1)
    labels = soft_labels(targets["xy"],visible)
    corr = masked_mean(-(labels*logp).sum(-1),valid)
    xy_error = F.smooth_l1_loss(pred["xy_mean"][:,:,:points],targets["xy"],
                               beta=.02,reduction="none").sum(-1)
    corr = corr+masked_mean(xy_error,valid & visible)
    state = F.cross_entropy(pred["state"][:,:,:points].reshape(-1,4),
                            targets["state"].reshape(-1),reduction="none").reshape_as(visible)
    evidence = masked_mean(state,valid)
    region = pred["region"][:,:,points:]
    region_valid = targets["region_active"][:,None,:].expand_as(targets["inframe"])
    evidence += masked_mean(F.binary_cross_entropy_with_logits(
        region[...,0],targets["inframe"].float(),reduction="none"),region_valid)
    evidence += masked_mean(F.smooth_l1_loss(
        region[...,1].sigmoid(),targets["ratio"].clamp_min(0),
        reduction="none",beta=.1),region_valid & targets["inframe"])
    predicted_xy = pred["xy"][:,:,:points].detach()
    distance = ((predicted_xy-targets["xy"])*256).norm(dim=-1)
    # Owner checks use the raster at the proposed coordinate; point state itself is continuous.
    ix = (predicted_xy[...,0]*128).long().clamp(0,127)
    iy = (predicted_xy[...,1]*80).long().clamp(0,79)
    owner_at = targets["owner_maps"].flatten(-2).gather(-1,iy*128+ix)
    correct = (distance <= 4) & visible & (owner_at == targets["owners"][:,None,:])
    calibration = masked_mean((pred["confidence"][:,:,:points].sigmoid()-correct.float()).square(),valid)
    hidden = masked_mean(xy_error,valid & ~visible & (targets["state"] != 3))
    total = corr+.5*evidence+.1*calibration
    if arm == "Hpos":
        total = total+.2*hidden
    return total, dict(corr=corr,evidence=evidence,cal=calibration,hidden=hidden)


def paired_content_loss(pred, targets, arm, temperature=.07):
    """Four views [a0,a1,b0,b1], same visible support & negatives in B and P.

    Same-owner surface points are excluded as negatives: repeated texture is
    not known distinguishable. Different initial visible instances are used.
    """
    q = targets["xy"].shape[2]
    assert pred["z"].shape[0] == 4
    z = pred["z"][:,:,:q]
    jointly = ((targets["state"] == 0) & targets["valid"] &
               targets["active"][:,None] & (targets["purity"] >= .8)).all(0)
    owner = targets["owners"][0]
    pairs = [(0,1),(1,0),(2,3),(3,2)] if arm == "B" else [(0,3),(3,0),(2,1),(1,2)]
    result = z.sum()*0
    for i,j in pairs:
        similarity = torch.einsum("tqd,tkd->tqk",z[i],z[j])/temperature
        same = torch.eye(q,device=z.device,dtype=torch.bool)[None]
        negatives = owner[:,None] != owner[None,:]
        allowed = (same | negatives[None]) & jointly[:,None,:]
        allowed = allowed | same  # finite masked logsumexp even without support
        logp = similarity.masked_fill(~allowed,-1e4).log_softmax(-1)
        informative = jointly & ((negatives[None] & jointly[:,None,:]).sum(-1) > 0)
        result += masked_mean(-logp.diagonal(dim1=-2,dim2=-1),informative)
    return result/len(pairs), int(jointly.sum().item())
