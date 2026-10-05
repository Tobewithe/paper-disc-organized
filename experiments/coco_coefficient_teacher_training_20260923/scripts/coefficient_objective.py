"""Training-only, fixed-prototype coefficient solver and matched direct objective."""
import torch
from torch.nn import functional as F


def objective(z, target, reference, beta=0.5, rho=0.01):
    sign = target.mul(2).sub(1)
    correct = ((reference > 0) == (target > .5)).to(z.dtype)
    margin = (sign * reference).clamp(max=1).detach()
    violation = (margin - sign * z).clamp_min(0) * correct
    bce = F.binary_cross_entropy_with_logits(z, target, reduction='none').mean(-1)
    guard = violation.square().sum(-1) / correct.sum(-1).clamp_min(1)
    trust = (z-reference).square().mean(-1)
    return bce + beta*guard + rho*trust


@torch.no_grad()
def solve(proto, initial, target, reference, steps=5, beta=.5, rho=.01, damping=.01):
    """Batched damped Newton with backtracking; tensors [n,points,32], [n,32]."""
    p, c, y, ref = proto.float(), initial.float().clone(), target.float(), reference.float()
    scale = p.square().mean(1).sqrt().clamp_min(.01)
    p = p / scale[:, None]
    c = c * scale
    z = torch.einsum('nsk,nk->ns', p, c)
    before = objective(z, y, ref, beta, rho)
    sign = 2*y-1
    correct = ((ref>0)==(y>.5)).float()
    margin = (sign*ref).clamp(max=1)
    denom = correct.sum(-1,keepdim=True).clamp_min(1)
    eye = torch.eye(p.shape[-1],device=p.device)[None]
    for _ in range(steps):
        prob = z.sigmoid()
        violation = (margin-sign*z).clamp_min(0)*correct
        grad_z = (prob-y)/p.shape[1] - 2*beta*sign*violation/denom + 2*rho*(z-ref)/p.shape[1]
        curv = (prob*(1-prob)+2*rho)/p.shape[1] + 2*beta*correct*(violation>0)/denom
        grad = torch.einsum('nsk,ns->nk',p,grad_z)
        hess = p.transpose(1,2) @ (p*curv[:,:,None])
        hess = hess + damping*eye
        direction = torch.linalg.solve(hess,grad[:,:,None])[:,:,0]
        old = objective(z,y,ref,beta,rho)
        chosen_c, chosen_z, chosen_f = c.clone(), z.clone(), old.clone()
        accepted = torch.zeros(len(c),device=c.device,dtype=torch.bool)
        for step_size in (1., .5, .25, .125, .0625):
            proposal = c-step_size*direction
            proposed_z = torch.einsum('nsk,nk->ns',p,proposal)
            f = objective(proposed_z,y,ref,beta,rho)
            good = (~accepted) & torch.isfinite(f) & (f < old)
            chosen_c[good], chosen_z[good], chosen_f[good] = proposal[good], proposed_z[good], f[good]
            accepted |= good
        c,z = chosen_c,chosen_z
    after = objective(z,y,ref,beta,rho)
    ok = torch.isfinite(after) & (after <= before+1e-6)
    result = c/scale
    result[~ok] = initial.float()[~ok]
    return result.detach(), {'before':before, 'after':after, 'accepted':ok & (after<before-1e-6)}


def self_test():
    torch.manual_seed(9107)
    p=torch.randn(3,128,32)
    c=torch.randn(3,32)*.1
    y=(torch.randn(3,128)>0).float()
    ref=torch.einsum('nsk,nk->ns',p,c)
    fitted,diag=solve(p,c,y,ref)
    assert (diag['after']<=diag['before']+1e-6).all()
    assert diag['accepted'].all()
    z=(ref+torch.randn_like(ref)*.5).requires_grad_()
    total=objective(z,y,ref).sum()
    g=torch.autograd.grad(total,z)[0]
    bcegrad=(z.detach().sigmoid()-y)/z.shape[1]
    assert (g-bcegrad).abs().max()>1e-5, 'reference protection must actually affect gradient'
    student=c.clone().requires_grad_()
    target_z=torch.einsum('nsk,nk->ns',p,fitted).detach()
    predicted_z=torch.einsum('nsk,nk->ns',p,student)
    loss=F.binary_cross_entropy_with_logits(predicted_z,target_z.sigmoid())
    loss.backward()
    assert torch.isfinite(student.grad).all() and student.grad.abs().sum()>0
    return {'solver_improves_all':True,'reference_penalty_nonzero_gradient':True,
            'student_receives_gradient':True,'inner_before':diag['before'].tolist(),'inner_after':diag['after'].tolist()}
