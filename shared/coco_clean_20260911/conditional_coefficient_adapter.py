"""A trainable coefficient residual conditioned on the current prototype geometry.

c = c0 + T(G) (W_level h + b_level), residual weights initially zero.
T uses only prototype samples inside a predicted box, never ground truth.
Full and diagonal inverse-root transforms have equal average output energy.
"""
import torch

def transform(samples,mode):
    n,_,k=samples.shape
    G=torch.einsum('nsk,nsl->nkl',samples.double(),samples.double())/samples.shape[1]
    scale=G.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-10)
    eye=torch.eye(k,device=samples.device,dtype=torch.float64).expand(n,k,k)
    if mode=='plain':T=eye
    elif mode=='diagonal':
        T=torch.diag_embed((G.diagonal(dim1=-2,dim2=-1)/scale[:,None]+.01).rsqrt())
    elif mode=='conditional':
        eig,Q=torch.linalg.eigh(G/scale[:,None,None]+.01*eye)
        assert eig.min()>0
        T=(Q*eig.rsqrt()[:,None,:])@Q.transpose(-1,-2)
    else:raise ValueError(mode)
    before=G.diagonal(dim1=-2,dim2=-1).sum(-1)
    after=torch.einsum('nki,nij,nkj->n',T,G,T).clamp_min(1e-10)
    if mode!='plain':T=T*(before/after).sqrt()[:,None,None]
    assert torch.isfinite(T).all()
    return T.float()

def predict(x,levels,initial,residual,T):
    base=torch.einsum('nki,ni->nk',initial['weight'][levels],x)+initial['bias'][levels]
    r=torch.einsum('nki,ni->nk',residual['weight'][levels],x)+residual['bias'][levels]
    return base+torch.einsum('nki,ni->nk',T,r)
