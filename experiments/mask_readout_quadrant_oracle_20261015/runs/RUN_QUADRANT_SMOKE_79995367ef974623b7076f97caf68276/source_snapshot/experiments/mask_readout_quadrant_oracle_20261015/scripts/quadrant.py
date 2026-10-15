"""Spatial coefficient oracle components; no deployable GT-free claim."""
import torch
from torch.nn import functional as F

SOLVER=dict(penalty=.00075,max_iter=120,max_eval=150,history_size=20,lr=1.,tolerance_grad=1e-9,
    tolerance_change=1e-12,gradient_acceptance=1e-6,line_search_fn='strong_wolfe',dtype='float64',initialization='zero_delta')


def partition(cache,rank,device='cpu',dtype=torch.float64):
    h,w=cache['proto'].shape[-2:];ih,iw=cache['input_hw'];box=cache['boxes_input'][rank].to(device=device,dtype=torch.float32)
    x=torch.arange(w,device=device,dtype=torch.float32)[None,:]*(iw/w);y=torch.arange(h,device=device,dtype=torch.float32)[:,None]*(ih/h)
    cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
    labels=(x>=cx).long()+2*(y>=cy).long();hard=F.one_hot(labels,num_classes=4).permute(2,0,1).to(dtype)
    soft=F.avg_pool2d(F.pad(hard[None],(1,1,1,1),mode='replicate'),3,stride=1)[0]
    assert torch.allclose(soft.sum(0),torch.ones_like(soft[0]),atol=1e-12 if dtype==torch.float64 else 1e-6,rtol=0)
    return hard,soft


def active_quadrants(cache,rank,neighbor):
    if neighbor<0:return []
    hard,_=partition(cache,rank);h,w=cache['proto'].shape[-2:];ih,iw=cache['input_hw']
    x=torch.arange(w,dtype=torch.float64)[None,:]*(iw/w);y=torch.arange(h,dtype=torch.float64)[:,None]*(ih/h)
    b=cache['boxes_input'].double();lo=torch.maximum(b[rank,:2],b[neighbor,:2]);hi=torch.minimum(b[rank,2:],b[neighbor,2:])
    overlap=(x>=lo[0])&(x<hi[0])&(y>=lo[1])&(y<hi[1])
    return [q for q in range(4) if bool((hard[q].bool()&overlap).any())]


def solve(design,target,base_logits):
    p,y,z0=(v.detach().cpu().double() for v in (design,target,base_logits))
    if p.ndim!=2 or y.shape!=(len(p),) or z0.shape!=y.shape or not len(p):raise ValueError('invalid oracle design')
    if p.shape[1]==0:return dict(delta=torch.zeros(0,dtype=torch.float64),solver_status='identity_no_active_quadrant',gradient_inf_norm=0.,iterations=0,function_evals=0)
    d=torch.zeros(p.shape[1],dtype=torch.float64,requires_grad=True)
    def objective():return F.binary_cross_entropy_with_logits(z0+p@d,y)+SOLVER['penalty']*d.square().sum()/2
    opt=torch.optim.LBFGS([d],**{k:SOLVER[k] for k in ('lr','max_iter','max_eval','history_size','tolerance_grad','tolerance_change','line_search_fn')})
    def closure():
        opt.zero_grad();value=objective()
        if not torch.isfinite(value):raise ValueError('nonfinite oracle loss')
        value.backward()
        if not torch.isfinite(d.grad).all():raise ValueError('nonfinite oracle gradient')
        return value
    opt.step(closure);value=objective();grad=torch.autograd.grad(value,d)[0];residual=float(grad.abs().max())
    if not torch.isfinite(d).all():raise ValueError('nonfinite oracle result')
    state=opt.state[d]
    return dict(delta=d.detach(),objective=float(value.detach()),gradient_inf_norm=residual,iterations=int(state['n_iter']),
        function_evals=int(state['func_evals']),solver_status='converged' if residual<=1e-6 else 'budget_exhausted')


def positive_control():
    p=torch.eye(128,dtype=torch.float64);star=torch.tensor([.2,-.2]*64,dtype=torch.float64);z0=torch.zeros(128,dtype=torch.float64)
    y=star.sigmoid()+128*SOLVER['penalty']*star;result=solve(p,y,z0);error=float((result['delta']-star).abs().max())
    return dict(passed=error<=1e-4 and result['solver_status']=='converged',max_error=error,gradient_inf_norm=result['gradient_inf_norm'])


def decode_logits(cache,logits):
    from ultralytics.utils import ops
    h,w=cache['native_decode_hw'];count=len(logits);step=max(1,32_000_000//(h*w));pieces=[]
    for offset in range(0,count,step):
        m=ops.scale_masks(logits[offset:offset+step][None],[h,w])[0].gt(0).byte()
        m=ops.crop_mask(m,cache['boxes_input'][offset:offset+step].to(logits.device))
        pieces.append(ops.scale_masks(m[None],cache['original_hw'],ratio_pad=cache['ratio_pad'])[0].byte())
    return torch.cat(pieces)


def changed_logits(cache,rank,deltas):
    p=cache['proto'].to('cuda').float();c=cache['c0'].to('cuda').float();h,w=p.shape[-2:]
    logits=(c@p.flatten(1)).reshape(len(c),h,w)
    if torch.count_nonzero(deltas):
        _,weights=partition(cache,rank,'cuda',torch.float32)
        response=(deltas.to(p)@p.flatten(1)).reshape(4,h,w)
        logits[rank]+=torch.sum(weights*response,dim=0)
    return logits
