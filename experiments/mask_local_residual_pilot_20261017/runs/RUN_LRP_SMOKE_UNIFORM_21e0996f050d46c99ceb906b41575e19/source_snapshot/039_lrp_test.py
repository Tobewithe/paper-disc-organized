from lrp_common import *
import numpy as np,torch
from lrp_math import PixelResidual,objective,sample_indices,deploy

def main():
    inspect_inputs()
    with Run('TESTS',budget=900) as run:
        run.freeze(dict(stage='synthetic_engineering_tests',scientific_data=False))
        torch.manual_seed(0); model=PixelResidual().cuda()
        require(sum(p.numel() for p in model.parameters())==3297,'architecture parameter count')
        x=torch.randn(7,68,device='cuda'); a=torch.tensor([-.8,-.1,1e-6,.3,1.,-1.5,1.5],device='cuda')
        rec=dict(x=x,a=a,annotation_id=1,target=1-(a>0).float())
        require(torch.equal(model(x),torch.zeros(7,device='cuda')),'initial not zero')
        for arm in ('LOCAL','UNIFORM'):
            model.zero_grad(set_to_none=True); value,info=objective(model,[rec],arm); value.backward()
            require(torch.count_nonzero(model.net[-1].bias.grad)>0,'zero initialization has no gradient')
            analytic=float(model.net[-1].bias.grad)
            with torch.no_grad(): model.net[-1].bias.fill_(.001)
            plus=float(objective(model,[rec],arm)[0].detach())
            with torch.no_grad(): model.net[-1].bias.fill_(-.001)
            minus=float(objective(model,[rec],arm)[0].detach())
            require(abs((plus-minus)/.002-analytic)<1e-4,'finite-difference gradient mismatch')
            with torch.no_grad(): model.net[-1].bias.zero_()
        native=np.array([[True,False,True],[False,True,False]])
        r=dict(native=native,band=np.array([[True,False,False],[True,False,False]]),yy=np.array([0,1]),xx=np.array([0,0]),x=x[:2],a=torch.tensor([1e-6,-1e-6],device='cuda'),sample=np.array([0,1]))
        for arm in ('LOCAL','UNIFORM'): require(np.array_equal(deploy(model,r,arm),native),'zero-delta final composition')
        require(np.array_equal(sample_indices(1200,12,3),sample_indices(1200,12,3)),'nondeterministic support sampling')
        require(len(set(sample_indices(1200,12,3)))==512,'sampling not without replacement')
        zero,_=objective(model,[],'LOCAL'); zero.backward()
        require(torch.isfinite(zero),'empty image failure')
        # Verify that constant local outputs are exactly the corresponding restricted output.
        with torch.no_grad(): model.net[-1].bias.fill_(.4)
        require(np.array_equal(deploy(model,r,'LOCAL'),deploy(model,r,'UNIFORM')),'constant intervention control')
        config_audit(run,new_optimizer_steps=0,tests=8)
        run.finish(dict(tests=8,passed=True),scientific='engineering_only')
    save(STUDY/'TESTS_POINTER.json',dict(run_id=run.run_id))

if __name__=='__main__': main()
