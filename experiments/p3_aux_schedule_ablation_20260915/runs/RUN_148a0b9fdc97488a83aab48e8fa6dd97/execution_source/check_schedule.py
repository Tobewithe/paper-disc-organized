"""Verify installed schedule, first-epoch equivalence and auxiliary gradients."""
import argparse,json,math
from pathlib import Path
from types import SimpleNamespace
import torch
from ultralytics.utils.loss import E2ELoss
import method_core as core
from recording import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    schedule=E2ELoss(None,loss_fn=lambda *args,**kw:SimpleNamespace(hyp=SimpleNamespace(epochs=3)))
    x=torch.tensor(2.,requires_grad=True)
    class Criterion:
        o2m_copy=.8
        o2m=.8
        def __call__(self,preds,batch):
            base=torch.stack([3*x.square(),x.square()]);return base,base.detach()
    model=SimpleNamespace(criterion=Criterion())
    batch={'img':torch.zeros(2,1),'_cf_teacher_guidance':[1]}
    core.METHOD_ENABLED=True;core.CF_TEACHER=object();core.AUX_WEIGHT=.5
    core.counterfactual_aux=lambda *args:x.square()
    records=[]
    for epoch,expected in enumerate([.5,.28125,.0625],1):
        model.criterion.o2m=schedule.o2m
        core.FOLLOW_O2M=True
        weight=core.effective_aux_weight(model.criterion)
        assert math.isclose(weight,expected,abs_tol=1e-12)
        loss,_=core.patched_loss(model,batch,preds=object())
        gradient=torch.autograd.grad(loss.sum(),x)[0]
        assert torch.isclose(gradient,torch.tensor(16.+8.*weight))
        if epoch==1:
            core.FOLLOW_O2M=False
            fixed,_=core.patched_loss(model,batch,preds=object())
            fixed_gradient=torch.autograd.grad(fixed.sum(),x)[0]
            assert torch.equal(loss,fixed) and torch.equal(gradient,fixed_gradient)
        records.append(dict(epoch=epoch,official_o2m=schedule.o2m,aux_weight=weight,gradient=float(gradient)))
        schedule.update()
    core.AUX_WEIGHT=0;core.FOLLOW_O2M=True
    loss,_=core.patched_loss(model,batch,preds=object());base,_=model.criterion(None,batch)
    assert torch.equal(loss,base)
    assert torch.equal(torch.autograd.grad(loss.sum(),x)[0],torch.autograd.grad(base.sum(),x)[0])
    atomic_json(a.out/'COMPLETE.json',dict(status='complete',schedule=records,first_epoch_loss_and_gradient_equal=True,zero_weight_base_loss_and_gradient_equal=True,scope='actual injection function and installed E2ELoss schedule; full model smoke follows'))
    print(json.dumps(records),flush=True)

if __name__=='__main__':main()
