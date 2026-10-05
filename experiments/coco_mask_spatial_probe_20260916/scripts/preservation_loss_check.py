"""Check meaningful behavior of the training-only correctness penalty."""
import argparse
from pathlib import Path
import json
import torch
from component_seed_probe import correctness_guard


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    b=torch.tensor([[[2.,-2.,-2.,2.]]]);g=torch.tensor([[[1.,0.,1.,0.]]])
    assert correctness_guard(b,b,g).item()==0
    z=torch.tensor([[[-.2,.2,4.,-4.]]],requires_grad=True)
    loss=correctness_guard(z,b,g).sum();loss.backward()
    assert loss.item()>0 and z.grad[0,0,0]<0 and z.grad[0,0,1]>0
    assert z.grad[0,0,2]==0 and z.grad[0,0,3]==0
    # Restoring target/background correctly beyond the retained margin is not penalized.
    assert correctness_guard(torch.tensor([[[1.5,-1.5,8.,-8.]]]),b,g).item()==0
    wrong=-b;assert torch.isfinite(correctness_guard(wrong,wrong,1-g)).all()
    no_correct=correctness_guard(-b,-b,g);assert torch.isfinite(no_correct).all()
    a.out.mkdir(exist_ok=True,parents=True)
    (a.out/'COMPLETE.json').write_text(json.dumps({'baseline_identity_zero':True,'opposite_sign_gradients':True,
        'original_errors_unconstrained':True,'margin_improvement_unpenalized':True,'finite_empty_regions':True}))


if __name__=='__main__':main()
