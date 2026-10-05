"""Synthetic algebra/gradient checks ONLY; never COCO performance evidence."""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from readout_input_probe import Readout, inputs, sample_regions, objective, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.manual_seed(17)
    dtype = torch.float64
    h = torch.randn(4, 8, dtype=dtype)
    region = torch.randn(4, 9, 8, dtype=dtype)
    level = torch.tensor([0, 1, 2, 0])
    norm = (h.mean(0), h.std(0).clamp_min(.01))
    keys = [(20, k) for k in range(4)]
    xx = {mode:inputs(h, region, level, mode, keys, 0, 0, norm)
          for mode in ['own', 'mean', 'ordered', 'shuffled']}
    expected = ((region-norm[0])/norm[1]).sort(dim=1).values
    shuffled = xx['shuffled'][:, 8:80].reshape(4, 9, 8)
    torch.testing.assert_close(shuffled.sort(dim=1).values, expected)
    assert not torch.equal(shuffled, ((region-norm[0])/norm[1]))
    assert not torch.equal(xx['shuffled'], inputs(h,region,level,'shuffled',keys,0,1,norm))
    assert torch.equal(xx['own'][:,8:80].reshape(4,9,8), ((h-norm[0])/norm[1])[:,None].expand(-1,9,-1))
    counts = []
    for mode, x in xx.items():
        torch.manual_seed(4)
        model = Readout(8, dtype=dtype)
        counts.append(sum(p.numel() for p in model.parameters()))
        assert not torch.count_nonzero(model(x,level))
    assert len(set(counts)) == 1
    # An affine coordinate ramp must be sampled at feature-cell centers exactly.
    yy, xx0 = torch.meshgrid(torch.arange(8, dtype=dtype)+.5, torch.arange(8,dtype=dtype)+.5,indexing='ij')
    fmap = torch.stack([xx0,yy])[None]
    box = torch.tensor([[1.,1.,7.,7.]],dtype=dtype)
    sampled = sample_regions([fmap,fmap,fmap],torch.tensor([0]),box,(8,8))[0]
    axis = torch.tensor([1.6,4.,6.4],dtype=dtype)
    ey, ex = torch.meshgrid(axis,axis,indexing='ij')
    torch.testing.assert_close(sampled,torch.stack([ex,ey],-1).reshape(9,2),rtol=1e-12,atol=1e-12)
    model = Readout(8,linear=True,dtype=dtype)
    p = torch.randn(4,21,32,dtype=dtype)
    y = torch.randint(0,2,(4,21)).to(dtype)
    c = torch.randn(4,32,dtype=dtype)
    fac = torch.tensor([.9,1.,1.1,.8],dtype=dtype)
    x = (h-norm[0])/norm[1]
    loss = objective(model(x,level),c,p,y,fac)
    loss.backward()
    analytic = float(model.weight.grad[0,0,0])
    eps=1e-5
    with torch.no_grad():
        model.weight[0,0,0] = eps
        plus=float(objective(model(x,level),c,p,y,fac))
        model.weight[0,0,0] = -eps
        minus=float(objective(model(x,level),c,p,y,fac))
        model.weight[0,0,0] = 0.
    error=abs((plus-minus)/(2*eps)-analytic)
    assert error<1e-8,error
    optimizer=torch.optim.SGD(model.parameters(),lr=.001)
    optimizer.step()
    after=float(objective(model(x,level),c,p,y,fac).detach())
    assert after<float(loss.detach())
    write_json(args.out,dict(status='PASS',evaluation_type='synthetic_algebra_only',real_COCO_evaluation=False,
         checks=['equal_parameter_counts','zero_residual','shuffle_preserves_values','shuffle_changes_each_draw',
                 'same_h_control','feature_coordinate_ramp','finite_difference','small_gradient_step'],
         nonlinear_parameter_counts=counts,finite_difference_error=error,
         note='Local pytorch CPU numerical witness. Does not execute pinned remote Ultralytics data pipeline or train a real model.'))
    print(args.out.read_text())


if __name__=='__main__':
    main()
