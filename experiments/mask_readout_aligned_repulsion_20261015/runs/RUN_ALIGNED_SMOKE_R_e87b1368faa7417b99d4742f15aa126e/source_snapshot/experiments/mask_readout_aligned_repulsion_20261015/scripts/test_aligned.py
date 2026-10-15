import unittest
import torch
from aligned_loss import terms,boxes_roi


class AlignedControls(unittest.TestCase):
    def test_empty_output_keeps_graph_and_zero_loss(self):
        from aligned_loss import backward_image
        cache=dict(proto=torch.ones(32,4,4),c0=torch.zeros(0,32),boxes_input=torch.zeros(0,4),
          input_hw=[16,16],native_decode_hw=[16,16],scores=torch.zeros(0),classes=torch.zeros(0,dtype=torch.long),raw_ids=[])
        coefficients=torch.zeros(0,32,requires_grad=True)
        loss,accounting=backward_image(cache,coefficients,None,{},'REP',4)
        self.assertEqual(loss,0.);self.assertEqual(accounting['matched_supervised_instances'],0)
        self.assertTrue(accounting['no_output_image']);self.assertIsNotNone(coefficients.grad)

    def test_repulsion_foreground_and_no_neighbor(self):
        z=torch.tensor([[[.2,-.4],[.5,-.1]]],requires_grad=True)
        y=torch.tensor([[[1.,0.],[1.,0.]]]);roi=torch.ones_like(y,dtype=torch.bool)
        _,rep=terms(z,y,roi,torch.ones_like(y),roi);rep.sum().backward()
        self.assertTrue(torch.equal(z.grad[y==1],torch.zeros_like(z.grad[y==1])))
        self.assertTrue((z.grad[y==0]>0).all())
        _,zero=terms(z,y,roi,torch.ones_like(y),torch.zeros_like(roi));self.assertEqual(float(zero.detach()),0.)

    def test_chunked_gradient_same_denominator(self):
        torch.manual_seed(0);basis=torch.randn(5,7,9,dtype=torch.float64);weights=torch.randn(3,5,dtype=torch.float64,requires_grad=True)
        y=(torch.rand(3,7,9)>.5).double();roi=boxes_roi(torch.tensor([[0,0,9,7],[2,1,7,5],[1,0,8,6]]),(7,9))
        neighbor=torch.rand_like(y);overlap=roi.clone();overlap[1]=False
        z=torch.einsum('nk,khw->nhw',weights,basis);a,b=terms(z,y,roi,neighbor,overlap)
        ((a+.1*b).mean()+.1*z.square().mean()).backward();reference=weights.grad.clone()
        weights.grad=None;z=torch.einsum('nk,khw->nhw',weights,basis)
        for i in range(3):
            a,b=terms(z[i:i+1],y[i:i+1],roi[i:i+1],neighbor[i:i+1],overlap[i:i+1])
            ((a+.1*b).sum()/3).backward(retain_graph=True)
        (.1*z.square().mean()).backward()
        self.assertTrue(torch.allclose(weights.grad,reference,rtol=1e-12,atol=1e-12))

    def test_half_open_roi(self):
        roi=boxes_roi(torch.tensor([[1.,1.,3.,3.]]),(4,4))
        self.assertEqual(int(roi.sum()),4);self.assertFalse(bool(roi[0,3,3]))

if __name__=='__main__':unittest.main()
