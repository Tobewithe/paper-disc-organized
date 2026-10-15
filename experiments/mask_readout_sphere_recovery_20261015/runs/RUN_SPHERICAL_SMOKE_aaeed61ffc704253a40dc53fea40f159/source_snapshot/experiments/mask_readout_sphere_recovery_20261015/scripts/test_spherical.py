import unittest
import torch
from spherical_model import SphericalAdapter,tangent_residual,neighbor_inputs


def fixture():
    torch.manual_seed(17)
    p=torch.randn(32,8,10);p[0]=1;c=torch.zeros(2,32);c[:,0]=3
    return dict(image_id=43,proto=p,c0=c,boxes_input=torch.tensor([[0.,0.,32.,32.],[4.,0.,36.,32.]]),input_hw=[32,40],
        head_feature_shapes=[[256,4,5],[512,2,3],[512,1,2]],features=[torch.randn(256),torch.randn(512)],
        raw_locations=[{'level':0},{'level':1}],scores=torch.tensor([.9,.5]),classes=torch.tensor([0,0]),raw_ids=[10,20])


class Tests(unittest.TestCase):
    def test_zero_output_forward_backward(self):
        x=fixture()
        for k in ('c0','boxes_input','scores','classes'):x[k]=x[k][:0]
        x['raw_ids']=[];x['features']=[];x['raw_locations']=[]
        features,gate,records=neighbor_inputs(x,'cpu')
        self.assertEqual(features.shape,(0,41));self.assertEqual(gate.shape,(0,1));self.assertEqual(records,[])
        model=SphericalAdapter();d=model(x,'train:0');self.assertEqual(d.shape,(0,32));d.sum().backward()

    def test_background_and_identity_filters(self):
        x=fixture();_,h,r=neighbor_inputs(x,'cpu');self.assertEqual(h[:,0].tolist(),[0.,1.])
        x['c0']=-x['c0'];_,h,_=neighbor_inputs(x,'cpu');self.assertFalse(h.any())
        x=fixture();x['raw_ids']=[10,10];_,h,r=neighbor_inputs(x,'cpu');self.assertFalse(h.any())

    def test_tangent_radius_and_boundary_not_fixed(self):
        c=torch.tensor([[1.,0.],[0.,2.]]);raw=torch.tensor([[0.,1.],[2.,1.]])
        delta=tangent_residual(c,raw,torch.ones(2,1));self.assertTrue(torch.allclose((c+delta).norm(dim=1),c.norm(dim=1),atol=1e-6))
        p=torch.tensor([[.2,-1.],[-1.,0.]])
        self.assertFalse(torch.equal((c[0]@p)>0,((c+delta)[0]@p)>0))
        self.assertTrue(torch.equal(tangent_residual(c,raw,torch.zeros(2,1)),torch.zeros_like(c)))

    def test_zero_initialization_retains_learning_gradient(self):
        model=SphericalAdapter();x=fixture();d=model(x,'train:0');self.assertTrue(torch.equal(d,torch.zeros_like(d)))
        d[1,1].backward()
        self.assertGreater(float(model.readout[-1].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.neighbor[-1].weight.grad.abs().sum()),0)
        self.assertEqual(sum(p.numel() for p in model.parameters()),172705)

    def test_gt_free_forward_and_no_base_gradients(self):
        model=SphericalAdapter();x=fixture();model.neighbor[-1].bias.data[1]=.2
        a=model(x,'eval:0');x['annotations']='must never be read';x['oracle_coefficients']=torch.full_like(x['c0'],100)
        b=model(x,'eval:0');self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(a[0],torch.zeros(32)))
        self.assertFalse(x['proto'].requires_grad or x['c0'].requires_grad)


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
