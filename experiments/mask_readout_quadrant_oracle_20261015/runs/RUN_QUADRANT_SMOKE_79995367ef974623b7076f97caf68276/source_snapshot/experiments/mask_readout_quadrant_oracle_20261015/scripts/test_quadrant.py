import unittest
import torch
from quadrant import partition,active_quadrants,positive_control


class Tests(unittest.TestCase):
    def test_partition_and_uniform_delta(self):
        cache=dict(proto=torch.randn(32,8,10),input_hw=[32,40],boxes_input=torch.tensor([[0.,0.,32.,32.],[24.,0.,40.,8.]]))
        hard,soft=partition(cache,0);self.assertTrue(torch.allclose(soft.sum(0),torch.ones(8,10,dtype=torch.float64)))
        delta=torch.randn(32,dtype=torch.float64);p=cache['proto'].double();response=torch.einsum('k,khw->hw',delta,p)
        self.assertTrue(torch.allclose((soft*response[None]).sum(0),response,atol=1e-12,rtol=0))
        self.assertEqual(active_quadrants(cache,0,1),[1]);self.assertEqual(active_quadrants(cache,0,-1),[])

    def test_solver_known_nonzero(self):self.assertTrue(positive_control()['passed'])


if __name__=='__main__':torch.set_num_threads(1);unittest.main()
