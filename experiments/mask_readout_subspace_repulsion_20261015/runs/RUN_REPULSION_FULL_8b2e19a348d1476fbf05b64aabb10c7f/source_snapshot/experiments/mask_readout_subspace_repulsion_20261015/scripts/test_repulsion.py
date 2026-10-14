import unittest
import numpy as np
import torch
from repulsion import choose_neighbors,update_coefficients


def cache():
    p=torch.ones(2,8,8);p[1]=torch.arange(8)[None,:]/8
    return dict(proto=p,c0=torch.tensor([[2.,1.],[1.,2.]]),boxes_input=torch.tensor([[12.,0.,32.,32.],[0.,0.,24.,32.]]),
                scores=torch.tensor([.9,.5]),classes=torch.tensor([0,0]),raw_ids=[10,20],input_hw=[32,32])


class Tests(unittest.TestCase):
    def test_identity_filters(self):
        b=[[0,0,10,10]]*4
        j,_=choose_neighbors(b,[.9,.8,.7,.6],[0,0,1,0],[10,10,30,40])
        self.assertEqual(j,[-1,-1,-1,0])
        j,_=choose_neighbors([[0,0,10,10],[20,20,30,30]],[.9,.5],[0,0],[10,20]);self.assertEqual(j,[-1,-1])

    def test_empty_identity(self):
        a=cache();a['raw_ids']=[10,10];new,r=update_coefficients(a,'cpu')
        self.assertTrue(torch.equal(new,a['c0']));self.assertFalse(any(x['applied'] for x in r))
        a=cache();a['boxes_input'][1]=a['boxes_input'][0];new,r=update_coefficients(a,'cpu')
        self.assertTrue(torch.equal(new,a['c0']))

    def test_explicit_gram(self):
        a=cache();new,r=update_coefficients(a,'cpu');self.assertTrue(r[1]['applied'])
        points=[]
        for y in range(8):
            for x in range(8):
                xx,yy=4*x,4*y
                if 12<=xx<24 and 0<=yy<32 and not(6<=xx<18 and 8<=yy<24) and (xx-22)**2<(xx-12)**2:
                    points.append(a['proto'][:,y,x].numpy())
        P=np.asarray(points,dtype=np.float64);ci=a['c0'][1].numpy();cj=a['c0'][0].numpy()
        G=[[sum(float(row[k])*float(row[l]) for row in P)/len(P) for l in range(2)] for k in range(2)]
        cross=sum(float(ci[k])*G[k][l]*float(cj[l]) for k in range(2) for l in range(2))
        energy=sum(float(cj[k])*G[k][l]*float(cj[l]) for k in range(2) for l in range(2))
        alpha=cross/(energy+1e-8)
        np.testing.assert_allclose(new[1].numpy(),ci-alpha*cj,atol=2e-6)
        self.assertTrue(torch.equal(new[0],a['c0'][0]));self.assertLess(abs(r[1]['correlation_after']),2e-6)

    def test_norm_does_not_lock_boundary(self):
        p=torch.tensor([[1.,-1.],[0.,0.]]);c=torch.tensor([1.,0.])
        self.assertTrue(torch.equal(c@p>0,(7*c)@p>0))
        rotated=torch.tensor([-1.,0.]);self.assertEqual(c.norm(),rotated.norm())
        self.assertFalse(torch.equal(c@p>0,rotated@p>0))


if __name__=='__main__':unittest.main()
