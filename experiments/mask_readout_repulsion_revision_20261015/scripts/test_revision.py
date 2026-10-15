import unittest
import torch
from revision import update


def fixture():
    p=torch.ones(2,8,8);p[1]=torch.arange(8)[None,:]/8
    return dict(proto=p,c0=torch.tensor([[3.,1.],[1.,.5]]),boxes_input=torch.tensor([[12.,0.,32.,32.],[0.,0.,24.,32.]]),
                scores=torch.tensor([.9,.5]),classes=torch.tensor([0,0]),raw_ids=[10,20],input_hw=[32,32])


class Tests(unittest.TestCase):
    def test_shared_background_and_no_neighbor_identity(self):
        x=fixture();x['c0']=-x['c0'];new,r=update(x,'cpu')
        self.assertTrue(r[1]['old_applied']);self.assertGreater(r[1]['old_negative_negative'],0)
        self.assertEqual(r[1]['conflict_pixels'],0)
        for v in new.values():self.assertTrue(torch.equal(v,x['c0']))
        x=fixture();x['raw_ids']=[10,10];new,r=update(x,'cpu')
        self.assertFalse(any(v['applied'] for v in r))

    def test_reference_scalar_geometry_and_two_dimensional_solve(self):
        x=fixture();new,rows=update(x,'cpu');r=rows[1];self.assertTrue(r['applied'])
        points=[];positive=[]
        for y in range(8):
            for xx in range(6):
                p=[float(v) for v in x['proto'][:,y,xx]];points.append(p)
                omega=12<=4*xx<24 and not(6<=4*xx<18 and 8<=4*y<24) and (4*xx-22)**2<(4*xx-12)**2
                positive.append(omega)
        ci=x['c0'][1].tolist();cj=x['c0'][0].tolist()
        zi=[sum(a*b for a,b in zip(ci,p)) for p in points];zj=[sum(a*b for a,b in zip(cj,p)) for p in points]
        positive=[q and a>0 and b>a for q,a,b in zip(positive,zi,zj)]
        count=sum(positive);self.assertEqual(count,r['conflict_pixels'])
        alpha=(sum(a*b for a,b,q in zip(zi,zj,positive) if q)/count)/(sum(b*b for b,q in zip(zj,positive) if q)/count+1e-8)
        self.assertAlmostEqual(alpha,r['alpha'],places=6)
        target=[-alpha*z if q else 0 for z,q in zip(zj,positive)];m=len(points)
        g=[[sum(p[k]*p[l] for p in points)/m for l in range(2)] for k in range(2)]
        ridge=.01*(g[0][0]+g[1][1])/2;g[0][0]+=ridge;g[1][1]+=ridge
        rhs=[sum(p[k]*t for p,t in zip(points,target))/m for k in range(2)]
        det=g[0][0]*g[1][1]-g[0][1]*g[1][0]
        delta=[(rhs[0]*g[1][1]-rhs[1]*g[0][1])/det,(g[0][0]*rhs[1]-g[1][0]*rhs[0])/det]
        for a,b in zip(new['B'][1].tolist(),[v+d for v,d in zip(ci,delta)]):self.assertAlmostEqual(a,b,places=5)
        for a,b in zip(new['A'][1].tolist(),[v-alpha*d for v,d in zip(ci,cj)]):self.assertAlmostEqual(a,b,places=5)
        self.assertTrue(torch.equal(new['A'][0],x['c0'][0]));self.assertTrue(torch.equal(new['B'][0],x['c0'][0]))
        self.assertLessEqual(r['objective_B'],r['objective_A']+1e-6)

    def test_minimum_conflict_pixels(self):
        for number in (3,4):
            x=fixture();x['proto']=-torch.ones(2,8,8)
            for y in range(number):x['proto'][:,y,5]=1
            new,r=update(x,'cpu');self.assertEqual(r[1]['conflict_pixels'],number)
            self.assertEqual(r[1]['applied'],number==4)
            if number==3:self.assertTrue(torch.equal(new['B'],x['c0']))

    def test_inverse_gram_does_not_preserve_background_sign(self):
        # Four conflicting foreground points and four background points; ridge is not a hard protection.
        p=torch.tensor([[1.,0.],[-.1,1.]],dtype=torch.float64).repeat_interleave(4,0)
        c=torch.tensor([1.,.09],dtype=torch.float64);neighbor=torch.tensor([2.,.2],dtype=torch.float64)
        z=p@c;zj=p@neighbor;fg=(z>0)&(zj>z);self.assertEqual(int(fg.sum()),4)
        delta=-.5*torch.linalg.solve(p.T@p/8+.1*torch.eye(2,dtype=torch.float64),p[fg].T@zj[fg].sigmoid()/4)
        self.assertLess(float(z[4]),0);self.assertGreater(float((p@(c+delta))[4]),0)


if __name__=='__main__':unittest.main()
