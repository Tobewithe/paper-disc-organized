"""Behavioral checks of target substitution, denominator preservation and fail-closed pins."""
import tempfile
from pathlib import Path
import unittest
import torch
from common import write_new,sha256
from targets import TeacherTargets,mask_hash,accept
from soft_iou_loss import soft_iou_grouped_loss

class TargetTests(unittest.TestCase):
    def fixture(self,root,replace=True):
        q=torch.tensor([[1,0],[0,1]],dtype=torch.uint8)
        pack=dict(image_id=7,prediction_indices=[21,22],join={0:10,1:11},
            targets={0:q} if replace else {},target_hashes={0:mask_hash(q)} if replace else {})
        path=root/'7.pt';torch.save(pack,path)
        write_new(root/'TEACHER_MANIFEST.json',dict(images={'7':dict(path='7.pt',sha256=sha256(path))}))
        original=lambda coco,cache,join: {k:torch.full((2,2),float(k)) for k in join}
        provider=TeacherTargets(root,original)
        cache=dict(image_id=7,prediction_indices=[21,22],proto=torch.zeros(32,2,2))
        return provider,cache,q
    def test_only_accepted_targets_replace(self):
        with tempfile.TemporaryDirectory() as d:
            p,c,q=self.fixture(Path(d))
            t=p(None,c,{0:10,1:11})
            self.assertEqual(set(t),{0,1})
            self.assertTrue(torch.equal(t[0],q.float()))
            self.assertTrue(torch.equal(t[1],torch.ones(2,2)))
            self.assertFalse(t[0].requires_grad)
    def test_zero_replacement_is_exact_original_loss_and_gradients(self):
        with tempfile.TemporaryDirectory() as d:
            p,c,_=self.fixture(Path(d),False)
            t=p(None,c,{0:10,1:11})
            x=torch.arange(8).float().reshape(2,2,2).requires_grad_()
            base=torch.zeros_like(x);roi=torch.ones_like(x,dtype=torch.bool)
            a,stats=p.loss(x,base,roi,t)
            ga=torch.autograd.grad(a,x)[0]
            b,ref=soft_iou_grouped_loss(x,base,roi,t)
            gb=torch.autograd.grad(b,x)[0]
            self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(ga,gb))
            self.assertEqual(stats['matched_supervised_instances'],2)
            self.assertEqual(stats['preserved_instances'],2)
    def test_preservation_and_denominators_unchanged_after_substitution(self):
        with tempfile.TemporaryDirectory() as d:
            p,c,_=self.fixture(Path(d))
            t=p(None,c,{0:10,1:11})
            x=torch.tensor([[[1.,-1.],[2.,-2.]],[[1.,2.],[3.,4.]]],requires_grad=True)
            base=x.detach()/2;roi=torch.ones_like(x,dtype=torch.bool)
            a,stats=p.loss(x,base,roi,t)
            b,ref=soft_iou_grouped_loss(x,base,roi,{0:torch.zeros(2,2),1:torch.ones(2,2)})
            self.assertEqual(stats['preservation_loss'],ref['preservation_loss'])
            self.assertEqual(stats['preserved_instances'],ref['preserved_instances'])
            self.assertEqual(stats['matched_supervised_instances'],ref['matched_supervised_instances'])
            self.assertNotEqual(a.item(),b.item())
            self.assertTrue(torch.isfinite(torch.autograd.grad(a,x)[0]).all())
    def test_wrong_join_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p,c,_=self.fixture(Path(d))
            with self.assertRaises((ValueError,AssertionError,RuntimeError)):
                p(None,c,{0:11,1:10})
    def test_changed_teacher_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p,c,_=self.fixture(Path(d))
            with (Path(d)/'7.pt').open('ab') as f:f.write(b'changed')
            with self.assertRaises((ValueError,AssertionError,RuntimeError)):
                p(None,c,{0:10,1:11})
    def test_fail_closed_acceptance(self):
        self.assertTrue(accept(.2,.3,'converged',1))
        self.assertFalse(accept(.2,.205,'converged',1))
        self.assertFalse(accept(.2,.3,'budget_exhausted',1))
        self.assertFalse(accept(.2,.3,'converged',0))
    def test_hard_function_target_is_realized_by_its_coefficients(self):
        proto=torch.tensor([[1.,0.,-1.,0.],[0.,1.,0.,-1.]])
        c=torch.tensor([[2.,-3.]])
        q=(c@proto).gt(0).float().reshape(1,2,2)
        self.assertTrue(torch.equal(q,torch.tensor([[[1.,0.],[0.,1.]]])))
        z=(20*c@proto).reshape(1,2,2)
        loss,_=soft_iou_grouped_loss(z,z,torch.ones_like(z,dtype=torch.bool),{0:q[0]})
        self.assertLess(loss.item(),1e-6)

if __name__=='__main__':
    unittest.main()
