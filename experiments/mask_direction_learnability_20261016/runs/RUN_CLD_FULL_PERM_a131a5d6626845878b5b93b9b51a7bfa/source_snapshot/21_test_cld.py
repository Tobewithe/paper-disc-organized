"""Behavioral and derivative checks for the new scientific operations."""
import unittest
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from cld_common import canonical
from cld_math import loss, functional_direction, remap, probe_coefficients
from cld_statistics import effects, summarize


class MathChecks(unittest.TestCase):
    def setUp(self):
        self.p = torch.tensor([[1., 2., -1., .5], [.5, -1., 2., 1.]], dtype=torch.double)
        self.c = torch.tensor([[.1, -.2], [.3, .2]], dtype=torch.double)
        self.roi = torch.tensor([[[True, True], [False, True]], [[False, True], [True, False]]])
        self.ctx = dict(p=self.p, c0=self.c, baseline=(self.c @ self.p).reshape(2, 2, 2), roi=self.roi, shape=(2, 2))

    def test_zero_initial_gradient(self):
        d = torch.zeros_like(self.c, requires_grad=True)
        value, _ = loss(d, self.ctx, {0: torch.ones(2, 2, dtype=torch.double)}, 2.)
        value.backward()
        self.assertTrue(torch.isfinite(d.grad).all())
        self.assertGreater(float(d.grad[0].abs().sum()), 0.)

    def test_grouped_manual_formula(self):
        d = torch.tensor([[.2, -.1], [.4, .1]], dtype=torch.double, requires_grad=True)
        targets = {0: torch.ones(2, 2, dtype=torch.double), 1: torch.zeros(2, 2, dtype=torch.double)}
        value, info = loss(d, self.ctx, targets, 1.3)
        r = (d @ self.p).reshape(2, 2, 2)
        manual_direction = torch.stack([(r[i][self.roi[i]] - targets[i][self.roi[i]]).square().mean() for i in targets]).mean()
        v = torch.stack([d[i] / max(1., float(r[i][self.roi[i]].square().mean().sqrt())) for i in range(2)])
        logits = ((self.c + 1.3 * v) @ self.p).reshape_as(r)
        manual_preserve = torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(logits[i][self.roi[i]], self.ctx['baseline'][i][self.roi[i]].sigmoid()) for i in range(2)]).mean()
        torch.testing.assert_close(value, manual_direction + .1 * manual_preserve, atol=1e-12, rtol=1e-12)
        self.assertEqual(info['teachers'], 2)

    def test_gradcheck_clipped_and_unclipped(self):
        for magnitude in (.01, 2.):
            d = (torch.tensor([[.2, -.1], [.4, .1]], dtype=torch.double) * magnitude).requires_grad_()
            self.assertTrue(torch.autograd.gradcheck(lambda x: loss(x, self.ctx, {0: torch.ones(2, 2, dtype=torch.double)}, 1.3)[0], (d,), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_empty_roi_and_no_teacher(self):
        self.ctx['roi'] = torch.zeros_like(self.roi)
        d = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        value, info = loss(d, self.ctx, {}, 1.)
        value.backward()
        self.assertEqual(float(value), 0.)
        self.assertTrue(torch.isfinite(d.grad).all())
        self.assertEqual(info['empty_roi_rows'], 2)
        self.assertEqual(int(torch.count_nonzero(functional_direction(d, self.ctx)[1])), 0)

    def test_permutation_gradient_routes_to_donor(self):
        d = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        mapped = remap(d, [1, 0], 'PERM')
        mapped[0].sum().backward()
        torch.testing.assert_close(d.grad, torch.tensor([[0., 0.], [1., 1.]]))
        torch.testing.assert_close(remap(d, [1, 0], 'J'), mapped)
        torch.testing.assert_close(remap(d[:1], [0], 'PERM'), d[:1])

    def test_equal_amplitude_and_zero_fallback(self):
        d = torch.tensor([[.2, -.1], [0., 0.]], dtype=torch.double)
        result = probe_coefficients(d, self.ctx, {0: 2., 1: 3.})
        actual = ((result - self.c) @ self.p).reshape(2, 2, 2)
        self.assertAlmostEqual(float(actual[0][self.roi[0]].square().mean().sqrt()), 2., places=12)
        torch.testing.assert_close(result[1], self.c[1])

    def test_bounded_response_does_not_amplify_small_direction(self):
        d = torch.tensor([[.01, -.02], [20., -10.]], dtype=torch.double)
        _, v, square_mean, _ = functional_direction(d, self.ctx)
        torch.testing.assert_close(v[0], d[0])
        bounded_response = (v @ self.p).reshape(2, 2, 2)
        self.assertAlmostEqual(float(bounded_response[1][self.roi[1]].square().mean()), 1., places=12)
        self.assertGreater(float(square_mean[1]), 1.)


class StatisticalChecks(unittest.TestCase):
    def test_macro_not_instance_weighted_and_shared_draws(self):
        rows=[dict(image_id=1,M=1.,N=0.,G=.5)] + [dict(image_id=2,M=0.,N=0.,G=-.5)]*9
        result, ids, draws=effects(rows,[('M','N'),('M','G')])
        self.assertEqual(result['M-N']['mean'],.5)
        self.assertEqual(result['M-G']['ci95'],[.5,.5])
        self.assertEqual(ids,[1,2])
        self.assertEqual(draws.shape,(2000,2))

    def test_three_decision_exits_and_independent_arithmetic(self):
        from verify_evaluation import statistical_check
        template=dict(N=.4,C=.8,R=.6,P=.5,J=.5,zero_R=False,zero_P=False,zero_J=False,cosine_R=.7,cosine_P=.4,cosine_J=.4)
        direction=[dict(template,split='heldout',image_id=i%208,gate=i<62) for i in range(581)]
        direction += [dict(template,split='fit',image_id=1000+i%867,gate=False) for i in range(2510)]
        pairs=[dict(image_id=i%203,population='T',M=.6,N=.5,G=.55,S=.58) for i in range(490)]
        pairs += [dict(image_id=i%394,population='S',M=.6,N=.5,G=.55,S=.58) for i in range(1756)]
        tasks=[dict(N=True,M=True,S=True,G=True) for _ in range(1808)]
        for case in ('positive','damage','inconclusive'):
            ds,ts=copy.deepcopy(direction),copy.deepcopy(tasks)
            if case=='damage':
                for r in ts[:19]: r['M']=False
            if case=='inconclusive':
                for r in ds:
                    if r['split']=='heldout': r['P']=r['R']-(.005+(.2 if r['image_id']%2 else -.2))
            summary,draws,ids=summarize(ds,pairs,ts)
            expected={'positive':'POSITIVE_INTERNAL_SCREEN','damage':'NEGATIVE_AT_BUDGET','inconclusive':'INCONCLUSIVE_AT_BUDGET'}[case]
            self.assertEqual(summary['decision'],expected)
            with tempfile.TemporaryDirectory(prefix='cld_math_test_') as directory:
                path=Path(directory)
                (path/'SUMMARY.json').write_text(json.dumps(summary),encoding='utf-8')
                (path/'STATISTICAL_IMAGE_ORDER.json').write_text(json.dumps(ids),encoding='utf-8')
                np.savez(path/'BOOTSTRAP_INDICES.npz',**draws)
                self.assertEqual(statistical_check(path,ds,pairs,ts)['decision'],expected)


if __name__ == '__main__':
    unittest.main()
