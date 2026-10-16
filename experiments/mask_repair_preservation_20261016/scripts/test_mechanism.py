"""Synthetic engineering tests; no COCO examples, no model/checkpoint loading."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import traceback
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'experiments/mask_repair_preservation_20261016'
sys.path.insert(0,str(ROOT/'experiments/mask_readout_mechanism_v2_20261012/scripts'))
from objectives import groups, objective, solve, equal_scale
from native_geometry import lifted_basis, target_grid, roi_for, pixel_counts
from analysis_stats import summarize

CFG=json.loads((STUDY/'CONFIG.json').read_text(encoding='utf8'))
torch.set_num_threads(1)


class MechanismTests(unittest.TestCase):
    def test_gradients_all_three_objectives(self):
        p=torch.tensor([[1.,.2],[-.5,1.],[.2,-.1],[.7,.3]],dtype=torch.float64)
        z=torch.tensor([-.5,.8,-.4,.6],dtype=torch.float64)
        y=torch.tensor([1.,1.,0.,0.],dtype=torch.float64)
        for arm in ('C','E','EK'):
            d=torch.tensor([.12,-.17],dtype=torch.float64,requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(lambda x:objective(x,p,z,y,groups(z,y),arm,CFG['solver']), (d,), eps=1e-6,atol=1e-5,rtol=1e-4))

    def test_known_nonzero_error_and_protected_optima(self):
        p=torch.tensor([[1.],[-1.]],dtype=torch.float64)
        z=torch.tensor([-.5,.2],dtype=torch.float64)
        y=torch.ones(2,dtype=torch.float64)
        for arm, expected in [('E',3/(2+.003)),('EK',3/(4+.003))]:
            d,receipt=solve(p,z,y,arm,CFG['solver'])
            self.assertEqual(receipt['solver_status'],'converged')
            self.assertAlmostEqual(float(d[0]),expected,places=5)
        # Deleting protection must change the known optimum, not just rename a loss.

    def test_known_binary_BCE_optimum(self):
        p=torch.ones((1,1),dtype=torch.float64);z=torch.tensor([-.3],dtype=torch.float64);y=torch.ones(1,dtype=torch.float64)
        lo,hi=0.,10.
        for _ in range(100):
            mid=(lo+hi)/2
            grad=1/(1+np.exp(-(-.3+mid)))-1+.003*mid
            if grad>0:hi=mid
            else:lo=mid
        d,r=solve(p,z,y,'C',CFG['solver'])
        self.assertEqual(r['solver_status'],'converged')
        self.assertAlmostEqual(float(d[0]),(lo+hi)/2,places=4)

    def test_correct_pixels_zero_penalty_and_fixed_groups(self):
        z=torch.tensor([0.,0.,-.5,1.],dtype=torch.float64);y=torch.tensor([0.,1.,0.,1.],dtype=torch.float64)
        g=groups(z,y)
        self.assertEqual(g[1].tolist(),[False,True,False,False])
        d=torch.zeros(4,dtype=torch.float64);p=torch.eye(4,dtype=torch.float64)
        self.assertEqual(float(objective(d,p,z,y,g,'EK',CFG['solver'])-objective(d,p,z,y,g,'E',CFG['solver'])),0.)

    def test_empty_groups_and_empty_roi_keep_rows(self):
        for z,y in [(torch.tensor([1.]),torch.tensor([1.])),(torch.tensor([-1.]),torch.tensor([1.])),(torch.empty(0),torch.empty(0))]:
            p=torch.ones((len(z),2),dtype=torch.float64)
            for arm in ('C','E','EK'):
                d,r=solve(p,z,y,arm,CFG['solver'])
                self.assertTrue(torch.isfinite(d).all())
                self.assertEqual(r['roi_pixels'],len(z))

    def test_equal_direction_zero_and_amplification(self):
        self.assertEqual(equal_scale(0,0),(0.,'both_zero'))
        self.assertEqual(equal_scale(0,1),(0.,'undefined_zero_direction'))
        self.assertEqual(equal_scale(2,3),(1.5,'defined'))

    def test_decoder_operator_and_odd_padding(self):
        from ultralytics.utils import ops
        p=torch.arange(3*3*4,dtype=torch.float32).reshape(3,3,4)/30-.5
        cache=dict(proto=p,input_hw=[12,16],native_decode_hw=[12,16],original_hw=[9,15],
                   ratio_pad=((1.,1.),(0.,1.)),boxes_input=torch.tensor([[1.5,2.,13.1,11.]]))
        basis=lifted_basis(cache)
        c=torch.tensor([.2,-.6,.7],dtype=torch.float64)
        direct=torch.nn.functional.interpolate((c@p.double().flatten(1)).reshape(1,1,3,4),(12,16),mode='bilinear',align_corners=False)[0,0]
        self.assertTrue(torch.allclose(c@basis.flatten(1),direct.flatten(),atol=1e-12,rtol=1e-12))
        truth=np.ones((9,15),dtype=np.uint8);gt=target_grid(truth,cache)
        self.assertEqual(int(gt.sum()),135)
        self.assertEqual(list(gt.shape),[12,16])
        roi=roi_for(cache,0)
        cropped=ops.crop_mask(torch.ones((1,12,16),dtype=torch.uint8),cache['boxes_input'])[0].bool()
        self.assertTrue(torch.equal(roi,cropped))

    def test_native_final_byte_is_not_half_threshold(self):
        from ultralytics.utils import ops
        x=torch.tensor([[[[0,1],[1,0]]]],dtype=torch.uint8)
        continuous=ops.scale_masks(x,(5,5),padding=False)
        self.assertFalse(torch.equal(continuous.byte(),(continuous>.5).byte()))

    def test_pixel_transition_counts(self):
        native=torch.tensor([[1,1,0,0]],dtype=torch.uint8);changed=1-native
        counts=pixel_counts(native,changed,torch.tensor([[1,0,1,0]]),torch.ones((1,4),dtype=torch.bool))
        for v in counts.values():self.assertEqual(v,1)

    def test_independent_RLE_decoder(self):
        from pycocotools import mask as masks
        from verify import decode
        rng=np.random.default_rng(1729)
        for binary in (np.zeros((9,13),dtype=np.uint8),np.ones((11,17),dtype=np.uint8),
                       (rng.random((37,29))>.4).astype(np.uint8)):
            encoded=masks.encode(np.asfortranarray(binary))
            encoded['counts']=encoded['counts'].decode('ascii')
            self.assertTrue(np.array_equal(decode(encoded),binary.astype(bool)))

    def test_image_macro_and_denominator_gate(self):
        def row(i,j,value):
            arms={a:dict(original_iou=value,tp_to_fn=0,solver=dict(solver_status='converged')) for a in ('N','C','E','EK','E_equal','R_G','R_S1')}
            return dict(image_id=i,output_detection_id=str(j),arms=arms,native_tp=1,numerical_valid=True)
        r=summarize([row(1,1,0),row(1,2,0),row(2,3,1)],CFG)
        self.assertEqual(r['metrics']['N']['original_iou']['mean'],.5)
        self.assertFalse(r['denominator_complete'])
        self.assertEqual(r['decision'],'INCONCLUSIVE_NUMERICAL')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MechanismTests))
    files=['CONFIG.json','scripts/objectives.py','scripts/native_geometry.py','scripts/analysis_stats.py','scripts/test_mechanism.py','scripts/verify.py']
    report=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        failures=[text for _,text in result.failures],errors=[text for _,text in result.errors],
        python=sys.version,torch=str(torch.__version__),scope='synthetic engineering only; no scientific outcomes',
        tested_sources={str((STUDY/p).relative_to(ROOT)).replace('\\','/'):hashlib.sha256((STUDY/p).read_bytes()).hexdigest() for p in files})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
    raise SystemExit(0 if result.wasSuccessful() else 1)
