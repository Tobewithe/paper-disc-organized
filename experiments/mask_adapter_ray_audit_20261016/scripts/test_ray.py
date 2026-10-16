"""Synthetic checks only: no real sample amplitude selection before freezing."""
import argparse,json,unittest
import numpy as np
from ray_common import ROOT,STUDY,sha256,write_new,now,configure_native
from ray_geometry import conversion_table,lookup,build_sweep,curve,candidates,predict,binary_iou,choose_actual
from analysis import summarize


class Checks(unittest.TestCase):
    def test_byte_conversion_patterns_and_random(self):
        import torch
        from ultralytics.utils import ops
        rng=np.random.default_rng(81)
        shapes=[((5,7),(13,17),(0,0)),((640,640),(375,500),(0,0)),((426,640),(427,640),(0,0)),
                ((640,480),(1024,768),(0,0)),((17,23),(11,15),(2,3)),((7,9),(7,9),(1,1))]
        for shape,out,pad in shapes:
            cache=dict(native_decode_hw=shape,original_hw=out,ratio_pad=((1.,1.),pad))
            transform=conversion_table(cache)
            data=np.stack([rng.random(shape)>.1,rng.random(shape)>.5,np.ones(shape,bool),np.zeros(shape,bool)])
            actual=ops.scale_masks(torch.tensor(data.astype(np.uint8),device='cuda')[None],out,ratio_pad=cache['ratio_pad'])[0].byte().cpu().numpy()
            for mask,result in zip(data,actual,strict=True):self.assertTrue(np.array_equal(lookup(mask,transform),result),(shape,out,pad))

    def test_sweep_every_interval_against_direct_masks(self):
        rng=np.random.default_rng(1729)
        for shape,out in [((7,9),(11,13)),((9,13),(6,8)),((8,8),(8,8))]:
            transform=conversion_table(dict(native_decode_hw=shape,original_hw=out,ratio_pad=((1.,1.),(0,0))))
            for _ in range(3):
                z=rng.normal(size=shape);d=rng.normal(size=shape);roi=rng.random(shape)>.15;gt=rng.random(out)>.5
                sweep=build_sweep(z,d,roi,gt,transform,16.);alpha,scores=curve(sweep)
                direct=[binary_iou(predict(z,d,roi,a,transform),gt) for a in alpha]
                np.testing.assert_allclose(scores,direct,rtol=0,atol=1e-15)

    def test_zero_direction_and_empty_roi(self):
        transform=conversion_table(dict(native_decode_hw=(4,4),original_hw=(4,4),ratio_pad=((1,1),(0,0))))
        z=np.ones((4,4));d=np.zeros_like(z);gt=z.astype(bool)
        for roi in (gt,~gt):
            sweep=build_sweep(z,d,roi,gt,transform,16.)
            self.assertEqual(candidates(sweep,dict(top_candidates=6,fixed_candidates=[0.,1.,16.])),[0.,1.,16.])
            self.assertEqual(len(sweep['times']),0)

    def test_simultaneous_events_and_zero_logits(self):
        transform=conversion_table(dict(native_decode_hw=(4,4),original_hw=(5,7),ratio_pad=((1,1),(0,0))))
        z=np.array([[0,1,-1,2]]*4,dtype=float);d=np.array([[1,-1,1,-2]]*4,dtype=float)
        roi=np.ones((4,4),bool);gt=np.indices((5,7)).sum(axis=0)%2==0
        sweep=build_sweep(z,d,roi,gt,transform,16.);alpha,scores=curve(sweep)
        for a,score in zip(alpha,scores,strict=True):self.assertEqual(score,binary_iou(predict(z,d,roi,a,transform),gt))

    def test_candidate_budget_endpoints_and_ties(self):
        cfg=dict(top_candidates=6,fixed_candidates=[0.,1.,16.])
        sweep=dict(times=np.arange(1.,10.),delta_area=np.ones(9,int),delta_intersection=np.ones(9,int),initial_area=0,initial_intersection=0,gt_area=10,alpha_max=16.,zero_direction=False)
        result=candidates(sweep,cfg)
        self.assertLessEqual(len(result),9);self.assertTrue({0.,1.,16.}.issubset(result))
        self.assertEqual(choose_actual([dict(alpha=a,original_iou=.5) for a in (0.,16.,1.)]),2)

    def test_statistics_and_decision_exits(self):
        cfg=json.loads((STUDY/'CONFIG.json').read_text());cfg['panel'].update(pairs=12,images=6)
        def rows(value):
            return [dict(image_id=i//2,output_detection_id=str(i),numerical_valid=True,arms={
                a:dict(original_iou=v,alpha=2.) for a,v in dict(N=.4,U=.5,G=.45,C=.9,R=value).items()}) for i in range(12)]
        self.assertEqual(summarize(rows(.85),cfg,True)['decision'],'AMPLITUDE_PRIORITY')
        self.assertEqual(summarize(rows(.55),cfg,True)['decision'],'DIRECTION_PRIORITY')
        self.assertEqual(summarize(rows(.75),cfg,True)['decision'],'MIXED_DIRECTION_AMPLITUDE')
        bad=rows(.85);bad[0]['numerical_valid']=False
        self.assertEqual(summarize(bad,cfg,True)['decision'],'INCONCLUSIVE_NUMERICAL')
        bad=rows(.85)
        for row in bad:row['arms']['C']['original_iou']=.5
        self.assertEqual(summarize(bad,cfg,True)['decision'],'INSUFFICIENT_REFERENCE')
        self.assertEqual(summarize(rows(.85),cfg,False)['decision'],'ENGINEERING_ONLY')

    def test_independent_RLE_and_coefficient_rounding(self):
        import torch
        from verify_ray import iou,intervals
        x=dict(size=[2,3],counts='0220');y=dict(size=[2,3],counts='123')
        self.assertEqual(sum(b-a for a,b in intervals(x)),4);self.assertEqual(iou(x,y),.2)
        a=torch.tensor([1.3,-3.1,0.001],dtype=torch.float32);d=torch.tensor([-.03,2.1,.009],dtype=torch.float32)
        for alpha in (0.,1.,.75,16.):
            actual=(a.double()+alpha*d.double()).float().numpy()
            expected=(a.numpy().astype(np.float64)+alpha*d.numpy().astype(np.float64)).astype(np.float32)
            self.assertTrue(np.array_equal(actual,expected))
        self.assertTrue(torch.equal(a+d,(a.double()+d.double()).float()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    configure_native()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    files=['scripts/ray_geometry.py','scripts/analysis.py','scripts/test_ray.py','scripts/verify_ray.py']
    write_new(STUDY/args.output,dict(status='passed' if result.wasSuccessful() else 'failed',at=now(),tests=result.testsRun,
        failures=[(str(t),trace) for t,trace in result.failures+result.errors],synthetic_only=True,
        tested_sources={str((STUDY/f).relative_to(ROOT)).replace('\\','/'):sha256(STUDY/f) for f in files}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
